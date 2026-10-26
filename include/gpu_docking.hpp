#ifndef GPU_DOCKING_HPP
#define GPU_DOCKING_HPP

#include "cuda_pose_pipeline.hpp"
#include "run_config.hpp"
#include "gpu_docking_output.hpp"
#include "pose_score_ranking.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace RDKit { class RWMol; }

enum class GpuRescoringBackend { Cpu, Cuda };

// Completed host-wall intervals, including wrapper allocation and device waits;
// these are not isolated kernel times. Disabled collection leaves them zero.
struct GpuDockingTimings {
    double preparation_seconds = 0.0;
    double search_seconds = 0.0;
    double selection_seconds = 0.0;
    double prefilter_seconds = 0.0;
    double weighted_interaction_seconds = 0.0;
    double vinardo_seconds = 0.0;
    double chemical_preparation_seconds = 0.0;
    double rescoring_upload_seconds = 0.0;
    double hybrid_seconds = 0.0;
    double score_preparation_seconds = 0.0;
    double rescoring_download_seconds = 0.0;
};

struct GpuDockingSettings {
    double cutoff = 24.0;
    bool collect_timings = false;
    bool diagnostic_stages = false;
    std::optional<double> min_target_clearance;
    bool interaction_full_protein = false;
    std::optional<std::size_t> fast_prefilter;
    PoseScoreRankingOptions ranking;
    // Keep CPU rescoring as the reference default; CUDA is selected explicitly.
    GpuRescoringBackend rescoring_backend = GpuRescoringBackend::Cpu;
};

struct GpuDockingResult {
    std::vector<GpuPipelinePose> poses;
    // Paired with poses; original survivor IDs survive prefilter and ranking.
    std::vector<GpuDockingPoseScores> scores;
    std::size_t candidate_pool = 0;
    std::size_t ligand_triplets = 0;
    std::size_t candidate_matches = 0;
    std::size_t alignments = 0;
    std::size_t valid_transforms = 0;
    std::size_t survivors = 0;
    double diameter = 0.0;
    GpuDockingTimings timings;
    StageProfile stages;
};

// One fixed protein/pocket and shared-cutoff index uploaded once. Calls process
// ligands in order; poses of each ligand run in parallel on CUDA. Not thread safe.
// Parsing, sanitization, RMSD and export belong to the caller. CPU rescoring
// reuses rescore_poses(); CUDA rescoring retains the survivor pool on device.
// All survivors (or an explicit Fast prefilter) reach rescoring before top-k.
class GpuDockingScreen {
public:
    GpuDockingScreen(const RunConfig& config, double cutoff = 24.0,
                     bool collect_timings = false);
    GpuDockingScreen(const RunConfig& config, const GpuDockingSettings& settings);
    ~GpuDockingScreen();
    GpuDockingScreen(const GpuDockingScreen&) = delete;
    GpuDockingScreen& operator=(const GpuDockingScreen&) = delete;

    // Borrows a sanitized molecule with coordinates. k must be positive.
    // Default settings preserve descending Fast. Other methods use the selected
    // rescoring backend, never RMSD. Ties follow the original survivor ID.
    GpuDockingResult dock(RDKit::RWMol& ligand, std::size_t k);
    std::size_t index_records() const;
    double startup_seconds() const; // zero when timing collection is disabled
    std::size_t targets_before_pruning() const;
    std::size_t targets_after_pruning() const;
    std::size_t vinardo_skipped_receptor_atoms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
