#ifndef GPU_DOCKING_HPP
#define GPU_DOCKING_HPP

#include "cuda_pose_pipeline.hpp"
#include "run_config.hpp"
#include "gpu_docking_output.hpp"
#include "pose_score_ranking.hpp"

#include <cstddef>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>

namespace RDKit { class RWMol; }
struct GpuPreparedLigand;

enum class GpuRescoringBackend { Cpu, Cuda };

// Host work durations, including allocation and device waits, not kernel times.
// Prepared search_seconds adds producer query work to the completed consumer
// search span; it is not pipeline elapsed time. Nested timings are not additive.
// Disabled collection leaves durations zero and absolute boundaries unset.
struct GpuDockingTimings {
    double preparation_seconds = 0.0;
    double search_seconds = 0.0;
    double query_preparation_seconds = 0.0;
    // Search excluding query preparation; includes packing, copies, CUDA waits
    // and cleanup. It is a completed wall interval, not isolated kernel time.
    double gpu_search_seconds = 0.0;
    int query_threads_requested = 1;
    int query_threads_used = 1;
    double selection_seconds = 0.0;
    double prefilter_seconds = 0.0;
    double weighted_interaction_seconds = 0.0;
    double vinardo_seconds = 0.0;
    double chemical_preparation_seconds = 0.0;
    double rescoring_upload_seconds = 0.0;
    double hybrid_seconds = 0.0;
    double score_preparation_seconds = 0.0;
    double rescoring_download_seconds = 0.0;
    // Absolute host boundaries for explicit prepared CUDA rescoring. These
    // completed wrapper spans include host work/copies/waits, not just kernels.
    // Legacy and Fast calls leave them unset to preserve their timing scopes.
    std::chrono::steady_clock::time_point geometry_start{}, geometry_end{};
    std::chrono::steady_clock::time_point rescoring_start{}, rescoring_end{};
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

// One completed aggregate search, including packing, copies and cleanup.
// Per-record GPU elapsed times are unavailable when kernels mix their work.
struct GpuDockingBatchTimings {
    double search_seconds = 0.0;
    double rescoring_seconds = 0.0;
    GpuDockingTimings rescoring;
    // Conservative planned allocation bound, not measured peak VRAM.
    std::size_t explicit_device_bytes = 0;
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
    // Does not repeat ligand/query preparation on the CUDA owner thread. The packet
    // remains owned by the caller and must stay unchanged throughout this call.
    // As with dock(), concurrent calls on one screen are not supported.
    GpuDockingResult dock_prepared(GpuPreparedLigand& ligand, std::size_t k);
    // Aggregated geometry and CUDA scoring, returning one result per packet.
    // Prefilter and ranking/Hybrid normalization remain per-ligand calls.
    // Explicit bytes exclude the shared receptor and wrapper-internal scratch.
    // Optional errors isolate record-level selection failures; device/resource
    // failures remain fatal. Calls on this screen must remain serialized.
    std::vector<GpuDockingResult> dock_batch_prepared(
        const std::vector<GpuPreparedLigand*>& ligands, std::size_t k,
        std::size_t max_device_bytes, GpuDockingBatchTimings* timings = nullptr,
        std::vector<std::exception_ptr>* record_errors = nullptr);
    // Release retained batch scratch after a completed/failed blocking call.
    // Intended for explicit-budget retries; never call concurrently with docking.
    void release_batch_scratch() noexcept;
    std::size_t index_records() const;
    double startup_seconds() const; // zero when timing collection is disabled
    std::size_t targets_before_pruning() const;
    std::size_t targets_after_pruning() const;
    std::size_t vinardo_skipped_receptor_atoms() const;

private:
    GpuDockingResult dock_impl(RDKit::RWMol& ligand, std::size_t k,
                              const GpuPreparedLigand* prepared);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
