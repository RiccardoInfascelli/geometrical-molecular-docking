#ifndef CONFORMER_EXPERIMENT_DOCKING_HPP
#define CONFORMER_EXPERIMENT_DOCKING_HPP

#include "ensemble_ranking.hpp"
#include "gpu_pose_search.hpp"
#include "gpu_docking.hpp"

struct ConformerDockingResult {
    std::size_t triplets{}, matches{}, alignments{}, valid_transforms{}, survivors{};
    double diameter{}, preparation_s{}, search_s{}, query_s{}, rescoring_s{}, rmsd_s{};
    double prefilter_s{}, weighted_interaction_s{}, vinardo_s{};
    std::vector<EnsemblePose> poses; // Every clash survivor, in original order.
    std::vector<std::size_t> rescored_ids; // Subset selected by the explicit Fast prefilter.
    std::optional<double> oracle_all_survivors, oracle_after_prefilter;
    bool shared_search = false, shared_rescoring = false;
};

// One unchanged production CUDA search; all survivors are downloaded before
// optional host rescoring. RMSD evaluates every survivor but never selects one.
// Target/index ownership stays with the caller across all conformers.
ConformerDockingResult dock_conformer(
    std::size_t conformer_id, RDKit::RWMol& ligand, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter);

struct ConformerExecutionOptions {
    bool gpu_batch = false;
    std::size_t batch_size = 8, batch_memory_bytes = std::size_t{2048} * 1024 * 1024;
    std::optional<bool> cpu_gpu_overlap;
    GpuRescoringBackend rescoring_backend = GpuRescoringBackend::Cpu;
    std::string query_method = "reference";
};

struct ConformerDockingInput {
    std::size_t conformer_id{};
    const RDKit::RWMol* molecule{};
};

struct ConformerExecutionTimings {
    double wall_s{}, preparation_s{}, query_s{}, search_s{}, rescoring_s{};
    double consumer_wait_s{}, producer_wait_s{};
    std::size_t adaptive_splits{}, memory_retries{}, queue_capacity{};
    std::size_t queue_high_water_records{}, queue_high_water_bytes{};
};

struct CudaRescoringTarget;

// Shared host prefetch and CUDA geometry/raw scorers. No ranking occurs here:
// all survivors remain available to RMSD and the ensemble's global ranker.
std::vector<ConformerDockingResult> dock_conformers(
    const std::vector<ConformerDockingInput>& inputs, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter,
    const ConformerExecutionOptions& options, ConformerExecutionTimings* timings = nullptr,
    const CudaRescoringTarget* device_scoring = nullptr);

#endif
