#ifndef CONFORMER_EXPERIMENT_DOCKING_HPP
#define CONFORMER_EXPERIMENT_DOCKING_HPP

#include "ensemble_ranking.hpp"
#include "gpu_pose_search.hpp"

struct ConformerDockingResult {
    std::size_t triplets{}, matches{}, alignments{}, valid_transforms{}, survivors{};
    double diameter{}, preparation_s{}, search_s{}, query_s{}, rescoring_s{}, rmsd_s{};
    double prefilter_s{}, weighted_interaction_s{}, vinardo_s{};
    std::vector<EnsemblePose> poses; // Every clash survivor, in original order.
    std::vector<std::size_t> rescored_ids; // Subset selected by the explicit Fast prefilter.
    std::optional<double> oracle_all_survivors, oracle_after_prefilter;
};

// One unchanged production CUDA search; all survivors are downloaded before
// optional host rescoring. RMSD evaluates every survivor but never selects one.
// Target/index ownership stays with the caller across all conformers.
ConformerDockingResult dock_conformer(
    std::size_t conformer_id, RDKit::RWMol& ligand, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter);

#endif
