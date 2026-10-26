#ifndef CONFORMER_ENSEMBLE_HYBRID_RANKING_HPP
#define CONFORMER_ENSEMBLE_HYBRID_RANKING_HPP

#include "pose_score_ranking.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace conformer_ensemble {

struct EnsemblePose
{
    std::size_t record_index{};
    std::string conformer_id;
    RescoredPose scored;
    double rmsd_heavy{};
};

struct RankedEnsemblePose
{
    EnsemblePose pose;
    std::size_t local_rank{};
    std::size_t global_rank{};
    double hybrid{};
    double vinardo_rep50{};
};

struct HybridEnsembleRanking
{
    // All three vectors retain scores normalized once on the complete input.
    std::vector<RankedEnsemblePose> all;
    std::vector<RankedEnsemblePose> local;
    std::vector<RankedEnsemblePose> global;
};

// Use existing Hybrid ranking with lambda=0.3 and Vinardo alpha=0.5. Supply
// generated conformers together; rank the crystallographic baseline separately.
// Ties use record_index then original pose_id. RMSD never enters selection.
// local_k >= global_x > 0 preserves the exact global Top-X after local Top-K.
HybridEnsembleRanking rank_ensemble_poses(
    std::vector<EnsemblePose> poses,
    std::size_t local_k = 10,
    std::size_t global_x = 10);

} // namespace conformer_ensemble

#endif // CONFORMER_ENSEMBLE_HYBRID_RANKING_HPP
