#ifndef LIGAND_CONFORMER_ENSEMBLE_RANKING_HPP
#define LIGAND_CONFORMER_ENSEMBLE_RANKING_HPP

#include "pose_score_ranking.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

struct EnsemblePose
{
    std::size_t conformer_id{};
    std::size_t pose_id{};
    RescoredPose scored;
    double rmsd{};
    std::size_t alignment_index{}; // Original CUDA alignment ID, independent of the survivor ordinal.
};

struct RankedEnsemblePose
{
    std::size_t source_index{};
    std::size_t local_rank{};
    std::size_t global_rank{};
    std::optional<double> hybrid;
    double score{};
};

// Rank the complete supplied ensemble with the production ranker. The caller
// supplies source order (conformer order, then pose order), after any explicitly
// requested per-conformer Fast prefilter; no implicit filtering occurs here.
// Keep the crystallographic baseline in a separate pool. Source offsets supply
// stable tie IDs, independent of conformer/pose IDs and RMSD. Vinardo uses
// alpha=0.5; Hybrid uses lambda=0.3 and one normalization over this whole pool.
// Ranks are one-based. Local rank is the conformer's position within this global
// order, using the same global Hybrid values without per-conformer normalization.
// RMSD is never read or validated by ranking. The input remains unchanged.
std::vector<RankedEnsemblePose> rank_ensemble_poses(
    const std::vector<EnsemblePose>& poses,
    PoseScoreMethod method,
    double* hybrid_normalization_seconds = nullptr);

inline constexpr std::array<std::size_t, 4> ensemble_summary_top_k{{1, 5, 10, 20}};

struct EnsembleRankingSummary
{
    std::optional<double> oracle_rmsd;
    std::optional<std::size_t> oracle_source_index;
    std::array<std::optional<double>, 4> top_k_best_rmsd{};
};

// Evaluation only: the oracle is the minimum RMSD over the entire supplied
// pool, independent of method or score. Top-k entries are minima over the first
// min(k,N) ranked rows. Empty pools have no metrics; oracle ties use source order.
// Require finite, nonnegative RMSDs and a complete ranking permutation in global
// rank order; validation never changes ranking or includes a separate baseline.
EnsembleRankingSummary summarize_ensemble_ranking(
    const std::vector<EnsemblePose>& poses,
    const std::vector<RankedEnsemblePose>& ranked);

#endif
