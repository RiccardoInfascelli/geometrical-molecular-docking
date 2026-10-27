#include "ensemble_ranking.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

std::vector<RankedEnsemblePose> rank_ensemble_poses(
    const std::vector<EnsemblePose>& poses,
    PoseScoreMethod method,
    double* hybrid_normalization_seconds)
{
    PoseRescoringResult pool;
    pool.input_pose_count = poses.size();
    pool.weighted_interaction_calculated =
        method == PoseScoreMethod::WeightedInteraction || method == PoseScoreMethod::Hybrid;
    pool.vinardo_calculated = method == PoseScoreMethod::Vinardo || method == PoseScoreMethod::Hybrid;
    pool.poses.reserve(poses.size());
    for (std::size_t i = 0; i < poses.size(); ++i) {
        pool.poses.push_back(poses[i].scored);
        // The production ranker resolves equal scores by pose_id. A temporary
        // source offset gives an ensemble-wide identity without mutating inputs.
        pool.poses.back().pose_id = i;
    }
    const auto ranked = rank_rescored_poses(
        pool, {method, 0.5, 0.3}, std::max(std::size_t{1}, poses.size()),
        hybrid_normalization_seconds);

    std::unordered_map<std::size_t, std::size_t> local_ranks;
    std::vector<RankedEnsemblePose> result;
    result.reserve(ranked.size());
    for (const auto& row : ranked) {
        const auto source = row.scored.pose_id;
        double score = 0.0;
        switch (method) {
        case PoseScoreMethod::Fast:
            score = row.scored.pose.fast_weighted_score;
            break;
        case PoseScoreMethod::WeightedInteraction:
            score = row.scored.pose.weighted_interaction_score.weighted_score;
            break;
        case PoseScoreMethod::Vinardo:
            score = row.vinardo_adjusted.value();
            break;
        case PoseScoreMethod::Hybrid:
            score = row.hybrid.value();
            break;
        }
        result.push_back({source, ++local_ranks[poses[source].conformer_id],
                          result.size() + 1, row.hybrid, score});
    }
    return result;
}

EnsembleRankingSummary summarize_ensemble_ranking(
    const std::vector<EnsemblePose>& poses,
    const std::vector<RankedEnsemblePose>& ranked)
{
    if (ranked.size() != poses.size())
        throw std::invalid_argument("Ensemble summary requires the complete ranked pool");
    EnsembleRankingSummary result;
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const double rmsd = poses[i].rmsd;
        if (!std::isfinite(rmsd) || rmsd < 0.0)
            throw std::invalid_argument("Ensemble evaluation RMSD must be finite and nonnegative");
        if (!result.oracle_rmsd || rmsd < *result.oracle_rmsd) {
            result.oracle_rmsd = rmsd;
            result.oracle_source_index = i;
        }
    }
    std::vector<bool> seen(poses.size(), false);
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const auto source = ranked[i].source_index;
        if (source >= poses.size() || seen[source] || ranked[i].global_rank != i + 1)
            throw std::invalid_argument("Ensemble summary requires a complete source permutation in global rank order");
        seen[source] = true;
        const double rmsd = poses[source].rmsd;
        for (std::size_t top = 0; top < ensemble_summary_top_k.size(); ++top) {
            auto& best = result.top_k_best_rmsd[top];
            if (i < ensemble_summary_top_k[top] && (!best || rmsd < *best))
                best = rmsd;
        }
    }
    return result;
}
