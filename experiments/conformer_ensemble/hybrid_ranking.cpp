#include "hybrid_ranking.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace conformer_ensemble {

HybridEnsembleRanking rank_ensemble_poses(
    std::vector<EnsemblePose> poses,
    std::size_t local_k,
    std::size_t global_x)
{
    if (global_x == 0 || local_k < global_x)
        throw std::invalid_argument("Require local K >= global X > 0");

    std::sort(poses.begin(), poses.end(), [](const auto& a, const auto& b) {
        return std::tie(a.record_index, a.scored.pose_id) <
               std::tie(b.record_index, b.scored.pose_id);
    });

    PoseRescoringResult pool;
    pool.input_pose_count = poses.size();
    pool.weighted_interaction_calculated = pool.vinardo_calculated = true;
    pool.poses.reserve(poses.size());
    std::map<std::size_t, std::string> record_names;
    std::map<std::string, std::size_t> conformer_records;
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const auto& row = poses[i];
        if (row.conformer_id.empty())
            throw std::invalid_argument("Conformer ID must not be empty");
        if (!std::isfinite(row.rmsd_heavy) || row.rmsd_heavy < 0.0)
            throw std::invalid_argument("Heavy-atom RMSD must be finite and non-negative");
        if (i > 0 && row.record_index == poses[i - 1].record_index &&
            row.scored.pose_id == poses[i - 1].scored.pose_id)
            throw std::invalid_argument("Duplicate conformer record and pose ID");
        const auto [record, new_record] = record_names.emplace(row.record_index, row.conformer_id);
        const auto [conformer, new_conformer] = conformer_records.emplace(row.conformer_id, row.record_index);
        if ((!new_record && record->second != row.conformer_id) ||
            (!new_conformer && conformer->second != row.record_index))
            throw std::invalid_argument("Conformer IDs and record indices must have a one-to-one mapping");
        pool.poses.push_back(row.scored);
        // pose_id is local to a conformer. A temporary global ID lets the core
        // ranker apply its normal tie-breaking without losing the original ID.
        pool.poses.back().pose_id = i;
    }

    const auto ranked = rank_rescored_poses(
        pool, {PoseScoreMethod::Hybrid, 0.5, 0.3},
        std::max<std::size_t>(1, poses.size()));
    HybridEnsembleRanking result;
    result.all.reserve(ranked.size());
    std::map<std::size_t, std::size_t> local_ranks;
    for (const auto& row : ranked) {
        const auto& original = poses.at(row.scored.pose_id);
        result.all.push_back({original, ++local_ranks[original.record_index],
                              result.all.size() + 1, *row.hybrid, *row.vinardo_adjusted});
    }

    // Filter already-normalized scores, then use the ordinary score selector.
    // No new Hybrid normalization is performed on the retained local winners.
    std::vector<PoseScoreEntry> candidates;
    for (const auto& row : result.all) {
        if (row.local_rank <= local_k) {
            result.local.push_back(row);
            candidates.push_back({row.global_rank, row.hybrid});
        }
    }
    for (const auto index : top_pose_score_indices(candidates, global_x))
        result.global.push_back(result.local[index]);
    return result;
}

} // namespace conformer_ensemble
