#include "pose_ranking.hpp"

#include "geometry.hpp"
#include "interaction_scoring.hpp"

#include <algorithm>
#include <chrono>

namespace
{

// Keep the raw comparisons and their order, including non-finite behavior.
// RMSD is an evaluation metric and never participates in ranking.
bool weighted_interaction_better(const RankedPoseDiagnostic& a, const RankedPoseDiagnostic& b)
{
    if (a.weighted_interaction_score.weighted_score != b.weighted_interaction_score.weighted_score) {
        return a.weighted_interaction_score.weighted_score > b.weighted_interaction_score.weighted_score;
    }
    if (a.weighted_interaction_score.satisfied_ligand_areas != b.weighted_interaction_score.satisfied_ligand_areas) {
        return a.weighted_interaction_score.satisfied_ligand_areas > b.weighted_interaction_score.satisfied_ligand_areas;
    }
    if (a.weighted_interaction_score.total_interactions != b.weighted_interaction_score.total_interactions) {
        return a.weighted_interaction_score.total_interactions > b.weighted_interaction_score.total_interactions;
    }
    return a.occupied_atoms < b.occupied_atoms;
}

} // namespace


PoseRankingResult rank_poses(
    const std::vector<RankedPoseDiagnostic>& surviving_poses,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& pocket_protein_areas,
    const ProteinGrid& protein_grid,
    std::size_t fast_prefilter_top_n)
{
    PoseRankingResult result;

    result.input_pose_count =
        surviving_poses.size();


    // Recompute the grid-based Fast score on every clash survivor.

    auto scored_poses =
        surviving_poses;


    const auto fast_start =
        std::chrono::steady_clock::now();


    for (auto& pose : scored_poses) {

        const auto transformed_matching =
            transform_matching_areas(
                ligand_matching_areas,
                pose.transform
            );


        const auto fast_score =
            score_matching_areas_weighted(
                transformed_matching,
                protein_grid
            );


        pose.fast_weighted_score =
            fast_score.weighted_score;
    }


    const auto fast_end =
        std::chrono::steady_clock::now();


    result.fast_scoring_seconds =
        std::chrono::duration<double>(
            fast_end -
            fast_start
        ).count();


    // Diagnostic baseline: rank by unweighted grid support, without RMSD.

    result.unweighted_ranking =
        scored_poses;


    std::sort(
        result.unweighted_ranking.begin(),
        result.unweighted_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.compatible_areas !=
                b.compatible_areas
            ) {
                return
                    a.compatible_areas >
                    b.compatible_areas;
            }

            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // Keep a separate Fast ranking; its prefix defines the WeightedInteraction candidate pool.

    result.fast_weighted_ranking =
        scored_poses;


    std::sort(
        result.fast_weighted_ranking.begin(),
        result.fast_weighted_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.fast_weighted_score !=
                b.fast_weighted_score
            ) {
                return
                    a.fast_weighted_score >
                    b.fast_weighted_score;
            }


            if (
                a.compatible_areas !=
                b.compatible_areas
            ) {
                return
                    a.compatible_areas >
                    b.compatible_areas;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // Only the first N Fast-ranked poses receive WeightedInteraction scores; zero selects none.

    result.weighted_interaction_candidates =
        result.fast_weighted_ranking;


    if (
        result.weighted_interaction_candidates.size() >
        fast_prefilter_top_n
    ) {
        result.weighted_interaction_candidates.resize(
            fast_prefilter_top_n
        );
    }


    // Oracle diagnostic after selection; it cannot influence the prefilter.

    result.best_prefilter_rmsd =
        std::numeric_limits<double>::infinity();


    for (const auto& pose :
         result.weighted_interaction_candidates) {

        result.best_prefilter_rmsd =
            std::min(
                result.best_prefilter_rmsd,
                pose.rmsd
            );
    }


    // Score Fast top-N once, recording the elapsed prefix time for up to 50 poses.

    const auto weighted_interaction_start =
        std::chrono::steady_clock::now();


    result.total_interactions = 0;


    const std::size_t top50_count =
        std::min<std::size_t>(
            50,
            result.weighted_interaction_candidates.size()
        );


    for (
        std::size_t pose_index = 0;
        pose_index < result.weighted_interaction_candidates.size();
        ++pose_index
    ) {
        auto& pose =
            result.weighted_interaction_candidates[
                pose_index
            ];


        const auto transformed_areas =
            transform_interaction_areas(
                ligand_areas,
                pose.transform
            );


        pose.weighted_interaction_score =
            score_weighted_interactions(
                pocket_protein_areas,
                transformed_areas
            );


        result.total_interactions +=
            pose.weighted_interaction_score
                .total_interactions;


        // Prefix timing includes area transforms and diagnostic bookkeeping.
        if (
            pose_index + 1 ==
            top50_count
        ) {
            const auto weighted_interaction_top50_end =
                std::chrono::steady_clock::now();


            result.weighted_interaction_top50_scoring_seconds =
                std::chrono::duration<double>(
                    weighted_interaction_top50_end -
                    weighted_interaction_start
                ).count();
        }
    }


    const auto weighted_interaction_end =
        std::chrono::steady_clock::now();


    result.weighted_interaction_scoring_seconds =
        std::chrono::duration<double>(
            weighted_interaction_end -
            weighted_interaction_start
        ).count();


    if (result.weighted_interaction_candidates.empty()) {

        result.weighted_interaction_top50_scoring_seconds =
            0.0;
    }


    // Rerank the first 50 Fast candidates with the WeightedInteraction scores already computed.
    // This is not the best 50 WeightedInteraction poses from the entire search.

    result.weighted_interaction_top50_ranking =
        result.weighted_interaction_candidates;


    if (
        result.weighted_interaction_top50_ranking.size() >
        50
    ) {
        result.weighted_interaction_top50_ranking.resize(
            50
        );
    }


    std::sort(
        result.weighted_interaction_top50_ranking.begin(),
        result.weighted_interaction_top50_ranking.end(),
        weighted_interaction_better
    );


    // Diagnostic ranking by interaction count, distinct from the weighted score.

    result.interaction_count_ranking =
        result.weighted_interaction_candidates;


    std::sort(
        result.interaction_count_ranking.begin(),
        result.interaction_count_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.weighted_interaction_score.total_interactions !=
                b.weighted_interaction_score.total_interactions
            ) {
                return
                    a.weighted_interaction_score.total_interactions >
                    b.weighted_interaction_score.total_interactions;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // Diagnostic ranking by the number of satisfied physical ligand areas.

    result.interaction_coverage_ranking =
        result.weighted_interaction_candidates;


    std::sort(
        result.interaction_coverage_ranking.begin(),
        result.interaction_coverage_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.weighted_interaction_score
                    .satisfied_ligand_areas !=
                b.weighted_interaction_score
                    .satisfied_ligand_areas
            ) {
                return
                    a.weighted_interaction_score
                        .satisfied_ligand_areas >
                    b.weighted_interaction_score
                        .satisfied_ligand_areas;
            }


            if (
                a.weighted_interaction_score.total_interactions !=
                b.weighted_interaction_score.total_interactions
            ) {
                return
                    a.weighted_interaction_score.total_interactions >
                    b.weighted_interaction_score.total_interactions;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // WeightedInteraction ranking of the Fast-selected candidate pool only.

    result.weighted_interaction_ranking =
        result.weighted_interaction_candidates;


    std::sort(
        result.weighted_interaction_ranking.begin(),
        result.weighted_interaction_ranking.end(),
        weighted_interaction_better
    );


    return result;
}


WeightedInteractionRankingResult weighted_interaction_score_and_rank(
    const std::vector<RankedPoseDiagnostic>& candidates,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas)
{
    WeightedInteractionRankingResult result;

    result.scored_candidates =
        candidates;


    const auto start =
        std::chrono::steady_clock::now();


    for (auto& pose :
         result.scored_candidates) {

        const auto transformed_areas =
            transform_interaction_areas(
                ligand_areas,
                pose.transform
            );

        pose.weighted_interaction_score =
            score_weighted_interactions(
                protein_areas,
                transformed_areas
            );


        result.total_interactions +=
            pose.weighted_interaction_score.total_interactions;
    }


    const auto end =
        std::chrono::steady_clock::now();


    result.scoring_seconds =
        std::chrono::duration<double>(
            end - start
        ).count();


    result.weighted_ranking =
        result.scored_candidates;


    std::sort(
        result.weighted_ranking.begin(),
        result.weighted_ranking.end(),
        weighted_interaction_better
    );


    return result;
}
