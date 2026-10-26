#ifndef POSE_RANKING_HPP
#define POSE_RANKING_HPP

#include "matching_areas.hpp"
#include "pose_generation.hpp"
#include "protein_grid.hpp"
#include <interaction/interaction_area.hpp>

#include <cstddef>
#include <limits>
#include <vector>


struct PoseRankingResult
{
    std::size_t input_pose_count{};

    std::vector<RankedPoseDiagnostic>
        unweighted_ranking;

    std::vector<RankedPoseDiagnostic>
        fast_weighted_ranking;

    std::vector<RankedPoseDiagnostic>
        weighted_interaction_candidates;

    std::vector<RankedPoseDiagnostic>
        interaction_count_ranking;

    std::vector<RankedPoseDiagnostic>
        interaction_coverage_ranking;

    std::vector<RankedPoseDiagnostic>
        weighted_interaction_ranking;

    // Scoring loops include coordinate transforms, but exclude subsequent sorts.
    double fast_scoring_seconds{};
    double weighted_interaction_scoring_seconds{};

    std::size_t total_interactions{};

    double best_prefilter_rmsd =
        std::numeric_limits<double>::infinity();

    std::vector<RankedPoseDiagnostic>
        weighted_interaction_top50_ranking;

    double weighted_interaction_top50_scoring_seconds{};
};


// Recompute Fast scores, rank, prefilter, then WeightedInteraction-score and rerank the selected
// poses. Both scores are maximized; RMSD is used only for evaluation diagnostics.
// fast_prefilter_top_n == 0 selects no WeightedInteraction candidates.
// WeightedInteraction top-50 reranks the first 50 Fast candidates using their already computed
// WeightedInteraction scores.
// Sorts are not stable for complete ties. Raw floating-point comparisons are kept:
// there is no score sanitization or total ordering of NaN values.
PoseRankingResult rank_poses(
    const std::vector<RankedPoseDiagnostic>& poses,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const ProteinGrid& protein_grid,
    std::size_t fast_prefilter_top_n
);


struct WeightedInteractionRankingResult
{
    std::vector<RankedPoseDiagnostic>
        scored_candidates;

    std::vector<RankedPoseDiagnostic>
        weighted_ranking;

    std::size_t total_interactions{};

    double scoring_seconds{};
};


// Recompute WeightedInteraction scores without a Fast prefilter. Order by descending weighted
// score, ligand coverage, interaction count, then ascending occupied atom count.
// RMSD is excluded, with the same raw comparison policy as rank_poses().
WeightedInteractionRankingResult weighted_interaction_score_and_rank(
    const std::vector<RankedPoseDiagnostic>& candidates,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas
);

#endif
