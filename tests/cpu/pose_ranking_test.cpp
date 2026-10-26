#include "interaction_scoring.hpp"
#include "pose_ranking.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The synthetic translations also identify the poses, without using RMSD.
std::vector<double> translations(const std::vector<RankedPoseDiagnostic>& poses)
{
    std::vector<double> result;
    for (const auto& pose : poses) {
        result.push_back(pose.transform.translation[0]);
    }
    return result;
}

void test_fast_scoring()
{
    ProteinGrid grid;
    grid.voxel_size = 1.0;
    grid.nx = 4;
    grid.ny = grid.nz = 1;
    grid.voxels.resize(4);
    add_interaction(grid.voxels[0].interactions, InteractionKey::Hydrophobic);
    add_interaction(grid.voxels[1].interactions, InteractionKey::HydrogenDonor);
    add_interaction(grid.voxels[1].interactions, InteractionKey::HydrogenAcceptor);
    add_interaction(grid.voxels[2].interactions, InteractionKey::Aromatic);
    grid.voxels[2].occupied = true;

    const std::vector<MatchingArea> areas{
        {0, {0.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {0, {0.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {1, {1.25, 0.25, 0.25}, InteractionKey::HydrogenDonor},
        {2, {1.25, 0.25, 0.25}, InteractionKey::HydrogenAcceptor},
        {3, {2.25, 0.25, 0.25}, InteractionKey::Aromatic},
        {4, {1.25, 0.25, 0.25}, InteractionKey::Cation},
        {5, {4.25, 0.25, 0.25}, InteractionKey::Metal},
        {6, {-0.25, 0.25, 0.25}, InteractionKey::Hydrophobic}
    };
    const auto score = score_matching_areas_weighted(areas, grid);
    require(score.total_areas == 8 && score.areas_in_grid == 6 &&
            score.areas_on_occupied_voxel == 1 && score.compatible_areas == 4 &&
            score.weighted_score == 6.0,
            "Fast scoring counts, default weights or unnormalized sum changed");

    FastScoreWeights weights;
    weights.hydrophobic = -1.25;
    weights.hydrogen_bond = 0.0;
    require(score_matching_areas_weighted(areas, grid, weights).weighted_score == -2.5,
            "Fast scoring must retain signed weights and repeated logical areas");
    weights.hydrophobic = std::numeric_limits<double>::infinity();
    require(score_matching_areas_weighted(areas, grid, weights).weighted_score ==
                std::numeric_limits<double>::infinity(), "Positive infinite weight changed");
    weights.hydrophobic = -std::numeric_limits<double>::infinity();
    require(score_matching_areas_weighted(areas, grid, weights).weighted_score ==
                -std::numeric_limits<double>::infinity(), "Negative infinite weight changed");
    weights.hydrophobic = std::numeric_limits<double>::quiet_NaN();
    require(std::isnan(score_matching_areas_weighted(areas, grid, weights).weighted_score),
            "Fast scoring NaN propagation changed");
}

void test_ranking()
{
    const std::vector<MatchingArea> areas{
        {0, {0.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {1, {1.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {2, {2.25, 0.25, 0.25}, InteractionKey::HydrogenDonor},
        {3, {3.25, 0.25, 0.25}, InteractionKey::Cation}
    };
    ProteinGrid grid;
    grid.voxel_size = 1.0;
    grid.nx = 24;
    grid.ny = grid.nz = 1;
    grid.voxels.resize(24);
    for (int offset : {0, 5, 15, 20}) {
        add_interaction(grid.voxels[offset].interactions, InteractionKey::Hydrophobic);
        add_interaction(grid.voxels[offset + 1].interactions, InteractionKey::Hydrophobic);
    }
    add_interaction(grid.voxels[2].interactions, InteractionKey::HydrogenDonor);
    add_interaction(grid.voxels[13].interactions, InteractionKey::Cation);

    std::vector<RankedPoseDiagnostic> poses(5);
    const std::vector<double> offsets{0, 5, 10, 15, 20};
    const std::vector<std::size_t> compatible{3, 2, 1, 2, 2};
    const std::vector<std::size_t> occupied{5, 3, 1, 2, 4};
    const std::vector<double> rmsds{100, 0, 50, 25, 1};
    for (std::size_t i = 0; i < poses.size(); ++i) {
        poses[i].transform.translation[0] = offsets[i];
        poses[i].compatible_areas = compatible[i];
        poses[i].occupied_atoms = occupied[i];
        poses[i].rmsd = rmsds[i];
        poses[i].fast_weighted_score = -99.0;
        poses[i].weighted_interaction_score.weighted_score = 99.0;
        poses[i].weighted_interaction_score.satisfied_ligand_areas = 99;
        poses[i].weighted_interaction_score.total_interactions = 99;
    }

    // Empty chemistry isolates the final WeightedInteraction tie-break and checks that scores
    // are recomputed instead of reusing the incoming values.
    const auto ranked = rank_poses(poses, areas, {}, {}, grid, 3);
    require(ranked.input_pose_count == 5 &&
            translations(ranked.fast_weighted_ranking) == std::vector<double>{0, 10, 15, 5, 20} &&
            translations(ranked.unweighted_ranking) == std::vector<double>{0, 15, 5, 20, 10},
            "Fast or unweighted ranking/tie-breaks changed");
    require(translations(ranked.weighted_interaction_candidates) == std::vector<double>{0, 10, 15} &&
            ranked.best_prefilter_rmsd == 25.0 && ranked.total_interactions == 0,
            "Fast prefilter selection or its RMSD diagnostic changed");
    const std::vector<double> weighted_interaction_order{10, 15, 0};
    require(translations(ranked.weighted_interaction_ranking) == weighted_interaction_order &&
            translations(ranked.weighted_interaction_top50_ranking) == weighted_interaction_order &&
            translations(ranked.interaction_count_ranking) == weighted_interaction_order &&
            translations(ranked.interaction_coverage_ranking) == weighted_interaction_order,
            "WeightedInteraction final tie-break or RMSD independence changed");

    auto changed_rmsds = poses;
    changed_rmsds[0].rmsd = std::numeric_limits<double>::quiet_NaN();
    changed_rmsds[2].rmsd = std::numeric_limits<double>::infinity();
    changed_rmsds[3].rmsd = 9.0;
    const auto changed = rank_poses(changed_rmsds, areas, {}, {}, grid, 3);
    require(translations(changed.fast_weighted_ranking) == translations(ranked.fast_weighted_ranking) &&
            translations(changed.unweighted_ranking) == translations(ranked.unweighted_ranking) &&
            translations(changed.weighted_interaction_candidates) == translations(ranked.weighted_interaction_candidates) &&
            translations(changed.weighted_interaction_ranking) == weighted_interaction_order &&
            translations(changed.weighted_interaction_top50_ranking) == weighted_interaction_order &&
            changed.best_prefilter_rmsd == 9.0,
            "Finite, infinite or NaN RMSD must not alter ranking or selection");

    const auto weighted_interaction = weighted_interaction_score_and_rank(poses, {}, {});
    require(translations(weighted_interaction.scored_candidates) == offsets &&
            translations(weighted_interaction.weighted_ranking) == std::vector<double>{10, 15, 5, 20, 0} &&
            weighted_interaction.total_interactions == 0, "Standalone WeightedInteraction ranking changed");
    for (const auto& pose : weighted_interaction.scored_candidates) {
        require(pose.weighted_interaction_score.weighted_score == 0.0 &&
                pose.weighted_interaction_score.satisfied_ligand_areas == 0 &&
                pose.weighted_interaction_score.total_interactions == 0,
                "WeightedInteraction scoring must recompute, not reuse incoming scores");
    }

    const auto none = rank_poses(poses, areas, {}, {}, grid, 0);
    require(none.weighted_interaction_candidates.empty() && none.weighted_interaction_ranking.empty() &&
            none.weighted_interaction_top50_ranking.empty() &&
            none.weighted_interaction_top50_scoring_seconds == 0.0 &&
            none.best_prefilter_rmsd == std::numeric_limits<double>::infinity(),
            "Zero-size Fast prefilter behavior changed");
    const auto empty = rank_poses({}, areas, {}, {}, grid, 3);
    require(empty.input_pose_count == 0 && empty.fast_weighted_ranking.empty() &&
            empty.weighted_interaction_candidates.empty() && weighted_interaction_score_and_rank({}, {}, {}).weighted_ranking.empty(),
            "Empty ranking input behavior changed");

    std::vector<RankedPoseDiagnostic> many(55);
    for (std::size_t i = 0; i < many.size(); ++i) {
        many[i].transform.translation[0] = static_cast<double>(i);
        many[i].occupied_atoms = many.size() - i;
    }
    const auto capped = rank_poses(many, {}, {}, {}, grid, many.size());
    require(capped.weighted_interaction_candidates.size() == 55 && capped.weighted_interaction_ranking.size() == 55 &&
            capped.weighted_interaction_top50_ranking.size() == 50 &&
            capped.weighted_interaction_top50_ranking.front().transform.translation[0] == 54.0 &&
            capped.weighted_interaction_top50_ranking.back().transform.translation[0] == 5.0,
            "WeightedInteraction top-50 must retain the first 50 Fast candidates");
}

} // namespace

int main()
{
    try {
        test_fast_scoring();
        test_ranking();
        std::cout << "CPU Fast scoring and pose ranking tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
