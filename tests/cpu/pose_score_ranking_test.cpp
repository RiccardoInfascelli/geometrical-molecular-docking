#include "pose_score_ranking.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12 * (1.0 + std::abs(expected)),
            "Unexpected numeric ranking result");
}

void rejects(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}

RescoredPose row(std::size_t id, double fast, double weighted_interaction, double vinardo, double repulsion = 0.0)
{
    RescoredPose result;
    result.pose_id = id;
    result.pose.fast_weighted_score = fast;
    result.pose.weighted_interaction_score.weighted_score = weighted_interaction;
    result.vinardo.total = vinardo;
    result.vinardo.repulsion = repulsion;
    result.vinardo.gaussian = vinardo - repulsion;
    return result;
}

PoseRescoringResult pool(std::vector<RescoredPose> rows)
{
    PoseRescoringResult result;
    result.input_pose_count = rows.size();
    result.poses = std::move(rows);
    result.weighted_interaction_calculated = result.vinardo_calculated = true;
    return result;
}

std::vector<std::size_t> ids(const std::vector<RankedPoseScore>& ranked)
{
    std::vector<std::size_t> result;
    for (const auto& pose : ranked) result.push_back(pose.scored.pose_id);
    return result;
}

void test_methods_and_complete_pool()
{
    require(parse_pose_score_method("weighted-interaction") == PoseScoreMethod::WeightedInteraction &&
            std::string(pose_score_method_name(PoseScoreMethod::WeightedInteraction)) == "weighted-interaction",
            "WeightedInteraction method name differs");
    rejects([] { parse_pose_score_method("exact"); });
    const auto all = pool({row(0, 100, 0, 10), row(1, 90, 1, 9), row(2, 1, 10, 0)});
    require(ids(rank_rescored_poses(all, {}, 1)) == std::vector<std::size_t>{0}, "Fast winner differs");
    for (const auto method : {PoseScoreMethod::WeightedInteraction, PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid}) {
        const auto ranked = rank_rescored_poses(all, {method}, 1);
        require(ids(ranked) == std::vector<std::size_t>{2},
                "Final ranking must see the winner outside the Fast top-k");
    }
    const auto full = rank_rescored_poses(all, {PoseScoreMethod::Hybrid}, 99);
    const auto top = rank_rescored_poses(all, {PoseScoreMethod::Hybrid}, 1);
    close(*top[0].hybrid, *full[0].hybrid);
    require(all.poses.front().pose_id == 0 && all.poses.size() == 3, "Ranking mutated input");

    const auto repulsion = pool({row(5, 1, 0, 2, 8), row(3, 1, 0, 0, 0)});
    require(ids(rank_rescored_poses(repulsion, {PoseScoreMethod::Vinardo, 1.0}, 1)) ==
            std::vector<std::size_t>{3}, "Vinardo raw winner differs");
    require(ids(rank_rescored_poses(repulsion, {PoseScoreMethod::Vinardo, 0.5}, 1)) ==
            std::vector<std::size_t>{5}, "Repulsion reweighting did not change the expected winner");
    close(vinardo_adjusted_score(repulsion.poses[0].vinardo, 1.0), 2.0);
    close(vinardo_adjusted_score(repulsion.poses[0].vinardo, 0.5), -2.0);
    close(vinardo_adjusted_score(repulsion.poses[0].vinardo, 0.0), -6.0);
}

void test_lightweight_selection()
{
    const std::vector<PoseScoreEntry> entries = {{7, 5}, {2, 9}, {1, 9}, {1, 9}, {0, -1}};
    require(top_pose_score_indices(entries, 3) == std::vector<std::size_t>({2, 3, 1}),
            "Selection must return source offsets, resolving ID ties by input order");
    require(top_pose_score_indices(entries, 2, true) == std::vector<std::size_t>({4, 0}),
            "Minimizing selection direction differs");
    require(top_pose_score_indices({}, 2).empty(), "Empty lightweight input differs");
    rejects([&] { top_pose_score_indices(entries, 0); });
    for (double score : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        auto invalid = entries;
        invalid.back().score = score;
        rejects([&] { top_pose_score_indices(invalid, 1); });
    }
}

void test_population_normalization_and_endpoints()
{
    const auto two = pool({row(8, 1, 0, 5), row(2, 1, 2, 5)});
    const auto ranked = rank_rescored_poses(two, {PoseScoreMethod::Hybrid, 1.0, 0.3}, 2);
    require(ids(ranked) == std::vector<std::size_t>({2, 8}), "Population hybrid order differs");
    close(*ranked[0].hybrid, 0.3); // Population std=1; sample std would yield 0.3/sqrt(2).
    close(*ranked[1].hybrid, -0.3);
    close(*rank_rescored_poses(two, {PoseScoreMethod::Hybrid, 1.0, 0.3}, 1)[0].hybrid, 0.3);

    const auto tied = pool({row(9, 3, 4, -1), row(1, 3, 4, -1)});
    for (const auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                              PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid}) {
        const auto result = rank_rescored_poses(tied, {method}, 20);
        require(ids(result) == std::vector<std::size_t>({1, 9}), "Ties must use increasing original ID");
        if (method == PoseScoreMethod::Hybrid) close(*result[0].hybrid, 0.0);
    }
    const auto single = rank_rescored_poses(pool({row(4, 1, 8, -3)}), {PoseScoreMethod::Hybrid}, 1);
    close(*single[0].hybrid, 0.0);

    const auto endpoints = pool({row(7, 9, 0, -9), row(1, 0, 10, 5), row(3, 1, 10, 5)});
    require(ids(rank_rescored_poses(endpoints, {PoseScoreMethod::Hybrid, 1.0, 0.0}, 99)) ==
            ids(rank_rescored_poses(endpoints, {PoseScoreMethod::Vinardo}, 99)), "Lambda zero changed Vinardo");
    require(ids(rank_rescored_poses(endpoints, {PoseScoreMethod::Hybrid, 1.0, 1.0}, 99)) ==
            ids(rank_rescored_poses(endpoints, {PoseScoreMethod::WeightedInteraction}, 99)), "Lambda one changed WeightedInteraction");

    // Numerically large scores remain finite after scale-invariant normalization.
    const double maximum = std::numeric_limits<double>::max();
    const auto extreme = rank_rescored_poses(pool({row(0, 0, -maximum, 0), row(1, 0, maximum, 0)}),
                                             {PoseScoreMethod::Hybrid, 1.0, 1.0}, 2);
    close(*extreme[0].hybrid, 1.0);
    close(*extreme[1].hybrid, -1.0);

    // A declared Fast prefilter changes the normalization pool; final top-k
    // must normalize all three supplied rows, not silently repeat a prefilter.
    const auto complete = pool({row(0, 100, 0, 5), row(1, 90, 2, 5), row(2, 1, 100, 5)});
    const auto complete_top = rank_rescored_poses(complete, {PoseScoreMethod::Hybrid}, 1);
    auto prefiltered = pool({complete.poses[0], complete.poses[1]});
    prefiltered.input_pose_count = complete.input_pose_count;
    const auto filtered_top = rank_rescored_poses(prefiltered, {PoseScoreMethod::Hybrid}, 1);
    require(complete_top[0].scored.pose_id == 2 && filtered_top[0].scored.pose_id == 1,
            "Explicit prefilter pool was silently changed");
    close(*filtered_top[0].hybrid, 0.3);
}

void test_validation()
{
    const auto empty = pool({});
    for (const auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                              PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
        require(rank_rescored_poses(empty, {method}, 1).empty(), "Valid empty pool did not remain empty");
    rejects([&] { rank_rescored_poses(empty, {}, 0); });
    rejects([&] { rank_rescored_poses(empty, {static_cast<PoseScoreMethod>(99)}, 1); });
    rejects([&] { pose_score_method_name(static_cast<PoseScoreMethod>(99)); });
    rejects([&] { parse_pose_score_method("unknown"); });
    for (const auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                              PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
        require(parse_pose_score_method(pose_score_method_name(method)) == method, "Method name roundtrip differs");

    auto missing = pool({row(0, 1, 2, 3)});
    missing.weighted_interaction_calculated = false;
    rejects([&] { rank_rescored_poses(missing, {PoseScoreMethod::WeightedInteraction}, 1); });
    rejects([&] { rank_rescored_poses(missing, {PoseScoreMethod::Hybrid}, 1); });
    missing.vinardo_calculated = false;
    require(rank_rescored_poses(missing, {}, 1).size() == 1, "Fast requires an unused rescoring method");
    rejects([&] { rank_rescored_poses(missing, {PoseScoreMethod::Vinardo}, 1); });

    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        rejects([&] { rank_rescored_poses(empty, {PoseScoreMethod::Fast, invalid}, 1); });
        rejects([&] { rank_rescored_poses(empty, {PoseScoreMethod::Hybrid, 1.0, invalid}, 1); });
        rejects([&] { vinardo_adjusted_score({}, invalid); });
        auto nonfinite = pool({row(0, 9, 4, 0), row(1, 0, 1, 4)});
        nonfinite.poses[1].pose.fast_weighted_score = invalid;
        rejects([&] { rank_rescored_poses(nonfinite, {}, 1); });
        nonfinite = pool({row(0, 9, 4, 0), row(1, 0, 1, 4)});
        nonfinite.poses[1].pose.weighted_interaction_score.weighted_score = invalid;
        rejects([&] { rank_rescored_poses(nonfinite, {PoseScoreMethod::WeightedInteraction}, 1); });
        nonfinite = pool({row(0, 9, 4, 0), row(1, 0, 1, 4)});
        nonfinite.poses[1].vinardo.repulsion = invalid;
        rejects([&] { rank_rescored_poses(nonfinite, {PoseScoreMethod::Vinardo}, 1); });
        for (auto component : {&VinardoLikeScore::total, &VinardoLikeScore::gaussian,
                               &VinardoLikeScore::repulsion, &VinardoLikeScore::hydrophobic,
                               &VinardoLikeScore::hydrogen_bond}) {
            VinardoLikeScore terms;
            terms.*component = invalid;
            rejects([&] { vinardo_adjusted_score(terms, 1.0); });
        }
    }
    rejects([&] { rank_rescored_poses(empty, {PoseScoreMethod::Fast, -0.1}, 1); });
    for (double lambda : {-0.1, 1.1})
        rejects([&] { rank_rescored_poses(empty, {PoseScoreMethod::Hybrid, 1.0, lambda}, 1); });
    auto overflow = row(0, 0, 0, std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max());
    rejects([&] { vinardo_adjusted_score(overflow.vinardo, 2.0); });
}

} // namespace

int main()
{
    try {
        test_methods_and_complete_pool();
        test_lightweight_selection();
        test_population_normalization_and_endpoints();
        test_validation();
        std::cout << "Pose score ranking tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
