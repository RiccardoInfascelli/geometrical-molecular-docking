#include "ensemble_ranking.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr std::array<PoseScoreMethod, 4> methods{{
    PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
    PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid
}};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected)
{
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= 1e-12 * (1.0 + std::abs(expected)),
            "Unexpected ensemble score or evaluation metric");
}

void rejects(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}

EnsemblePose pose(std::size_t conformer, std::size_t id, double fast,
                  double weighted, double vinardo, double repulsion = 0.0, double rmsd = 4.0)
{
    EnsemblePose result;
    result.conformer_id = conformer;
    result.pose_id = id;
    result.scored.pose_id = id + 100;
    result.scored.pose.fast_weighted_score = fast;
    result.scored.pose.weighted_interaction_score.weighted_score = weighted;
    result.scored.vinardo.total = vinardo;
    result.scored.vinardo.repulsion = repulsion;
    result.scored.vinardo.gaussian = vinardo - repulsion;
    result.rmsd = rmsd;
    return result;
}

std::vector<std::size_t> sources(const std::vector<RankedEnsemblePose>& ranked)
{
    std::vector<std::size_t> result;
    for (const auto& row : ranked) result.push_back(row.source_index);
    return result;
}

void same_ranking(const std::vector<RankedEnsemblePose>& actual,
                  const std::vector<RankedEnsemblePose>& expected)
{
    require(actual.size() == expected.size(), "Ranking changed ensemble multiplicity");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].source_index == expected[i].source_index &&
                actual[i].local_rank == expected[i].local_rank &&
                actual[i].global_rank == expected[i].global_rank &&
                actual[i].hybrid == expected[i].hybrid && actual[i].score == expected[i].score,
                "RMSD or timing changed ensemble ranking");
    }
}

void test_methods_and_reweighting()
{
    const std::vector<EnsemblePose> all{
        pose(5, 9, 100, 1, 2, 8, 9),
        pose(5, 2, 90, 10, 0, 0, 0.5),
        pose(99, 0, 1, 0, -1, 0, 3)
    };
    const std::array<std::vector<std::size_t>, 4> expected{{
        {0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {0, 2, 1}
    }};
    for (std::size_t method = 0; method < methods.size(); ++method) {
        double normalization_seconds = -1.0;
        const auto ranked = rank_ensemble_poses(all, methods[method], &normalization_seconds);
        require(sources(ranked) == expected[method], "Unexpected method direction or rep50 order");
        require(std::isfinite(normalization_seconds) && normalization_seconds >= 0.0,
                "Invalid Hybrid normalization interval");
        if (methods[method] != PoseScoreMethod::Hybrid)
            require(normalization_seconds == 0.0, "Non-Hybrid ranking reported normalization time");
        same_ranking(ranked, rank_ensemble_poses(all, methods[method]));
        const auto evaluation = summarize_ensemble_ranking(all, ranked);
        close(*evaluation.oracle_rmsd, 0.5);
        require(evaluation.oracle_source_index == 1, "Oracle used score order instead of minimum RMSD");
        for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
            const auto source = ranked[rank].source_index;
            require(ranked[rank].global_rank == rank + 1, "Global ranks are not one-based");
            require(ranked[rank].hybrid.has_value() == (methods[method] == PoseScoreMethod::Hybrid),
                    "Unexpected Hybrid field for a ranking method");
            double expected_score = 0.0;
            if (methods[method] == PoseScoreMethod::Fast)
                expected_score = all[source].scored.pose.fast_weighted_score;
            else if (methods[method] == PoseScoreMethod::WeightedInteraction)
                expected_score = all[source].scored.pose.weighted_interaction_score.weighted_score;
            else if (methods[method] == PoseScoreMethod::Vinardo)
                expected_score = std::array<double, 3>{{-2, 0, -1}}[source];
            else {
                // Independent closed-form population moments for WI={1,10,0}
                // and -V_rep50={2,0,1}; these values do not call the core ranker.
                const double weighted_z =
                    (all[source].scored.pose.weighted_interaction_score.weighted_score - 11.0 / 3.0) /
                    std::sqrt(182.0 / 9.0);
                const double reward_z = std::array<double, 3>{{1, -1, 0}}[source] / std::sqrt(2.0 / 3.0);
                expected_score = 0.3 * weighted_z + 0.7 * reward_z;
                close(*ranked[rank].hybrid, expected_score);
            }
            close(ranked[rank].score, expected_score);
        }
    }
    require(all[0].pose_id == 9 && all[0].scored.pose_id == 109 && all[0].rmsd == 9,
            "Ranking changed source IDs or evaluation data");
}

void test_one_global_hybrid_normalization()
{
    const std::vector<EnsemblePose> all{
        pose(9, 90, 4, 0, -1), pose(9, 20, 3, 2, 0),
        pose(4, 7, 2, 0, -1000), pose(4, 7, 1, 0, 1000)
    };
    const auto global = rank_ensemble_poses(all, PoseScoreMethod::Hybrid);
    require(sources(global) == std::vector<std::size_t>({2, 1, 0, 3}),
            "Hybrid was not normalized on the full ensemble");
    const std::array<std::size_t, 4> expected_local{{1, 1, 2, 2}};
    const std::array<double, 4> weighted{{0, 2, 0, 0}}, reward{{1, 0, 1000, -1000}};
    for (std::size_t rank = 0; rank < global.size(); ++rank) {
        const auto source = global[rank].source_index;
        const double expected = 0.3 * (weighted[source] - 0.5) / std::sqrt(0.75) +
            0.7 * (reward[source] - 0.25) / std::sqrt(500000.1875);
        close(global[rank].score, expected);
        require(global[rank].local_rank == expected_local[rank],
                "Local ranks did not follow the globally normalized order");
    }
    const auto separate = rank_ensemble_poses({all[0], all[1]}, PoseScoreMethod::Hybrid);
    require(sources(separate) == std::vector<std::size_t>({0, 1}),
            "Fixture must distinguish per-conformer and global Hybrid order");
    close(separate[0].score, 0.4);
    close(separate[1].score, -0.4);

    // The caller may explicitly retain one Fast winner per conformer. The
    // ranker uses exactly the resulting pool and never repeats a prefilter.
    const auto prefiltered = rank_ensemble_poses({all[0], all[2]}, PoseScoreMethod::Hybrid);
    require(global.size() == all.size() && sources(prefiltered) == std::vector<std::size_t>({1, 0}),
            "Ranking implicitly filtered or restored excluded poses");
    close(prefiltered[0].score, 0.7);
    close(prefiltered[1].score, -0.7);
}

void test_ties_duplicates_and_rmsd_independence()
{
    std::vector<EnsemblePose> tied{
        pose(9, 99, 2, 3, -4, 0, 9), pose(1, 0, 2, 3, -4, 0, 0.1),
        pose(9, 99, 2, 3, -4, 0, 2), pose(1, 0, 2, 3, -4, 0, 0.1)
    };
    for (const auto method : methods) {
        const auto ranked = rank_ensemble_poses(tied, method);
        require(sources(ranked) == std::vector<std::size_t>({0, 1, 2, 3}),
                "Score ties must preserve source order and duplicate occurrences");
        for (std::size_t i = 0; i < ranked.size(); ++i) {
            require(ranked[i].local_rank == i / 2 + 1 && ranked[i].global_rank == i + 1,
                    "Tied local/global ranks changed");
            if (method == PoseScoreMethod::Hybrid) close(ranked[i].score, 0.0);
        }
        auto changed = tied;
        for (auto& row : changed) row.rmsd = std::numeric_limits<double>::quiet_NaN();
        same_ranking(rank_ensemble_poses(changed, method), ranked);
        rejects([&] { summarize_ensemble_ranking(changed, ranked); });
        std::swap(changed[0], changed[1]);
        require(sources(rank_ensemble_poses(changed, method)) == std::vector<std::size_t>({0, 1, 2, 3}),
                "Tie breaking sorted conformer/pose IDs instead of source order");
    }
}

void test_evaluation_top_k_and_oracle()
{
    std::vector<EnsemblePose> all;
    for (std::size_t i = 0; i < 22; ++i)
        all.push_back(pose(i % 3, i, 22.0 - i, i, -static_cast<double>(i), 0, 22.0 - i));
    all[20].rmsd = all[21].rmsd = 1.0;
    for (const auto method : methods) {
        const auto ranked = rank_ensemble_poses(all, method);
        const auto evaluation = summarize_ensemble_ranking(all, ranked);
        close(*evaluation.oracle_rmsd, 1.0);
        require(evaluation.oracle_source_index == 20, "Oracle RMSD ties must use source order");
        const std::array<double, 4> expected = method == PoseScoreMethod::Fast
            ? std::array<double, 4>{{22, 18, 13, 3}} : std::array<double, 4>{{1, 1, 1, 1}};
        for (std::size_t top = 0; top < expected.size(); ++top)
            close(*evaluation.top_k_best_rmsd[top], expected[top]);
    }
    const auto single = rank_ensemble_poses({all[0]}, PoseScoreMethod::Hybrid);
    const auto evaluation = summarize_ensemble_ranking({all[0]}, single);
    for (const auto& top : evaluation.top_k_best_rmsd) close(*top, 22.0);
    close(single[0].score, 0.0);
}

void test_validation()
{
    for (const auto method : methods) {
        double seconds = -1;
        require(rank_ensemble_poses({}, method, &seconds).empty() && std::isfinite(seconds) && seconds >= 0.0,
                "Empty ensemble ranking differs");
        if (method != PoseScoreMethod::Hybrid)
            require(seconds == 0.0, "Empty non-Hybrid ranking reported normalization time");
    }
    const auto empty = summarize_ensemble_ranking({}, {});
    require(!empty.oracle_rmsd && !empty.oracle_source_index, "Empty ensemble invented an oracle");
    for (const auto& top : empty.top_k_best_rmsd) require(!top, "Empty ensemble invented a Top-k RMSD");
    rejects([] { rank_ensemble_poses({}, static_cast<PoseScoreMethod>(99)); });
    const std::vector<EnsemblePose> valid{pose(1, 0, 1, 2, -3), pose(1, 1, 0, 1, -2)};
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity()}) {
        auto changed = valid;
        changed.back().scored.pose.fast_weighted_score = invalid;
        for (const auto method : methods) rejects([&] { rank_ensemble_poses(changed, method); });
        changed = valid;
        changed.back().scored.pose.weighted_interaction_score.weighted_score = invalid;
        for (const auto method : {PoseScoreMethod::WeightedInteraction, PoseScoreMethod::Hybrid})
            rejects([&] { rank_ensemble_poses(changed, method); });
        require(rank_ensemble_poses(changed, PoseScoreMethod::Fast).size() == changed.size(),
                "Fast validated an unused WeightedInteraction field");
        for (auto component : {&VinardoLikeScore::total, &VinardoLikeScore::gaussian,
                               &VinardoLikeScore::repulsion, &VinardoLikeScore::hydrophobic,
                               &VinardoLikeScore::hydrogen_bond}) {
            changed = valid;
            changed.back().scored.vinardo.*component = invalid;
            for (const auto method : {PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
                rejects([&] { rank_ensemble_poses(changed, method); });
            require(rank_ensemble_poses(changed, PoseScoreMethod::Fast).size() == changed.size(),
                    "Fast validated an unused Vinardo field");
        }
        changed = valid;
        changed.back().rmsd = invalid;
        rejects([&] { summarize_ensemble_ranking(changed, rank_ensemble_poses(changed, PoseScoreMethod::Fast)); });
    }
    auto changed = valid;
    changed.back().scored.vinardo.total = -std::numeric_limits<double>::max();
    changed.back().scored.vinardo.repulsion = std::numeric_limits<double>::max();
    changed.back().scored.vinardo.gaussian = 0.0;
    for (const auto method : {PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
        rejects([&] { rank_ensemble_poses(changed, method); });
    changed = valid;
    changed.back().rmsd = -1.0;
    rejects([&] { summarize_ensemble_ranking(changed, rank_ensemble_poses(changed, PoseScoreMethod::Fast)); });
    const auto ranked = rank_ensemble_poses(valid, PoseScoreMethod::Fast);
    auto malformed = ranked;
    malformed.pop_back();
    rejects([&] { summarize_ensemble_ranking(valid, malformed); });
    malformed = ranked;
    malformed.back().source_index = malformed.front().source_index;
    rejects([&] { summarize_ensemble_ranking(valid, malformed); });
    malformed.back().source_index = valid.size();
    rejects([&] { summarize_ensemble_ranking(valid, malformed); });
    malformed = ranked;
    malformed.back().global_rank = 1;
    rejects([&] { summarize_ensemble_ranking(valid, malformed); });
}

} // namespace

int main()
{
    try {
        test_methods_and_reweighting();
        test_one_global_hybrid_normalization();
        test_ties_duplicates_and_rmsd_independence();
        test_evaluation_top_k_and_oracle();
        test_validation();
        std::cout << "Global conformer ensemble ranking and evaluation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
