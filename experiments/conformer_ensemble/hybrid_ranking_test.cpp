#include "hybrid_ranking.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace conformer_ensemble;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <=
                1e-12 * (1.0 + std::abs(expected)), "Unexpected Hybrid score");
}

void rejects(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("Expected invalid ranking input to be rejected");
}

EnsemblePose pose(std::size_t record, std::size_t id, double weighted_interaction,
                  double gaussian, double repulsion = 0.0)
{
    EnsemblePose row;
    row.record_index = record;
    row.conformer_id = "conf-" + std::to_string(record);
    row.scored.pose_id = id;
    row.scored.pose.fast_weighted_score = 100.0 - double(id);
    row.scored.pose.weighted_interaction_score.weighted_score = weighted_interaction;
    row.scored.vinardo = {gaussian + repulsion, gaussian, repulsion, 0.0, 0.0};
    row.rmsd_heavy = 4.0;
    return row;
}

void test_complete_pool_and_core_normalization()
{
    std::vector<EnsemblePose> rows;
    for (std::size_t record = 1; record <= 2; ++record)
        for (std::size_t id = 0; id < 15; ++id)
            rows.push_back(pose(record, id, double(record * id),
                                -double(3 * id + record), double(id)));
    const auto actual = rank_ensemble_poses(rows);
    require(actual.all.size() == 30 && actual.local.size() == 20 && actual.global.size() == 10,
            "All clash survivors must participate before local selection");
    PoseRescoringResult complete;
    complete.input_pose_count = rows.size();
    complete.weighted_interaction_calculated = complete.vinardo_calculated = true;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        complete.poses.push_back(rows[i].scored);
        complete.poses.back().pose_id = i;
    }
    const auto expected = rank_rescored_poses(
        complete, {PoseScoreMethod::Hybrid, 0.5, 0.3}, rows.size());
    std::map<std::size_t, std::size_t> local_counts;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& original = rows[expected[i].scored.pose_id];
        const auto& ranked = actual.all[i];
        require(ranked.pose.record_index == original.record_index &&
                    ranked.pose.scored.pose_id == original.scored.pose_id,
                "Ensemble ranking must equal the engine ranking on the full union");
        require(ranked.global_rank == i + 1 &&
                    ranked.local_rank == ++local_counts[original.record_index],
                "Local and global ranks must come from the same score order");
        close(ranked.hybrid, *expected[i].hybrid);
        close(ranked.vinardo_rep50, vinardo_adjusted_score(original.scored.vinardo, 0.5));
        if (i < actual.global.size()) {
            require(actual.global[i].pose.record_index == ranked.pose.record_index &&
                        actual.global[i].pose.scored.pose_id == ranked.pose.scored.pose_id,
                    "Local Top-K merge must preserve the true global Top-X");
            close(actual.global[i].hybrid, ranked.hybrid);
        }
    }
    const auto one = rank_ensemble_poses(rows, 10, 1);
    close(one.global[0].hybrid, actual.global[0].hybrid);
    const auto enlarged = rank_ensemble_poses(rows, 15, 10);
    for (std::size_t i = 0; i < actual.global.size(); ++i)
        close(enlarged.global[i].hybrid, actual.global[i].hybrid);

    const auto separate = rank_ensemble_poses(
        std::vector<EnsemblePose>(rows.begin(), rows.begin() + 15), 10, 10);
    bool pooled_differs = false;
    for (const auto& row : separate.all)
        for (const auto& global : actual.all)
            if (row.pose.record_index == global.pose.record_index &&
                row.pose.scored.pose_id == global.pose.scored.pose_id &&
                std::abs(row.hybrid - global.hybrid) > 1e-6)
                pooled_differs = true;
    require(pooled_differs, "Fixture must detect incorrect per-conformer normalization");
}

void test_no_fast_prefilter_and_no_rmsd_selection()
{
    std::vector<EnsemblePose> rows;
    for (std::size_t id = 0; id < 12; ++id)
        rows.push_back(pose(1, id, 5.0, -double(2 * id), double(id)));
    const auto ranked = rank_ensemble_poses(rows);
    require(ranked.global[0].pose.scored.pose_id == 11,
            "Hybrid must retain a winner outside the Fast Top-10");
    rows[0].rmsd_heavy = 0.1;
    rows[11].rmsd_heavy = 100.0;
    std::reverse(rows.begin(), rows.end());
    const auto changed = rank_ensemble_poses(rows);
    for (std::size_t i = 0; i < ranked.all.size(); ++i) {
        require(changed.all[i].pose.scored.pose_id == ranked.all[i].pose.scored.pose_id,
                "RMSD and input order must not change Hybrid selection");
        close(changed.all[i].hybrid, ranked.all[i].hybrid);
    }
}

void test_ties_empty_and_validation()
{
    const auto tied = rank_ensemble_poses({pose(2, 0, 3, -1), pose(1, 4, 3, -1), pose(1, 0, 3, -1)});
    require(tied.global.size() == 3 && tied.global[0].pose.record_index == 1 &&
                tied.global[0].pose.scored.pose_id == 0 && tied.global[1].pose.scored.pose_id == 4 &&
                tied.global[2].pose.record_index == 2,
            "Ties must retain distinct conformer-local IDs and deterministic order");
    for (const auto& row : tied.all) close(row.hybrid, 0.0);
    const auto empty = rank_ensemble_poses({});
    require(empty.all.empty() && empty.local.empty() && empty.global.empty(),
            "Empty pool must remain empty");
    rejects([] { rank_ensemble_poses({}, 0, 0); });
    rejects([] { rank_ensemble_poses({}, 5, 10); });
    rejects([] { rank_ensemble_poses({pose(1, 0, 1, -1), pose(1, 0, 1, -1)}); });
    rejects([] {
        auto row = pose(1, 0, 1, -1);
        row.scored.pose.weighted_interaction_score.weighted_score = std::numeric_limits<double>::quiet_NaN();
        rank_ensemble_poses({row});
    });
    rejects([] {
        auto row = pose(1, 0, 1, -1);
        row.rmsd_heavy = std::numeric_limits<double>::infinity();
        rank_ensemble_poses({row});
    });
    rejects([] {
        auto row = pose(1, 1, 1, -1);
        row.conformer_id = "other";
        rank_ensemble_poses({pose(1, 0, 1, -1), row});
    });
}

} // namespace

int main()
{
    try {
        test_complete_pool_and_core_normalization();
        test_no_fast_prefilter_and_no_rmsd_selection();
        test_ties_empty_and_validation();
        std::cout << "Global conformer Hybrid ranking tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Global conformer Hybrid ranking test failed: " << error.what() << '\n';
        return 1;
    }
}
