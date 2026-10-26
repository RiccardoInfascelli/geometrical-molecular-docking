#include "ablation.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace scoring_ablation;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected)
{
    require(std::isfinite(actual) &&
                std::abs(actual - expected) <= 1e-12 * (1.0 + std::abs(expected)),
            "Unexpected numeric ablation result");
}

void rejects(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("Expected invalid input to be rejected");
}

std::size_t family(double WeightedInteractionWeights::* member)
{
    for (std::size_t i = 0; i < weighted_interaction_families().size(); ++i)
        if (weighted_interaction_families()[i].weight == member) return i;
    throw std::runtime_error("Missing WeightedInteraction family");
}

std::size_t term(const std::string& name)
{
    for (std::size_t i = 0; i < vinardo_terms().size(); ++i)
        if (vinardo_terms()[i] == name) return i;
    throw std::runtime_error("Missing Vinardo term");
}

interaction::interaction_area area(interaction::interaction_smarts_pattern pattern,
                                   interaction::point3D center, bool aromatic = false)
{
    interaction::interaction_area result{};
    result.pattern = pattern;
    result.pattern_class = aromatic ? interaction::interaction_class_pattern::Aromatic :
                                     interaction::interaction_class_pattern::Hydrophobic;
    result.centroid = center;
    result.num_atoms = aromatic ? 3 : 1;
    result.x[0] = center.x + (aromatic ? 1.0f : 0.0f);
    result.y[0] = center.y;
    result.z[0] = center.z;
    if (aromatic) {
        result.x[1] = center.x; result.y[1] = center.y + 1.0f; result.z[1] = center.z;
        result.x[2] = center.x; result.y[2] = center.y; result.z[2] = center.z;
    }
    return result;
}

void test_weighted_interaction_maximum()
{
    using Pattern = interaction::interaction_smarts_pattern;
    const std::vector<interaction::interaction_area> ligand{area(Pattern::Cation, {0, 0, 0})};
    const std::vector<interaction::interaction_area> protein{
        area(Pattern::Anion, {3, 0, 0}), area(Pattern::Aromatic_ring_6, {0, 0, 3}, true)};
    const auto full = score_weighted_interactions(protein, ligand);
    require(full.ionic == 1 && full.cation_pi == 1 && full.satisfied_ligand_areas == 1,
            "Fixture must form two interaction families on one physical ligand area");
    close(full.weighted_score, 3.0); // max(ionic=3, cation-pi=2.5), not their sum.
    const auto no_ionic = score_weighted_interactions(
        protein, ligand, weighted_interaction_weights_without(family(&WeightedInteractionWeights::ionic)));
    const auto no_cation_pi = score_weighted_interactions(
        protein, ligand, weighted_interaction_weights_without(family(&WeightedInteractionWeights::cation_pi)));
    close(no_ionic.weighted_score, 2.5);
    close(no_cation_pi.weighted_score, 3.0);
    require(no_ionic.ionic == full.ionic && no_ionic.cation_pi == full.cation_pi &&
                no_ionic.satisfied_ligand_areas == full.satisfied_ligand_areas &&
                no_ionic.total_interactions == full.total_interactions,
            "Zero weight must preserve geometric contacts and coverage");
    WeightedInteractionWeights custom;
    custom.ionic = 0.25;
    custom.cation_pi = 4.0;
    close(score_weighted_interactions(protein, ligand, custom).weighted_score, 4.0);
    custom.cation_pi = 0.0;
    close(score_weighted_interactions(protein, ligand, custom).weighted_score, 0.25);
    for (std::size_t i = 0; i < weighted_interaction_families().size(); ++i) {
        const auto weights = weighted_interaction_weights_without(i);
        for (std::size_t j = 0; j < weighted_interaction_families().size(); ++j)
            close(weights.*weighted_interaction_families()[j].weight,
                  i == j ? 0.0 : WeightedInteractionWeights{}.*weighted_interaction_families()[j].weight);
    }
    rejects([] { weighted_interaction_weights_without(weighted_interaction_families().size()); });
}

void test_vinardo_removal()
{
    const VinardoLikeScore full{2.0, -2.0, 8.0, -1.0, -3.0};
    const auto repulsion = term("repulsion");
    for (std::size_t i = 0; i < vinardo_terms().size(); ++i) {
        const auto removed = vinardo_without(full, i);
        close(removed.total, removed.gaussian + removed.repulsion +
                               removed.hydrophobic + removed.hydrogen_bond);
        const std::string name = vinardo_terms()[i];
        const double original = name == "gaussian" ? full.gaussian :
            name == "repulsion" ? full.repulsion :
            name == "hydrophobic" ? full.hydrophobic : full.hydrogen_bond;
        close(removed.total, full.total - original);
        for (double alpha : {1.0, 0.5})
            close(vinardo_adjusted_score(removed, alpha),
                  vinardo_adjusted_score(full, alpha) - (i == repulsion ? alpha : 1.0) * original);
    }
    close(full.total, 2.0);
    const VinardoLikeScore dominated{1e20, -2.0, 1e20, -1.0, -3.0};
    close(vinardo_without(dominated, repulsion).total, -6.0);
    rejects([&] { vinardo_without(full, vinardo_terms().size()); });
}

PoseAblation pose(std::size_t id, double weighted_interaction, double energy)
{
    PoseAblation result;
    result.baseline.pose_id = id;
    result.baseline.pose.fast_weighted_score = 100.0 - id;
    result.baseline.pose.weighted_interaction_score.weighted_score = weighted_interaction;
    result.baseline.pose.transform.translation = {double(id), -double(id), 0.5};
    result.baseline.vinardo = {energy, energy, 0.0, 0.0, 0.0};
    for (auto& score : result.weighted_interaction_without) score = result.baseline.pose.weighted_interaction_score;
    return result;
}

void test_fixed_pool_and_full_ranking()
{
    std::vector<PoseAblation> poses;
    for (std::size_t i = 0; i < 12; ++i) {
        auto row = pose(100 + i, 12.0 - i, -double(i));
        row.weighted_interaction_without[0].weighted_score = i == 11 ? 1.0 : 0.0;
        poses.push_back(row);
    }
    const auto full = make_pool(poses);
    const auto changed = make_pool(poses, RemovedScore::WeightedInteraction, 0);
    const auto vinardo = make_pool(poses, RemovedScore::Vinardo, term("gaussian"));
    require(full.poses.size() == 12 && changed.poses.size() == 12 && vinardo.poses.size() == 12 &&
                full.input_pose_count == 12 && changed.input_pose_count == 12 && vinardo.input_pose_count == 12,
            "Ablation must retain all supplied poses");
    require(full.weighted_interaction_calculated && full.vinardo_calculated, "Calculated-score flags missing");
    for (std::size_t i = 0; i < poses.size(); ++i) {
        for (const auto* pool : {&full, &changed, &vinardo}) {
            const auto& row = pool->poses[i];
            require(row.pose_id == poses[i].baseline.pose_id &&
                        row.pose.transform.rotation == poses[i].baseline.pose.transform.rotation &&
                        row.pose.transform.translation == poses[i].baseline.pose.transform.translation,
                    "Ablation must preserve IDs, transforms and input order");
            close(row.pose.fast_weighted_score, poses[i].baseline.pose.fast_weighted_score);
        }
        close(changed.poses[i].pose.weighted_interaction_score.weighted_score, i == 11 ? 1.0 : 0.0);
        close(changed.poses[i].vinardo.total, full.poses[i].vinardo.total);
        close(vinardo.poses[i].pose.weighted_interaction_score.weighted_score, full.poses[i].pose.weighted_interaction_score.weighted_score);
        close(vinardo.poses[i].vinardo.total, 0.0);
    }
    const auto original_top = rank_rescored_poses(full, {PoseScoreMethod::WeightedInteraction}, 10);
    for (const auto& row : original_top)
        require(row.scored.pose_id != 111, "Fixture winner must be outside the original top ten");
    require(rank_rescored_poses(changed, {PoseScoreMethod::WeightedInteraction}, 1)[0].scored.pose_id == 111,
            "Counterfactual ranking must consider poses outside the original top ten");
}

void test_hybrid_normalization_and_empty()
{
    std::vector<PoseAblation> rows{pose(9, 0, -1), pose(8, 2, -1), pose(7, 100, -1)};
    rows[0].weighted_interaction_without[0].weighted_score = 0;
    rows[1].weighted_interaction_without[0].weighted_score = rows[2].weighted_interaction_without[0].weighted_score = 2;
    const auto changed = make_pool(rows, RemovedScore::WeightedInteraction, 0);
    const auto all = rank_rescored_poses(changed, {PoseScoreMethod::Hybrid}, 3);
    for (const auto& row : all)
        close(*row.hybrid, 0.3 * (row.scored.pose_id == 9 ? -std::sqrt(2.0) : 1.0 / std::sqrt(2.0)));
    const auto top = rank_rescored_poses(changed, {PoseScoreMethod::Hybrid}, 1);
    require(top[0].scored.pose_id == 7, "Hybrid tie must use original pose ID");
    close(*top[0].hybrid, *all[0].hybrid); // top_k must not alter normalization.
    std::vector<PoseAblation> repulsive{pose(9, 2, -10), pose(8, 2, -8), pose(7, 2, 90)};
    repulsive[2].baseline.vinardo.gaussian = -8;
    repulsive[2].baseline.vinardo.repulsion = 98;
    const auto without_repulsion = make_pool(repulsive, RemovedScore::Vinardo, term("repulsion"));
    for (const auto& row : rank_rescored_poses(without_repulsion, {PoseScoreMethod::Hybrid, 0.5}, 3))
        close(*row.hybrid, 0.7 * (row.scored.pose_id == 9 ? std::sqrt(2.0) : -1.0 / std::sqrt(2.0)));
    const auto constant = make_pool({pose(8, 2, -1), pose(3, 2, -1)});
    const auto tied = rank_rescored_poses(constant, {PoseScoreMethod::Hybrid}, 1);
    require(tied[0].scored.pose_id == 3, "Constant-score tie must use original pose ID");
    close(*tied[0].hybrid, 0.0);
    require(rank_rescored_poses(make_pool({}), {PoseScoreMethod::Hybrid}, 1).empty(),
            "An empty supplied pool must remain empty");
}

std::string csv_header()
{
    return "record_index,pose_id,fast_score,weighted_interaction_score,vinardo_like_score,clash_fraction,occupied_atoms,"
           "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
}

std::string csv_row(std::size_t record, std::size_t id, const std::string& fast = "5")
{
    return std::to_string(record) + ',' + std::to_string(id) + ',' + fast +
           ",3,-2,0,0,1,0,0,0,1,0,0,0,1,2,3,4\n";
}

void test_csv_validation()
{
    std::istringstream source(csv_header() + csv_row(1, 90) + csv_row(2, 7) +
                              csv_row(1, 3) + csv_row(2, 7));
    const auto selected = read_pose_pool(source, 1);
    require(selected.size() == 2 && selected[0].scored.pose_id == 90 && selected[1].scored.pose_id == 3,
            "CSV record selection must preserve IDs and input order");
    close(selected[0].scored.pose.transform.translation[2], 4);
    close(*selected[0].saved_weighted_interaction, 3);
    close(*selected[0].saved_vinardo, -2);
    std::istringstream header_only(csv_header());
    require(read_pose_pool(header_only, 1).empty(), "A CSV with no poses must remain empty");
    rejects([] {
        std::istringstream duplicate(csv_header() + csv_row(1, 7) + csv_row(1, 7));
        read_pose_pool(duplicate, 1);
    });
    for (const std::string fast : {"nan", "inf", "-inf", "5junk", ""})
        rejects([&] {
            std::istringstream invalid(csv_header() + csv_row(1, 2, fast));
            read_pose_pool(invalid, 1);
        });
    rejects([] {
        std::istringstream missing("record_index,pose_id,fast_score\n1,0,3\n");
        read_pose_pool(missing, 1);
    });
    rejects([] {
        std::istringstream invalid(csv_header() + csv_row(1, 2));
        read_pose_pool(invalid, 0);
    });
    rejects([] {
        auto text = csv_header() + csv_row(1, 2);
        const auto location = text.rfind(",2,3,4");
        text.replace(location, 6, ",2,nan,4");
        std::istringstream invalid(text);
        read_pose_pool(invalid, 1);
    });
    const std::string gpu_header =
        "record_index,molecule_id,rank,alignment_index,fast_weighted_score,clash_fraction,occupied_atoms,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
    const std::string gpu_row =
        "1,\"ligand, \"\"quoted\"\"\nsecond line\",1,31,7,0,0,1,0,0,0,1,0,0,0,1,2,3,4\n";
    std::istringstream gpu_source(gpu_header + gpu_row);
    const auto gpu = read_pose_pool(gpu_source, 1);
    require(gpu.size() == 1 && gpu[0].scored.pose_id == 31 &&
                !gpu[0].saved_weighted_interaction && !gpu[0].saved_vinardo,
            "Plain GPU CSV must retain alignment ID without requiring saved rescoring");
    close(gpu[0].scored.pose.fast_weighted_score, 7);
    close(gpu[0].scored.pose.transform.translation[2], 4);
    rejects([&] {
        std::istringstream duplicate(gpu_header + gpu_row + gpu_row);
        read_pose_pool(duplicate, 1);
    });
}

} // namespace

int main()
{
    try {
        test_weighted_interaction_maximum();
        test_vinardo_removal();
        test_fixed_pool_and_full_ranking();
        test_hybrid_normalization_and_empty();
        test_csv_validation();
        std::cout << "Scoring ablation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Scoring ablation test failed: " << error.what() << '\n';
        return 1;
    }
}
