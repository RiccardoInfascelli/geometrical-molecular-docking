#include "ensemble_diagnostics.hpp"
#include "pose_score_ranking.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12 * (1 + std::abs(expected)),
        "Unexpected ensemble diagnostic value");
}

void rejects(const std::function<void()>& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected diagnostic input rejection");
}

RDKit::RWMol molecule() {
    RDKit::RWMol result;
    result.addAtom(new RDKit::Atom(6), true, true);
    result.addAtom(new RDKit::Atom(8), true, true);
    result.addAtom(new RDKit::Atom(1), true, true);
    result.addBond(0, 1, RDKit::Bond::SINGLE);
    result.addBond(0, 2, RDKit::Bond::SINGLE);
    auto* coordinates = new RDKit::Conformer(3);
    coordinates->set3D(true);
    coordinates->setAtomPos(0, {1, 0, 0});
    coordinates->setAtomPos(1, {0, 2, 0});
    coordinates->setAtomPos(2, {100, 200, 300});
    result.addConformer(coordinates, true);
    result.updatePropertyCache(false);
    return result;
}

EnsemblePose pose(std::size_t conformer, std::size_t id, double fast, double weighted,
    double vinardo, double repulsion, double rmsd, double center) {
    EnsemblePose value;
    value.conformer_id = conformer;
    value.pose_id = value.scored.pose_id = id;
    value.alignment_index = 3 * id + 1;
    value.scored.pose.fast_weighted_score = fast;
    value.scored.pose.weighted_interaction_score.weighted_score = weighted;
    value.scored.vinardo.total = vinardo;
    value.scored.vinardo.repulsion = repulsion;
    value.scored.vinardo.gaussian = vinardo - repulsion;
    value.scored.pose.transform.translation = {center, 0, 0};
    value.rmsd = rmsd;
    return value;
}

ConformerDockingResult docking(std::vector<EnsemblePose> poses) {
    ConformerDockingResult result;
    result.survivors = poses.size();
    result.poses = std::move(poses);
    for (const auto& value : result.poses) result.rescored_ids.push_back(value.pose_id);
    return result;
}

using Row = std::map<std::string, std::string>;
std::vector<std::string> fields(std::string_view line) {
    std::vector<std::string> values;
    std::string value;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                value += '"'; ++i;
            } else quoted = !quoted;
        } else if (line[i] == ',' && !quoted) {
            values.push_back(value); value.clear();
        } else value += line[i];
    }
    require(!quoted, "Unclosed CSV quote");
    values.push_back(value);
    return values;
}

std::vector<Row> table(const std::string& text) {
    std::istringstream input(text);
    std::string line;
    require(bool(std::getline(input, line)), "Missing CSV header");
    const auto header = fields(line);
    std::vector<Row> rows;
    while (std::getline(input, line)) {
        const auto values = fields(line);
        require(values.size() == header.size(), "Diagnostic CSV field count differs");
        Row row;
        for (std::size_t i = 0; i < header.size(); ++i) row.emplace(header[i], values[i]);
        rows.push_back(std::move(row));
    }
    return rows;
}

struct Analysis {
    std::vector<Row> summary, topk;
    Analysis(const std::vector<conformer_diagnostics::Input>& inputs,
        const RDKit::ROMol& reference, const std::string& complex = "test-complex") {
        std::ostringstream s, t;
        conformer_diagnostics::write_diagnostics(s, t, complex, inputs, reference);
        summary = table(s.str());
        topk = table(t.str());
    }
};

const Row& find(const std::vector<Row>& rows, std::size_t budget, bool original,
    std::string_view pool, std::string_view method, std::size_t k = 0) {
    const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
        return row.at("budget_requested") == std::to_string(budget) &&
            row.at("original_included") == (original ? "1" : "0") &&
            row.at("pool") == pool && row.at("method") == method &&
            (!k || row.at("k") == std::to_string(k));
    });
    require(found != rows.end(), "Missing diagnostic condition/budget/method row");
    return *found;
}

std::vector<RankedPoseScore> expected_ranking(
    const std::vector<conformer_diagnostics::Input>& inputs, std::size_t budget, bool original,
    bool fast500, const PoseScoreRankingOptions& options) {
    std::vector<const EnsemblePose*> all, admitted;
    std::size_t accepted = 0;
    for (const auto& input : inputs) {
        for (const auto& value : input.docking->poses) all.push_back(&value);
        const bool keep = input.conformer_id == 0 ? original : ++accepted <= budget;
        if (keep)
            for (const auto& value : input.docking->poses) admitted.push_back(&value);
    }
    const auto canonical = [](const auto* a, const auto* b) {
        return std::tie(a->conformer_id, a->pose_id) < std::tie(b->conformer_id, b->pose_id);
    };
    std::sort(all.begin(), all.end(), canonical);
    std::sort(admitted.begin(), admitted.end(), canonical);
    PoseRescoringResult scores;
    scores.weighted_interaction_calculated = scores.vinardo_calculated = true;
    for (const auto* value : admitted) {
        auto row = value->scored;
        row.pose_id = static_cast<std::size_t>(std::find(all.begin(), all.end(), value) - all.begin());
        scores.poses.push_back(row);
    }
    if (fast500) {
        std::vector<PoseScoreEntry> entries;
        for (const auto& row : scores.poses) entries.push_back({row.pose_id, row.pose.fast_weighted_score});
        auto kept = top_pose_score_indices(entries, 500);
        std::sort(kept.begin(), kept.end());
        std::vector<RescoredPose> selected;
        for (auto i : kept) selected.push_back(scores.poses[i]);
        scores.poses = std::move(selected);
    }
    scores.input_pose_count = scores.poses.size();
    return rank_rescored_poses(scores, options, 20);
}

void test_conditions_prefixes_and_metrics() {
    const auto ligand = molecule();
    auto original = docking({pose(0, 0, 10, 9, 20, 30, 4, 4),
        pose(0, 1, 10, 8, 30, 0, 1.5, 0)});
    original.poses[1].scored.pose.transform.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    original.poses[1].scored.pose.transform.translation = {1.5, 0.5, 0};
    auto first = docking({pose(10, 0, 10, 5, 10, 0, 6, 5),
        pose(10, 1, 9, 4, 9, 0, 2, 2), pose(10, 2, 8, 3, 8, 0, 3, 0.2)});
    auto second = docking({pose(2, 0, 100, 100, 100, 0, 1, 1)});
    const std::vector<conformer_diagnostics::Input> inputs{
        {0, &ligand, &original}, {10, &ligand, &first}, {2, &ligand, &second}};
    const Analysis analysis(inputs, ligand, "complex,\"quoted\"");
    require(analysis.summary.size() == 80 && analysis.topk.size() == 1600,
        "Four conditions per budget must produce 80 summary and 1600 Top-K rows");
    const auto& excluded = find(analysis.summary, 1, false, "full", "fast");
    require(excluded.at("complex_id") == "complex,\"quoted\"", "Complex ID CSV escaping differs");
    require(excluded.at("pool_size") == "3" && excluded.at("budget_available") == "1" &&
        excluded.at("budget_complete") == "1" && excluded.at("top1_conformer_id") == "10" &&
        excluded.at("top1_pose_id") == "0" && excluded.at("top1_global_pose_id") == "3",
        "Budget prefix followed numeric conformer IDs or lost stable global identity");
    const auto& included = find(analysis.summary, 1, true, "full", "fast");
    require(included.at("pool_size") == "5" && included.at("top1_conformer_id") == "0" &&
        included.at("top1_pose_id") == "0" && included.at("top1_score_tie_count") == "3" &&
        included.at("original_in_top1_score_tie") == "1", "Original inclusion or deterministic ties differ");
    require(included.at("best_rmsd_conformer_id") == "0" && included.at("best_rmsd_pose_id") == "1",
        "Original survivor was not admitted to the included Oracle");
    close(std::stod(included.at("min_center_distance_A")), 0);
    require(find(analysis.summary, 1, true, "full", "vinardo").at("top1_conformer_id") == "10" &&
        find(analysis.summary, 1, true, "full", "vinardo-rep50").at("top1_conformer_id") == "0",
        "Standard Vinardo and rep50 were conflated");
    const auto& larger = find(analysis.summary, 5, false, "full", "fast");
    require(larger.at("budget_available") == "2" && larger.at("budget_complete") == "0" &&
        larger.at("pool_size") == "4" && larger.at("top1_conformer_id") == "2" &&
        larger.at("top1_global_pose_id") == "2", "Short budget or retained prefix membership differs");
    const auto& first_k = find(analysis.topk, 1, false, "full", "fast", 1);
    require(first_k.at("rmsd_lt_2") == "0" && first_k.at("rmsd_lt_3") == "0" &&
        first_k.at("center_lt_5") == "0", "Threshold equality must not count as success");
    const auto& second_k = find(analysis.topk, 1, false, "full", "fast", 2);
    require(second_k.at("rmsd_lt_2") == "0" && second_k.at("rmsd_lt_3") == "1",
        "RMSD strict thresholds differ");
    const auto& third_k = find(analysis.topk, 1, false, "full", "fast", 3);
    require(third_k.at("rmsd_pose_id") == "1" && third_k.at("rmsd_rank") == "2" &&
        third_k.at("center_pose_id") == "2" && third_k.at("center_rank") == "3",
        "RMSD and center minima lost independent IDs or ranks");
    for (std::size_t budget : {1, 5, 10, 20})
        for (bool with_original : {false, true})
            for (const auto* pool : {"full", "fast_top500"})
                for (const auto* method : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
                    double rmsd = std::numeric_limits<double>::infinity(), center = rmsd;
                    for (std::size_t k = 1; k <= 20; ++k) {
                        const auto& row = find(analysis.topk, budget, with_original, pool, method, k);
                        const double next_rmsd = std::stod(row.at("min_rmsd_heavy_A"));
                        const double next_center = std::stod(row.at("min_center_distance_A"));
                        require(next_rmsd <= rmsd && next_center <= center, "Top-K minima are not cumulative");
                        rmsd = next_rmsd; center = next_center;
                    }
                }
    for (bool with_original : {false, true})
        for (bool fast500 : {false, true}) {
            const auto expected = expected_ranking(inputs, 1, with_original, fast500,
                {PoseScoreMethod::Hybrid, 0.5, 0.3});
            const auto& row = find(analysis.summary, 1, with_original, fast500 ? "fast_top500" : "full", "hybrid");
            require(std::stoull(row.at("top1_global_pose_id")) == expected.front().scored.pose_id,
                "Hybrid differs from the native ranker on the declared population");
            close(std::stod(row.at("top1_score")), *expected.front().hybrid);
        }
    auto unrelated = original;
    unrelated.poses[0].rmsd = 99;
    unrelated.poses[0].scored.pose.transform.translation = {1000, 2000, 3000};
    const Analysis changed({{0, &ligand, &unrelated}, {10, &ligand, &first}, {2, &ligand, &second}}, ligand);
    for (std::size_t i = 0; i < analysis.summary.size(); ++i)
        require(analysis.summary[i].at("top1_global_pose_id") == changed.summary[i].at("top1_global_pose_id") &&
            analysis.summary[i].at("top1_score") == changed.summary[i].at("top1_score"), "Evaluation metrics influenced ranking");
    require(first.poses.size() == 3 && first.poses[0].pose_id == 0 &&
        first.poses[0].scored.pose_id == 0 && first.poses[0].scored.pose.fast_weighted_score == 10,
        "Diagnostics mutated raw poses or scores");
}

void test_global_fast_selection_and_hybrid_normalization() {
    const auto ligand = molecule();
    auto original = docking({pose(0, 0, 100, 100, 30, 0, 0.05, 0.01)});
    std::vector<EnsemblePose> a, b;
    for (std::size_t id = 0; id < 300; ++id)
        a.push_back(pose(10, id, 1, static_cast<double>(id),
            static_cast<double>((id * 37) % 101), id % 3, 1.0 + id, 0.2));
    for (std::size_t id = 0; id < 201; ++id)
        b.push_back(pose(2, id, 1, static_cast<double>(400 + id),
            static_cast<double>((id * 13) % 79), id % 3, 500.0 + id, 0.3));
    a.back().rmsd = 0.1;
    a.back().scored.pose.weighted_interaction_score.weighted_score = 1e6;
    auto first = docking(a), second = docking(b);
    const std::vector<conformer_diagnostics::Input> inputs{
        {0, &ligand, &original}, {10, &ligand, &first}, {2, &ligand, &second}};
    const Analysis analysis(inputs, ligand);
    const auto& full = find(analysis.summary, 5, false, "full", "fast");
    const auto& selected = find(analysis.summary, 5, false, "fast_top500", "fast");
    require(full.at("pool_size") == "501" && selected.at("pool_size") == "500" &&
        full.at("best_rmsd_conformer_id") == "10" && full.at("best_rmsd_pose_id") == "299" &&
        selected.at("best_rmsd_pose_id") != "299", "Fast Top-500 was not global or its boundary tie differs");
    require(find(analysis.summary, 5, true, "full", "fast").at("pool_size") == "502" &&
        find(analysis.summary, 5, true, "fast_top500", "fast").at("pool_size") == "500",
        "Original inclusion was not applied before global Fast selection");
    bool changed_normalization = false;
    for (bool with_original : {false, true}) {
        for (bool fast500 : {false, true}) {
            const auto ranked = expected_ranking(inputs, 5, with_original, fast500,
                {PoseScoreMethod::Hybrid, 0.5, 0.3});
            const auto& row = find(analysis.summary, 5, with_original,
                fast500 ? "fast_top500" : "full", "hybrid");
            require(std::stoull(row.at("top1_global_pose_id")) == ranked.front().scored.pose_id,
                "Hybrid Top-1 did not use the independently normalized current pool");
            close(std::stod(row.at("top1_score")), *ranked.front().hybrid);
        }
        const auto all = expected_ranking(inputs, 5, with_original, false, {PoseScoreMethod::Hybrid, 0.5, 0.3});
        const auto subset = expected_ranking(inputs, 5, with_original, true, {PoseScoreMethod::Hybrid, 0.5, 0.3});
        const auto found = std::find_if(all.begin(), all.end(), [&](const auto& row) {
            return row.scored.pose_id == subset.front().scored.pose_id;
        });
        if (found != all.end())
            changed_normalization = changed_normalization || std::abs(*found->hybrid - *subset.front().hybrid) > 1e-5;
    }
    require(changed_normalization, "Fixture did not exercise Hybrid renormalization");
    require(first.poses.size() == 300 && second.poses.size() == 201 &&
        first.poses.back().scored.pose.weighted_interaction_score.weighted_score == 1e6,
        "Global selection modified raw survivors");
}

void test_empty_pools_and_invalid_inputs() {
    const auto ligand = molecule();
    auto original = docking({}), generated = docking({});
    const std::vector<conformer_diagnostics::Input> inputs{{0, &ligand, &original}, {5, &ligand, &generated}};
    const Analysis empty(inputs, ligand);
    require(empty.summary.size() == 80 && empty.topk.size() == 1600, "Empty condition output is incomplete");
    for (const auto& row : empty.summary)
        require(row.at("status") == "no_poses" && row.at("pool_size") == "0" &&
            row.at("best_rmsd_heavy_A").empty() && row.at("top1_pose_id").empty(),
            "Empty pools must not invent Oracle or Top-1 values");
    for (const auto& row : empty.topk)
        require(row.at("effective_k") == "0" && row.at("min_rmsd_heavy_A").empty() &&
            row.at("center_lt_5").empty(), "Empty prefixes must retain missing metrics");
    const Analysis none_available({{0, &ligand, &original}}, ligand);
    for (const auto& row : none_available.summary)
        require(row.at("budget_available") == "0" && row.at("budget_complete") == "0",
            "No generated conformers must remain an incomplete budget");
    rejects([&] { Analysis invalid({{5, &ligand, &generated}}, ligand); });
    rejects([&] { Analysis invalid({{0, &ligand, &original}, {0, &ligand, &generated}}, ligand); });
    auto wrong = ligand;
    wrong.getAtomWithIdx(0)->setAtomicNum(7);
    rejects([&] { Analysis invalid(inputs, wrong); });
    auto populated = docking({pose(5, 0, 1, 1, 1, 0, 1, 1)});
    populated.rescored_ids.clear();
    rejects([&] { Analysis invalid({{0, &ligand, &original}, {5, &ligand, &populated}}, ligand); });
    populated.rescored_ids = {0};
    ++populated.survivors;
    rejects([&] { Analysis invalid({{0, &ligand, &original}, {5, &ligand, &populated}}, ligand); });
    --populated.survivors;
    populated.poses[0].rmsd = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { Analysis invalid({{0, &ligand, &original}, {5, &ligand, &populated}}, ligand); });
}

} // namespace

int main() {
    try {
        test_conditions_prefixes_and_metrics();
        test_global_fast_selection_and_hybrid_normalization();
        test_empty_pools_and_invalid_inputs();
        std::cout << "Ensemble diagnostic tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
