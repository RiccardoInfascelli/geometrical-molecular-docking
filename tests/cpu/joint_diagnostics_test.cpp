#include "joint_diagnostics.hpp"
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
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 1e-12 * (1 + std::abs(expected)),
        "Unexpected diagnostic numeric value");
}

void rejects(const std::function<void()>& operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
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

RescoredPose pose(std::size_t id, double fast, double weighted, double vinardo, double repulsion = 0) {
    RescoredPose row;
    row.pose_id = id;
    row.pose.fast_weighted_score = fast;
    row.pose.weighted_interaction_score.weighted_score = weighted;
    row.vinardo.total = vinardo;
    row.vinardo.repulsion = repulsion;
    row.vinardo.gaussian = vinardo - repulsion;
    return row;
}

PoseRescoringResult pool(std::vector<RescoredPose> rows) {
    PoseRescoringResult result;
    result.input_pose_count = rows.size();
    result.poses = std::move(rows);
    result.weighted_interaction_calculated = result.vinardo_calculated = true;
    return result;
}

using Row = std::map<std::string, std::string>;
std::vector<Row> read_csv(const std::string& text) {
    const auto split = [](const std::string& line) {
        std::vector<std::string> fields;
        std::size_t start = 0;
        for (std::size_t end = 0; end <= line.size(); ++end) {
            if (end == line.size() || line[end] == ',') {
                fields.push_back(line.substr(start, end - start));
                start = end + 1;
            }
        }
        return fields;
    };
    std::istringstream input(text);
    std::string line;
    require(bool(std::getline(input, line)), "Missing CSV header");
    const auto header = split(line);
    std::vector<Row> result;
    while (std::getline(input, line)) {
        const auto fields = split(line);
        require(header.size() == fields.size(), "CSV column count differs");
        Row row;
        for (std::size_t i = 0; i < header.size(); ++i) row.emplace(header[i], fields[i]);
        result.push_back(std::move(row));
    }
    return result;
}

struct Analysis {
    std::vector<Row> rankings, topk, summary;
    Analysis(const PoseRescoringResult& scores, const std::vector<joint_diagnostics::PoseMetrics>& metrics) {
        std::ostringstream r, t, s;
        joint_diagnostics::write_analysis_csvs(r, t, s, scores, metrics);
        rankings = read_csv(r.str()); topk = read_csv(t.str()); summary = read_csv(s.str());
    }
};

const Row& find(const std::vector<Row>& rows, std::string_view pool_name,
    std::string_view method, const char* index = nullptr, std::string_view value = "") {
    const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
        return row.at("pool") == pool_name && row.at("method") == method &&
            (!index || row.at(index) == value);
    });
    require(found != rows.end(), "Missing diagnostic CSV row");
    return *found;
}

void test_metric_geometry_and_mapping() {
    const auto ligand = molecule();
    auto original = pose(7, 1, 2, 3);
    auto translated = original;
    translated.pose_id = 9;
    translated.pose.transform.translation = {3, 4, 0};
    auto rotated = original;
    rotated.pose_id = 11;
    rotated.pose.transform.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    rotated.pose.transform.translation = {1.5, 0.5, 0};
    const auto scores = pool({original, translated, rotated});
    const auto metrics = joint_diagnostics::evaluate_pose_metrics(scores, ligand, ligand);
    close(metrics[0].rmsd_heavy_A, 0); close(metrics[0].center_distance_A, 0);
    close(metrics[1].rmsd_heavy_A, 5); close(metrics[1].center_distance_A, 5);
    close(metrics[2].rmsd_heavy_A, std::sqrt(2.5)); close(metrics[2].center_distance_A, 0);
    require(metrics[1].pose_id == 9 && metrics[2].pose_id == 11, "Metric IDs changed");
    auto moved_hydrogen = ligand;
    moved_hydrogen.getConformer().setAtomPos(2, {-900, 500, 600});
    const auto no_h = joint_diagnostics::evaluate_pose_metrics(scores, ligand, moved_hydrogen);
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        close(no_h[i].rmsd_heavy_A, metrics[i].rmsd_heavy_A);
        close(no_h[i].center_distance_A, metrics[i].center_distance_A);
    }
    auto wrong = ligand;
    wrong.getAtomWithIdx(0)->setAtomicNum(7);
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(scores, ligand, wrong); });
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(pool({}), ligand, wrong); });
    wrong = ligand;
    wrong.clearConformers();
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(scores, ligand, wrong); });
    wrong = ligand;
    wrong.getConformer().setAtomPos(0, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(scores, ligand, wrong); });
    auto invalid = scores;
    invalid.poses[0].pose.transform.translation[0] = std::numeric_limits<double>::infinity();
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(invalid, ligand, ligand); });
    require(scores.poses[1].pose.fast_weighted_score == 1 && scores.poses[1].pose.transform.translation[0] == 3,
        "Evaluation mutated scores or transforms");
    std::ostringstream csv;
    joint_diagnostics::write_pose_csv(csv, scores, metrics);
    const auto rows = read_csv(csv.str());
    std::ostringstream legacy_csv;
    write_pose_scores_csv(legacy_csv, scores, 1);
    const auto legacy_rows = read_csv(legacy_csv.str());
    require(rows.size() == scores.input_pose_count, "Export omitted a survivor");
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (const auto& [key, value] : legacy_rows[i])
            require(rows[i].at(key) == value, "Diagnostics changed a legacy joint pose column");
    require(rows[1].at("pose_id") == "9" && rows[1].at("tx") == "3", "Pose/transform association changed");
    close(std::stod(rows[1].at("rmsd_heavy_A")), 5);
    close(std::stod(rows[2].at("center_distance_A")), 0);
    auto bad_metrics = metrics;
    bad_metrics.pop_back();
    rejects([&] { std::ostringstream out; joint_diagnostics::write_pose_csv(out, scores, bad_metrics); });
    bad_metrics = metrics;
    std::swap(bad_metrics[0], bad_metrics[1]);
    rejects([&] { Analysis rejected(scores, bad_metrics); });
    auto duplicate = scores;
    duplicate.poses[1].pose_id = 7;
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(duplicate, ligand, ligand); });
    auto truncated = scores;
    ++truncated.input_pose_count;
    rejects([&] { joint_diagnostics::evaluate_pose_metrics(truncated, ligand, ligand); });
}

void test_prefixes_ids_and_thresholds() {
    const auto scores = pool({pose(9, 10, 3, 3), pose(4, 9, 2, 2), pose(2, 8, 1, 1)});
    const std::vector<joint_diagnostics::PoseMetrics> metrics{{9, 4, 6}, {4, 2, 5}, {2, 3, 2}};
    const Analysis analysis(scores, metrics);
    require(analysis.rankings.size() == 30 && analysis.topk.size() == 200 && analysis.summary.size() == 10,
        "Unexpected number of diagnostic rows");
    const auto& top2 = find(analysis.topk, "full", "fast", "k", "2");
    require(top2.at("rmsd_pose_id") == "4" && top2.at("rmsd_rank") == "2" &&
        top2.at("rmsd_lt_2") == "0" && top2.at("rmsd_lt_3") == "1" && top2.at("center_lt_5") == "0",
        "Strict prefix thresholds differ");
    const auto& top3 = find(analysis.topk, "full", "fast", "k", "3");
    require(top3.at("rmsd_pose_id") == "4" && top3.at("center_pose_id") == "2" &&
        top3.at("center_rank") == "3", "Independent metric argmins were conflated");
    const auto& top20 = find(analysis.topk, "full", "fast", "k", "20");
    require(top20.at("effective_k") == "3" && top20.at("center_lt_5") == "1", "Short prefix coverage differs");
    for (const auto* pool_name : {"full", "fast_top500"}) {
        for (const auto* method : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
            double rmsd = std::numeric_limits<double>::infinity(), center = rmsd;
            for (int k = 1; k <= 20; ++k) {
                const auto& prefix = find(analysis.topk, pool_name, method, "k", std::to_string(k));
                require(std::stod(prefix.at("min_rmsd_heavy_A")) <= rmsd &&
                    std::stod(prefix.at("min_center_distance_A")) <= center, "Cumulative minima are not monotone");
                rmsd = std::stod(prefix.at("min_rmsd_heavy_A"));
                center = std::stod(prefix.at("min_center_distance_A"));
            }
        }
    }
    auto unrelated_metrics = metrics;
    for (auto& value : unrelated_metrics) { value.rmsd_heavy_A = 99 - value.pose_id; value.center_distance_A = value.pose_id; }
    const Analysis changed(scores, unrelated_metrics);
    for (std::size_t i = 0; i < analysis.rankings.size(); ++i) {
        require(analysis.rankings[i].at("pose_id") == changed.rankings[i].at("pose_id") &&
            analysis.rankings[i].at("score") == changed.rankings[i].at("score"), "Metrics influenced ranking");
    }
    const auto ties = pool({pose(9, 1, 1, 1), pose(3, 1, 1, 1)});
    const Analysis tied(ties, {{9, 2, 2}, {3, 2, 2}});
    for (const auto* method : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"})
        require(find(tied.rankings, "full", method, "rank", "1").at("pose_id") == "3", "Score tie is not deterministic");
    require(find(tied.summary, "full", "fast").at("best_rmsd_pose_id") == "3", "Global metric tie did not use pose_id");
}

void test_subset_and_hybrid_normalization() {
    std::vector<RescoredPose> rows;
    std::vector<joint_diagnostics::PoseMetrics> metrics;
    for (std::size_t id = 0; id <= 500; ++id) {
        rows.push_back(pose(id, 1, static_cast<double>(id), static_cast<double>((id * 37) % 101), id % 3));
        metrics.push_back({id, static_cast<double>(501 - id), static_cast<double>(id + 1)});
    }
    rows.back().pose.weighted_interaction_score.weighted_score = 1e6;
    const auto full = pool(rows);
    const Analysis analysis(full, metrics);
    require(find(analysis.summary, "full", "fast").at("pool_size") == "501" &&
        find(analysis.summary, "fast_top500", "fast").at("pool_size") == "500", "Fast subset size differs");
    require(find(analysis.summary, "full", "fast").at("best_rmsd_pose_id") == "500" &&
        find(analysis.summary, "fast_top500", "fast").at("best_rmsd_pose_id") == "499", "Fast selection did not resolve boundary tie by ID");
    rows.pop_back();
    const auto subset = pool(rows);
    const auto expected = rank_rescored_poses(subset, {PoseScoreMethod::Hybrid, 0.5, 0.3}, 20);
    const auto full_ranked = rank_rescored_poses(full, {PoseScoreMethod::Hybrid, 0.5, 0.3}, 501);
    bool normalization_changed = false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& actual = find(analysis.rankings, "fast_top500", "hybrid", "rank", std::to_string(i + 1));
        require(std::stoull(actual.at("pose_id")) == expected[i].scored.pose_id, "Subset Hybrid ranking differs from native ranker");
        close(std::stod(actual.at("score")), *expected[i].hybrid);
        const auto full_value = std::find_if(full_ranked.begin(), full_ranked.end(), [&](const auto& value) {
            return value.scored.pose_id == expected[i].scored.pose_id;
        });
        normalization_changed = normalization_changed || std::abs(*full_value->hybrid - *expected[i].hybrid) > 1e-5;
    }
    require(normalization_changed, "Fixture did not exercise separate Hybrid normalization");
    require(full.poses.size() == 501 && full.poses.back().pose_id == 500, "Analysis mutated the full survivor pool");
}

void test_empty_pool() {
    const auto empty = pool({});
    const auto ligand = molecule();
    const auto metrics = joint_diagnostics::evaluate_pose_metrics(empty, ligand, ligand);
    const Analysis analysis(empty, metrics);
    require(analysis.rankings.empty() && analysis.summary.size() == 10 && analysis.topk.size() == 200,
        "Empty pool output is incomplete");
    for (const auto& row : analysis.summary)
        require(row.at("status") == "no_poses" && row.at("pool_size") == "0" &&
            row.at("best_rmsd_heavy_A").empty() && row.at("top1_pose_id").empty(), "Empty metrics must not become zero");
    for (const auto& row : analysis.topk)
        require(row.at("status") == "no_poses" && row.at("effective_k") == "0" &&
            row.at("min_rmsd_heavy_A").empty() && row.at("rmsd_lt_2").empty(), "Empty prefix has invented metrics");
    std::ostringstream csv;
    joint_diagnostics::write_pose_csv(csv, empty, metrics);
    require(read_csv(csv.str()).empty(), "Empty pose export contains rows");
}

} // namespace

int main() {
    try {
        test_metric_geometry_and_mapping();
        test_prefixes_ids_and_thresholds();
        test_subset_and_hybrid_normalization();
        test_empty_pool();
        std::cout << "Joint diagnostics tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
