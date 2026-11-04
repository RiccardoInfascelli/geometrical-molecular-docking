#include "joint_diagnostics.hpp"

#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "pose_score_ranking.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace joint_diagnostics {
namespace {

void validate_pool(const PoseRescoringResult& pool) {
    if (pool.input_pose_count != pool.poses.size())
        throw std::invalid_argument("Joint diagnostics require the complete survivor pool");
    if (!pool.weighted_interaction_calculated || !pool.vinardo_calculated)
        throw std::invalid_argument("Joint diagnostics require WeightedInteraction and Vinardo scores");
    std::unordered_set<std::size_t> ids;
    for (const auto& row : pool.poses) {
        if (!ids.insert(row.pose_id).second)
            throw std::invalid_argument("Duplicate survivor pose_id in joint diagnostics");
        if (!std::isfinite(row.pose.fast_weighted_score) ||
            !std::isfinite(row.pose.weighted_interaction_score.weighted_score))
            throw std::invalid_argument("Nonfinite joint diagnostic score");
        vinardo_adjusted_score(row.vinardo, 1.0);
    }
}

void validate_metrics(const PoseRescoringResult& pool, const std::vector<PoseMetrics>& metrics) {
    validate_pool(pool);
    if (metrics.size() != pool.poses.size())
        throw std::invalid_argument("Joint diagnostic metric count differs from survivors");
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const auto& value = metrics[i];
        if (value.pose_id != pool.poses[i].pose_id)
            throw std::invalid_argument("Joint diagnostic metrics do not match survivor IDs/order");
        if (!std::isfinite(value.rmsd_heavy_A) || value.rmsd_heavy_A < 0 ||
            !std::isfinite(value.center_distance_A) || value.center_distance_A < 0)
            throw std::invalid_argument("Invalid RMSD or centroid distance in joint diagnostics");
    }
}

struct Method {
    const char* name;
    PoseScoreRankingOptions options;
};

const std::array<Method, 5> methods{{
    {"fast", {PoseScoreMethod::Fast, 1.0, 0.3}},
    {"weighted-interaction", {PoseScoreMethod::WeightedInteraction, 1.0, 0.3}},
    {"vinardo", {PoseScoreMethod::Vinardo, 1.0, 0.3}},
    {"vinardo-rep50", {PoseScoreMethod::Vinardo, 0.5, 0.3}},
    {"hybrid", {PoseScoreMethod::Hybrid, 0.5, 0.3}}
}};

double score_value(const RankedPoseScore& row, PoseScoreMethod method) {
    switch (method) {
    case PoseScoreMethod::Fast: return row.scored.pose.fast_weighted_score;
    case PoseScoreMethod::WeightedInteraction: return row.scored.pose.weighted_interaction_score.weighted_score;
    case PoseScoreMethod::Vinardo: return row.vinardo_adjusted.value();
    case PoseScoreMethod::Hybrid: return row.hybrid.value();
    }
    throw std::invalid_argument("Unknown joint diagnostic score method");
}

void write_success(std::ostream& output, const PoseMetrics* value) {
    if (!value) output << ",,,";
    else output << ',' << (value->rmsd_heavy_A < 2.0) << ',' << (value->rmsd_heavy_A < 3.0)
                << ',' << (value->center_distance_A < 5.0);
}

void check_output(const std::ostream& output) {
    if (!output) throw std::runtime_error("Could not write joint diagnostics CSV");
}

} // namespace

std::vector<PoseMetrics> evaluate_pose_metrics(const PoseRescoringResult& pool,
    const RDKit::ROMol& ligand, const RDKit::ROMol& reference) {
    validate_pool(pool);
    validate_gpu_docking_reference(ligand, reference);
    const auto ligand_center = heavy_atom_centroid(ligand);
    const auto reference_center = heavy_atom_centroid(reference);
    std::vector<PoseMetrics> metrics;
    metrics.reserve(pool.poses.size());
    for (const auto& row : pool.poses) {
        GpuRigidTransform transform;
        std::copy(row.pose.transform.rotation.begin(), row.pose.transform.rotation.end(), transform.rotation);
        std::copy(row.pose.transform.translation.begin(), row.pose.transform.translation.end(), transform.translation);
        transform.valid = 1;
        metrics.push_back({row.pose_id, gpu_pose_heavy_atom_rmsd(ligand, transform, reference),
            pose_center_distance(ligand_center, row.pose.transform, reference_center)});
    }
    return metrics;
}

void write_pose_csv(std::ostream& output, const PoseRescoringResult& pool,
    const std::vector<PoseMetrics>& metrics) {
    validate_metrics(pool, metrics);
    output << "record_index,pose_id,fast_score,weighted_interaction_score,vinardo_like_score,"
        "occupied_atoms,clash_fraction,weighted_interaction_total_interactions,"
        "weighted_interaction_satisfied_ligand_areas,weighted_interaction_satisfied_protein_areas,"
        "vinardo_gaussian,vinardo_repulsion,vinardo_hydrophobic,vinardo_hbond,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz,"
        "vinardo_rep50_score,rmsd_heavy_A,center_distance_A\n";
    const auto precision = output.precision();
    output << std::setprecision(17);
    for (std::size_t i = 0; i < pool.poses.size(); ++i) {
        const auto& row = pool.poses[i];
        const auto& pose = row.pose;
        output << "1," << row.pose_id << ',' << pose.fast_weighted_score << ','
            << pose.weighted_interaction_score.weighted_score << ',' << row.vinardo.total << ','
            << pose.occupied_atoms << ',' << pose.clash_fraction << ','
            << pose.weighted_interaction_score.total_interactions << ','
            << pose.weighted_interaction_score.satisfied_ligand_areas << ','
            << pose.weighted_interaction_score.satisfied_protein_areas << ','
            << row.vinardo.gaussian << ',' << row.vinardo.repulsion << ','
            << row.vinardo.hydrophobic << ',' << row.vinardo.hydrogen_bond;
        for (double value : pose.transform.rotation) output << ',' << value;
        for (double value : pose.transform.translation) output << ',' << value;
        output << ',' << vinardo_adjusted_score(row.vinardo, 0.5) << ','
            << metrics[i].rmsd_heavy_A << ',' << metrics[i].center_distance_A << '\n';
    }
    output.precision(precision);
    check_output(output);
}

void write_analysis_csvs(std::ostream& rankings, std::ostream& topk, std::ostream& summary,
    const PoseRescoringResult& full, const std::vector<PoseMetrics>& metrics) {
    validate_metrics(full, metrics);
    std::unordered_map<std::size_t, const PoseMetrics*> by_id;
    for (const auto& value : metrics) by_id.emplace(value.pose_id, &value);
    rankings << "pool,method,rank,pose_id,score,rmsd_heavy_A,center_distance_A,"
        "rmsd_lt_2,rmsd_lt_3,center_lt_5\n";
    topk << "pool,method,status,k,effective_k,min_rmsd_heavy_A,rmsd_pose_id,rmsd_rank,"
        "min_center_distance_A,center_pose_id,center_rank,rmsd_lt_2,rmsd_lt_3,center_lt_5\n";
    summary << "pool,method,status,input_survivors,pool_size,best_rmsd_heavy_A,best_rmsd_pose_id,"
        "min_center_distance_A,min_center_pose_id,oracle_rmsd_lt_2,oracle_rmsd_lt_3,"
        "oracle_center_lt_5,top1_pose_id,top1_rmsd_heavy_A,top1_center_distance_A,"
        "top1_rmsd_lt_2,top1_rmsd_lt_3,top1_center_lt_5\n";
    rankings << std::setprecision(17);
    topk << std::setprecision(17);
    summary << std::setprecision(17);

    // Select by native Fast order, then restore survivor order before normalization.
    std::vector<PoseScoreEntry> fast;
    fast.reserve(full.poses.size());
    for (const auto& row : full.poses) fast.push_back({row.pose_id, row.pose.fast_weighted_score});
    auto offsets = top_pose_score_indices(fast, 500);
    std::sort(offsets.begin(), offsets.end());
    PoseRescoringResult selected;
    selected.input_pose_count = offsets.size();
    selected.weighted_interaction_calculated = selected.vinardo_calculated = true;
    selected.poses.reserve(offsets.size());
    for (auto offset : offsets) selected.poses.push_back(full.poses[offset]);

    for (const auto& [pool_name, pool] : std::array<std::pair<const char*, const PoseRescoringResult*>, 2>{{
             {"full", &full}, {"fast_top500", &selected}}}) {
        const PoseMetrics* oracle = nullptr;
        const PoseMetrics* center = nullptr;
        for (const auto& row : pool->poses) {
            const auto* value = by_id.at(row.pose_id);
            if (!oracle || value->rmsd_heavy_A < oracle->rmsd_heavy_A ||
                (value->rmsd_heavy_A == oracle->rmsd_heavy_A && value->pose_id < oracle->pose_id)) oracle = value;
            if (!center || value->center_distance_A < center->center_distance_A ||
                (value->center_distance_A == center->center_distance_A && value->pose_id < center->pose_id)) center = value;
        }
        const char* status = pool->poses.empty() ? "no_poses" : "ok";
        for (const auto& method : methods) {
            const auto ranked = rank_rescored_poses(*pool, method.options, 20);
            summary << pool_name << ',' << method.name << ',' << status << ','
                << full.input_pose_count << ',' << pool->poses.size();
            if (oracle) summary << ',' << oracle->rmsd_heavy_A << ',' << oracle->pose_id
                << ',' << center->center_distance_A << ',' << center->pose_id
                << ',' << (oracle->rmsd_heavy_A < 2.0) << ',' << (oracle->rmsd_heavy_A < 3.0)
                << ',' << (center->center_distance_A < 5.0);
            else summary << ",,,,,,,";
            const auto* first = ranked.empty() ? nullptr : by_id.at(ranked.front().scored.pose_id);
            if (first) summary << ',' << first->pose_id << ',' << first->rmsd_heavy_A << ',' << first->center_distance_A;
            else summary << ",,,";
            write_success(summary, first);
            summary << '\n';

            const PoseMetrics* rmsd_min = nullptr;
            const PoseMetrics* center_min = nullptr;
            std::size_t rmsd_rank = 0, center_rank = 0;
            for (std::size_t k = 1; k <= 20; ++k) {
                if (k <= ranked.size()) {
                    const auto& row = ranked[k - 1];
                    const auto* value = by_id.at(row.scored.pose_id);
                    rankings << pool_name << ',' << method.name << ',' << k << ',' << value->pose_id << ','
                        << score_value(row, method.options.method) << ',' << value->rmsd_heavy_A << ','
                        << value->center_distance_A;
                    write_success(rankings, value);
                    rankings << '\n';
                    // Equal metric minima retain the earliest rank in the prefix.
                    if (!rmsd_min || value->rmsd_heavy_A < rmsd_min->rmsd_heavy_A) {
                        rmsd_min = value; rmsd_rank = k;
                    }
                    if (!center_min || value->center_distance_A < center_min->center_distance_A) {
                        center_min = value; center_rank = k;
                    }
                }
                topk << pool_name << ',' << method.name << ',' << status << ',' << k << ',' << std::min(k, ranked.size());
                if (rmsd_min) topk << ',' << rmsd_min->rmsd_heavy_A << ',' << rmsd_min->pose_id << ',' << rmsd_rank
                    << ',' << center_min->center_distance_A << ',' << center_min->pose_id << ',' << center_rank
                    << ',' << (rmsd_min->rmsd_heavy_A < 2.0) << ',' << (rmsd_min->rmsd_heavy_A < 3.0)
                    << ',' << (center_min->center_distance_A < 5.0);
                else topk << ",,,,,,,,,";
                topk << '\n';
            }
        }
    }
    check_output(rankings);
    check_output(topk);
    check_output(summary);
}

} // namespace joint_diagnostics
