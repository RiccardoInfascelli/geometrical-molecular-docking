#include "ensemble_diagnostics.hpp"

#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "pose_score_ranking.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <set>
#include <stdexcept>
#include <tuple>

namespace conformer_diagnostics {
namespace {

using Clock = std::chrono::steady_clock;

struct PoseView {
    const EnsemblePose* pose{};
    std::size_t accepted_order{};
    std::size_t global_id{};
    double center_distance{};
};

struct Method {
    const char* name;
    PoseScoreRankingOptions options;
};

constexpr std::array<std::size_t, 4> budgets{1, 5, 10, 20};
const std::array<Method, 5> methods{{
    {"fast", {PoseScoreMethod::Fast, 1.0, 0.3}},
    {"weighted-interaction", {PoseScoreMethod::WeightedInteraction, 1.0, 0.3}},
    {"vinardo", {PoseScoreMethod::Vinardo, 1.0, 0.3}},
    {"vinardo-rep50", {PoseScoreMethod::Vinardo, 0.5, 0.3}},
    {"hybrid", {PoseScoreMethod::Hybrid, 0.5, 0.3}}
}};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void quoted(std::ostream& output, const std::string& value) {
    output << '"';
    for (char c : value) {
        if (c == '"') output << '"';
        output << c;
    }
    output << '"';
}

std::vector<PoseView> make_views(const std::vector<Input>& inputs,
    const RDKit::ROMol& reference) {
    if (inputs.empty() || inputs.front().conformer_id != 0)
        throw std::invalid_argument("Conformer diagnostics require the original record with ID 0 first");
    validate_gpu_docking_reference(reference, reference);
    const auto reference_center = heavy_atom_centroid(reference);
    std::set<std::size_t> conformer_ids;
    std::vector<PoseView> views;
    for (std::size_t source = 0; source < inputs.size(); ++source) {
        const auto& input = inputs[source];
        if (!input.molecule || !input.docking || !conformer_ids.insert(input.conformer_id).second)
            throw std::invalid_argument("Conformer diagnostics require owned molecules and unique conformer IDs");
        validate_gpu_docking_reference(*input.molecule, reference);
        const auto center = heavy_atom_centroid(*input.molecule);
        const auto& docking = *input.docking;
        if (docking.survivors != docking.poses.size() || docking.rescored_ids.size() != docking.poses.size())
            throw std::invalid_argument("Conformer diagnostics require every clash survivor and its raw scores");
        std::set<std::size_t> pose_ids, rescored_ids;
        for (auto id : docking.rescored_ids)
            if (!rescored_ids.insert(id).second)
                throw std::invalid_argument("Duplicate rescored pose ID in conformer diagnostics");
        for (const auto& pose : docking.poses) {
            if (pose.conformer_id != input.conformer_id || pose.scored.pose_id != pose.pose_id ||
                !pose_ids.insert(pose.pose_id).second)
                throw std::invalid_argument("Conformer diagnostic pose identity is inconsistent");
            if (!std::isfinite(pose.rmsd) || pose.rmsd < 0.0)
                throw std::invalid_argument("Conformer diagnostics require finite nonnegative heavy-atom RMSD");
            if (!std::isfinite(pose.scored.pose.fast_weighted_score) ||
                !std::isfinite(pose.scored.pose.weighted_interaction_score.weighted_score))
                throw std::invalid_argument("Nonfinite conformer diagnostic score");
            vinardo_adjusted_score(pose.scored.vinardo, 1.0);
            views.push_back({&pose, source, 0,
                pose_center_distance(center, pose.scored.pose.transform, reference_center)});
        }
        if (pose_ids != rescored_ids)
            throw std::invalid_argument("Conformer diagnostic scores do not cover the survivor IDs");
    }
    std::sort(views.begin(), views.end(), [](const auto& a, const auto& b) {
        return std::tie(a.pose->conformer_id, a.pose->pose_id) <
            std::tie(b.pose->conformer_id, b.pose->pose_id);
    });
    for (std::size_t i = 0; i < views.size(); ++i) views[i].global_id = i;
    return views;
}

double score_value(const RankedPoseScore& row, PoseScoreMethod method) {
    switch (method) {
    case PoseScoreMethod::Fast: return row.scored.pose.fast_weighted_score;
    case PoseScoreMethod::WeightedInteraction: return row.scored.pose.weighted_interaction_score.weighted_score;
    case PoseScoreMethod::Vinardo: return row.vinardo_adjusted.value();
    case PoseScoreMethod::Hybrid: return row.hybrid.value();
    }
    throw std::invalid_argument("Unknown conformer diagnostic score method");
}

void write_id(std::ostream& output, const PoseView* view) {
    if (!view) output << ",,,";
    else output << ',' << view->global_id << ',' << view->pose->conformer_id << ',' << view->pose->pose_id;
}

void write_success(std::ostream& output, const PoseView* rmsd, const PoseView* center) {
    if (!rmsd || !center) output << ",,,";
    else output << ',' << (rmsd->pose->rmsd < 2.0) << ',' << (rmsd->pose->rmsd < 3.0)
        << ',' << (center->center_distance < 5.0);
}

} // namespace

void write_diagnostics(std::ostream& summary, std::ostream& topk,
    const std::string& complex_id, const std::vector<Input>& inputs,
    const RDKit::ROMol& reference, Timings* timings) {
    if (timings) *timings = {};
    const auto metrics_start = Clock::now();
    const auto views = make_views(inputs, reference);
    if (timings) timings->metrics_s = elapsed(metrics_start);
    const auto analysis_start = Clock::now();
    const char* common = "complex_id,budget_requested,budget_available,budget_complete,original_included,"
        "pool,method,status,used_conformers,conformers_with_survivors,input_survivors,pool_size";
    summary << common << ",best_rmsd_heavy_A,best_rmsd_global_pose_id,best_rmsd_conformer_id,best_rmsd_pose_id,"
        "min_center_distance_A,min_center_global_pose_id,min_center_conformer_id,min_center_pose_id,"
        "oracle_rmsd_lt_2,oracle_rmsd_lt_3,oracle_center_lt_5,top1_global_pose_id,top1_conformer_id,"
        "top1_pose_id,top1_alignment_index,top1_score,top1_rmsd_heavy_A,top1_center_distance_A,"
        "top1_rmsd_lt_2,top1_rmsd_lt_3,top1_center_lt_5,top1_score_tie_count,original_in_top1_score_tie,"
        "top1_rmsd_minus_oracle_A,top1_center_minus_min_A\n";
    topk << common << ",k,effective_k,min_rmsd_heavy_A,rmsd_global_pose_id,rmsd_conformer_id,rmsd_pose_id,rmsd_rank,"
        "min_center_distance_A,center_global_pose_id,center_conformer_id,center_pose_id,center_rank,"
        "rmsd_lt_2,rmsd_lt_3,center_lt_5\n";
    summary << std::setprecision(17);
    topk << std::setprecision(17);

    for (auto budget : budgets) {
        const auto available = std::min(budget, inputs.size() - 1);
        for (bool original : {false, true}) {
            std::vector<std::size_t> full_ids;
            for (const auto& view : views)
                if (view.accepted_order ? view.accepted_order <= available : original)
                    full_ids.push_back(view.global_id);
            const auto with_survivors = std::count_if(inputs.begin(), inputs.begin() + available + 1,
                [&](const auto& input) {
                    return (input.conformer_id || original) && input.docking->survivors;
                });
            for (bool fast_subset : {false, true}) {
                auto selected_ids = full_ids;
                if (fast_subset) {
                    std::vector<PoseScoreEntry> fast;
                    fast.reserve(full_ids.size());
                    for (auto id : full_ids) fast.push_back({id, views[id].pose->scored.pose.fast_weighted_score});
                    const auto offsets = top_pose_score_indices(fast, 500);
                    selected_ids.clear();
                    for (auto offset : offsets) selected_ids.push_back(full_ids[offset]);
                    std::sort(selected_ids.begin(), selected_ids.end());
                }
                PoseRescoringResult pool;
                pool.input_pose_count = selected_ids.size();
                pool.weighted_interaction_calculated = pool.vinardo_calculated = true;
                pool.poses.reserve(selected_ids.size());
                const PoseView* oracle = nullptr;
                const PoseView* center = nullptr;
                for (auto id : selected_ids) {
                    const auto& view = views[id];
                    pool.poses.push_back(view.pose->scored);
                    pool.poses.back().pose_id = id;
                    if (!oracle || view.pose->rmsd < oracle->pose->rmsd) oracle = &view;
                    if (!center || view.center_distance < center->center_distance) center = &view;
                }
                const auto prefix = [&](std::ostream& output, const char* method) {
                    quoted(output, complex_id);
                    output << ',' << budget << ',' << available << ',' << (available == budget) << ',' << original
                        << ',' << (fast_subset ? "fast_top500" : "full") << ',' << method << ','
                        << (pool.poses.empty() ? "no_poses" : "ok") << ',' << available + original << ','
                        << with_survivors << ',' << full_ids.size() << ',' << pool.poses.size();
                };
                for (const auto& method : methods) {
                    // The complete native ranking supplies exact score-tie counts;
                    // only the current method/pool is materialized at a time.
                    const auto ranked = rank_rescored_poses(pool, method.options,
                        std::max(std::size_t{1}, pool.poses.size()));
                    const auto* first = ranked.empty() ? nullptr : &views.at(ranked.front().scored.pose_id);
                    const double first_score = ranked.empty() ? 0.0 : score_value(ranked.front(), method.options.method);
                    std::size_t tie_count = 0;
                    bool original_tied = false;
                    for (const auto& row : ranked) {
                        if (score_value(row, method.options.method) != first_score) break;
                        ++tie_count;
                        original_tied = original_tied || views.at(row.scored.pose_id).pose->conformer_id == 0;
                    }
                    prefix(summary, method.name);
                    if (oracle) summary << ',' << oracle->pose->rmsd;
                    else summary << ',';
                    write_id(summary, oracle);
                    if (center) summary << ',' << center->center_distance;
                    else summary << ',';
                    write_id(summary, center);
                    write_success(summary, oracle, center);
                    write_id(summary, first);
                    if (first) summary << ',' << first->pose->alignment_index << ',' << first_score
                        << ',' << first->pose->rmsd << ',' << first->center_distance;
                    else summary << ",,,,";
                    write_success(summary, first, first);
                    if (first) summary << ',' << tie_count << ',' << original_tied << ','
                        << first->pose->rmsd - oracle->pose->rmsd << ',' << first->center_distance - center->center_distance;
                    else summary << ",,,,";
                    summary << '\n';

                    const PoseView* rmsd_min = nullptr;
                    const PoseView* center_min = nullptr;
                    std::size_t rmsd_rank = 0, center_rank = 0;
                    for (std::size_t k = 1; k <= 20; ++k) {
                        if (k <= ranked.size()) {
                            const auto& view = views.at(ranked[k - 1].scored.pose_id);
                            if (!rmsd_min || view.pose->rmsd < rmsd_min->pose->rmsd) {
                                rmsd_min = &view; rmsd_rank = k;
                            }
                            if (!center_min || view.center_distance < center_min->center_distance) {
                                center_min = &view; center_rank = k;
                            }
                        }
                        prefix(topk, method.name);
                        topk << ',' << k << ',' << std::min(k, ranked.size());
                        if (rmsd_min) topk << ',' << rmsd_min->pose->rmsd;
                        else topk << ',';
                        write_id(topk, rmsd_min);
                        if (rmsd_min) topk << ',' << rmsd_rank;
                        else topk << ',';
                        if (center_min) topk << ',' << center_min->center_distance;
                        else topk << ',';
                        write_id(topk, center_min);
                        if (center_min) topk << ',' << center_rank;
                        else topk << ',';
                        write_success(topk, rmsd_min, center_min);
                        topk << '\n';
                    }
                }
            }
        }
    }
    if (!summary || !topk) throw std::runtime_error("Could not write conformer diagnostics CSV");
    if (timings) timings->analysis_s = elapsed(analysis_start);
}

} // namespace conformer_diagnostics
