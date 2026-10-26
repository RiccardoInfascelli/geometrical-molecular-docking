#include "pose_score_ranking.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require_finite(double value, const char* field)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string("Nonfinite pose score: ") + field);
}

void validate_alpha(double alpha)
{
    if (!std::isfinite(alpha) || alpha < 0.0)
        throw std::invalid_argument("Vinardo alpha must be finite and non-negative");
}

// Scale before Welford accumulation to limit overflow/underflow in squared
// differences. CPU statistics use long double; CUDA reductions use double.
// No epsilon is applied, and zero variance produces zero z-scores.
std::vector<double> population_zscores(const std::vector<double>& values)
{
    std::vector<double> z(values.size(), 0.0);
    if (values.empty()) return z;
    long double scale = 0.0L;
    for (double value : values)
        scale = std::max(scale, std::abs(static_cast<long double>(value)));
    if (scale == 0.0L) return z;

    long double mean = 0.0L, sum_squares = 0.0L;
    std::size_t count = 0;
    for (double value : values) {
        const long double scaled = static_cast<long double>(value) / scale;
        const long double delta = scaled - mean;
        mean += delta / static_cast<long double>(++count);
        sum_squares += delta * (scaled - mean);
    }
    const long double variance = sum_squares / static_cast<long double>(count);
    if (!std::isfinite(mean) || !std::isfinite(variance) || variance < 0.0L)
        throw std::invalid_argument("Invalid population score variance");
    if (variance == 0.0L) return z;
    const long double deviation = std::sqrt(variance);
    if (!std::isfinite(deviation))
        throw std::invalid_argument("Nonfinite population score deviation");
    for (std::size_t i = 0; i < values.size(); ++i) {
        z[i] = static_cast<double>((static_cast<long double>(values[i]) / scale - mean) / deviation);
        require_finite(z[i], "population z-score");
    }
    return z;
}

} // namespace

std::vector<std::size_t> top_pose_score_indices(
    const std::vector<PoseScoreEntry>& entries,
    std::size_t top_k,
    bool minimize)
{
    if (top_k == 0) throw std::invalid_argument("Top-k must be positive");
    for (const auto& entry : entries) require_finite(entry.score, "ranking score");
    std::vector<std::size_t> indices(entries.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    const auto before = [&](std::size_t a, std::size_t b) {
        if (entries[a].score != entries[b].score)
            return minimize ? entries[a].score < entries[b].score : entries[a].score > entries[b].score;
        if (entries[a].pose_id != entries[b].pose_id)
            return entries[a].pose_id < entries[b].pose_id;
        return a < b;
    };
    const auto count = std::min(top_k, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), before);
    indices.resize(count);
    return indices;
}

PoseScoreMethod parse_pose_score_method(std::string_view name)
{
    if (name == "fast") return PoseScoreMethod::Fast;
    if (name == "weighted-interaction") return PoseScoreMethod::WeightedInteraction;
    if (name == "vinardo") return PoseScoreMethod::Vinardo;
    if (name == "hybrid") return PoseScoreMethod::Hybrid;
    throw std::invalid_argument("Score method must be fast, weighted-interaction, vinardo or hybrid");
}

const char* pose_score_method_name(PoseScoreMethod method)
{
    switch (method) {
    case PoseScoreMethod::Fast: return "fast";
    case PoseScoreMethod::WeightedInteraction: return "weighted-interaction";
    case PoseScoreMethod::Vinardo: return "vinardo";
    case PoseScoreMethod::Hybrid: return "hybrid";
    }
    throw std::invalid_argument("Invalid pose score method");
}

double vinardo_adjusted_score(const VinardoLikeScore& terms, double alpha)
{
    validate_alpha(alpha);
    require_finite(terms.total, "Vinardo total");
    require_finite(terms.gaussian, "Vinardo gaussian");
    require_finite(terms.repulsion, "Vinardo repulsion");
    require_finite(terms.hydrophobic, "Vinardo hydrophobic");
    require_finite(terms.hydrogen_bond, "Vinardo hydrogen bond");
    const double adjusted = terms.total + (alpha - 1.0) * terms.repulsion;
    require_finite(adjusted, "Vinardo adjusted");
    return adjusted;
}

std::vector<RankedPoseScore> rank_rescored_poses(
    const PoseRescoringResult& input,
    const PoseScoreRankingOptions& options,
    std::size_t top_k,
    double* hybrid_normalization_seconds)
{
    if (hybrid_normalization_seconds) *hybrid_normalization_seconds = 0.0;
    if (top_k == 0) throw std::invalid_argument("Top-k must be positive");
    // Validate the enum even for a pool with no rows.
    pose_score_method_name(options.method);
    validate_alpha(options.vinardo_alpha);
    if (!std::isfinite(options.hybrid_lambda) || options.hybrid_lambda < 0.0 ||
        options.hybrid_lambda > 1.0)
        throw std::invalid_argument("Hybrid lambda must be finite and in [0, 1]");
    const bool hybrid = options.method == PoseScoreMethod::Hybrid;
    if ((options.method == PoseScoreMethod::WeightedInteraction || hybrid) && !input.weighted_interaction_calculated)
        throw std::invalid_argument("WeightedInteraction scores were not calculated");
    if ((options.method == PoseScoreMethod::Vinardo || hybrid) && !input.vinardo_calculated)
        throw std::invalid_argument("Vinardo scores were not calculated");

    std::vector<double> adjusted_values, hybrid_values, weighted_interaction_values, vinardo_rewards;
    if (input.vinardo_calculated) adjusted_values.reserve(input.poses.size());
    if (hybrid) {
        weighted_interaction_values.reserve(input.poses.size());
        vinardo_rewards.reserve(input.poses.size());
    }
    for (const auto& row : input.poses) {
        require_finite(row.pose.fast_weighted_score, "Fast");
        if (input.weighted_interaction_calculated)
            require_finite(row.pose.weighted_interaction_score.weighted_score, "WeightedInteraction");
        if (input.vinardo_calculated)
            adjusted_values.push_back(vinardo_adjusted_score(row.vinardo, options.vinardo_alpha));
        if (hybrid) {
            weighted_interaction_values.push_back(row.pose.weighted_interaction_score.weighted_score);
            vinardo_rewards.push_back(-adjusted_values.back());
        }
    }
    if (hybrid) {
        const auto normalization_start = hybrid_normalization_seconds
            ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto weighted_interaction_z = population_zscores(weighted_interaction_values);
        const auto vinardo_z = population_zscores(vinardo_rewards);
        hybrid_values.reserve(input.poses.size());
        for (std::size_t i = 0; i < input.poses.size(); ++i) {
            const double score = options.hybrid_lambda * weighted_interaction_z[i] +
                (1.0 - options.hybrid_lambda) * vinardo_z[i];
            require_finite(score, "Hybrid");
            hybrid_values.push_back(score);
        }
        if (hybrid_normalization_seconds)
            *hybrid_normalization_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - normalization_start).count();
    }

    std::vector<PoseScoreEntry> entries;
    entries.reserve(input.poses.size());
    bool minimize = false;
    for (std::size_t i = 0; i < input.poses.size(); ++i) {
        const auto& row = input.poses[i];
        double score = 0.0;
        switch (options.method) {
        case PoseScoreMethod::Fast:
            score = row.pose.fast_weighted_score;
            break;
        case PoseScoreMethod::WeightedInteraction:
            score = row.pose.weighted_interaction_score.weighted_score;
            break;
        case PoseScoreMethod::Vinardo:
            score = adjusted_values[i];
            minimize = true;
            break;
        case PoseScoreMethod::Hybrid:
            // At the endpoints, rank by the raw component while still exporting Hybrid.
            if (options.hybrid_lambda == 0.0) {
                score = adjusted_values[i];
                minimize = true;
            } else if (options.hybrid_lambda == 1.0) {
                score = row.pose.weighted_interaction_score.weighted_score;
            } else {
                score = hybrid_values[i];
            }
            break;
        }
        entries.push_back({row.pose_id, score});
    }
    const auto indices = top_pose_score_indices(entries, top_k, minimize);
    std::vector<RankedPoseScore> ranked;
    ranked.reserve(indices.size());
    for (const auto index : indices) {
        RankedPoseScore output{input.poses[index], std::nullopt, std::nullopt};
        if (input.vinardo_calculated) output.vinardo_adjusted = adjusted_values[index];
        if (hybrid) output.hybrid = hybrid_values[index];
        ranked.push_back(std::move(output));
    }
    return ranked;
}
