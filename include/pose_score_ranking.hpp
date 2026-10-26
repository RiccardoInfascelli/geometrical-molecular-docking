#ifndef POSE_SCORE_RANKING_HPP
#define POSE_SCORE_RANKING_HPP

#include "pose_rescoring.hpp"

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

enum class PoseScoreMethod { Fast, WeightedInteraction, Vinardo, Hybrid };

struct PoseScoreRankingOptions
{
    PoseScoreMethod method = PoseScoreMethod::Fast;
    double vinardo_alpha = 1.0;
    double hybrid_lambda = 0.3;
};

struct RankedPoseScore
{
    RescoredPose scored;
    std::optional<double> vinardo_adjusted;
    std::optional<double> hybrid;
};

struct PoseScoreEntry
{
    std::size_t pose_id{};
    double score{};
};

// Return original vector offsets, without copying pose records. Validate every
// score before selection. Ties use pose_id, then original offset, ascending.
std::vector<std::size_t> top_pose_score_indices(
    const std::vector<PoseScoreEntry>& entries,
    std::size_t top_k,
    bool minimize = false);

// Canonical names only: fast, weighted-interaction, vinardo, hybrid. Unknown values throw.
PoseScoreMethod parse_pose_score_method(std::string_view name);
const char* pose_score_method_name(PoseScoreMethod method);

// Reweight existing terms, without rescoring atom pairs:
// V_alpha = total + (alpha - 1) * repulsion, minimized. alpha is finite >= 0.
// Reject nonfinite components or a nonfinite adjusted result.
double vinardo_adjusted_score(const VinardoLikeScore& terms, double alpha);

// Fast/WeightedInteraction/Hybrid descending, Vinardo adjusted ascending; ties use pose_id.
// Normalize Hybrid on the complete supplied pool before taking top_k, with
// population standard deviation (N), zero deviation -> zero z-score, and
// H = lambda*z(WeightedInteraction) + (1-lambda)*z(-V_alpha). Lambda is in [0,1].
// Callers define this pool; scores normalized on separate pools are not comparable.
// Lambda endpoints retain the raw WeightedInteraction/Vinardo ordering. No RMSD, clustering,
// implicit prefilter, or changes to the scores occur. Complete ties in both
// score and pose_id retain input order.
// top_k must be positive. Validate every Fast score and every calculated method
// score, including rows outside top_k. A valid empty pool returns empty.
// Optional timing covers only Hybrid normalization/combination, not selection;
// it is a subinterval of the caller's complete ranking time.
std::vector<RankedPoseScore> rank_rescored_poses(
    const PoseRescoringResult& result,
    const PoseScoreRankingOptions& options,
    std::size_t top_k,
    double* hybrid_normalization_seconds = nullptr);

#endif // POSE_SCORE_RANKING_HPP
