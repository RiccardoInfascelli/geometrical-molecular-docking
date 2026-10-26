#ifndef SCORING_ABLATION_HPP
#define SCORING_ABLATION_HPP

#include "pose_score_ranking.hpp"

#include <array>
#include <iosfwd>
#include <optional>
#include <vector>

namespace scoring_ablation {

struct WeightedInteractionFamily {
    const char* name;
    double WeightedInteractionWeights::* weight;
    std::size_t WeightedInteractionScore::* count;
};

const std::array<WeightedInteractionFamily, 7>& weighted_interaction_families();
const std::array<const char*, 4>& vinardo_terms();

// A zero reward removes a family from the per-area maximum, not from geometry.
WeightedInteractionWeights weighted_interaction_weights_without(std::size_t family);
VinardoLikeScore vinardo_without(const VinardoLikeScore& score, std::size_t term);

struct InputPose {
    RescoredPose scored;
    std::optional<double> saved_weighted_interaction;
    std::optional<double> saved_vinardo;
};

// Accept joint-rescoring and gpu_docking CSVs. Never select or reorder poses.
// IDs must be unique within the selected record; malformed/nonfinite data fail.
// Plain GPU Fast exports use alignment_index when pose_id is not available.
std::vector<InputPose> read_pose_pool(std::istream& input, std::size_t record);

struct PoseAblation {
    RescoredPose baseline;
    std::array<WeightedInteractionScore, 7> weighted_interaction_without;
    std::optional<double> rmsd_heavy;
};

enum class RemovedScore { None, WeightedInteraction, Vinardo };

// Build a counterfactual on exactly the same IDs, transforms and input order.
// The existing ranker recomputes Hybrid normalization on this complete pool.
PoseRescoringResult make_pool(const std::vector<PoseAblation>& poses,
                             RemovedScore removed = RemovedScore::None,
                             std::size_t term = 0);

} // namespace scoring_ablation

#endif
