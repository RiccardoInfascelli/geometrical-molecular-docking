#ifndef JOINT_DIAGNOSTICS_HPP
#define JOINT_DIAGNOSTICS_HPP

#include "pose_rescoring.hpp"

#include <iosfwd>
#include <vector>

namespace joint_diagnostics {

struct PoseMetrics {
    std::size_t pose_id{};
    double rmsd_heavy_A{};
    double center_distance_A{};
};

// Evaluate every survivor in the receptor frame, without fitting or symmetry remapping.
std::vector<PoseMetrics> evaluate_pose_metrics(const PoseRescoringResult& pool,
    const RDKit::ROMol& ligand, const RDKit::ROMol& reference);

void write_pose_csv(std::ostream& output, const PoseRescoringResult& pool,
    const std::vector<PoseMetrics>& metrics);

// Full and Fast Top-500 use the same rows. Hybrid is normalized on each supplied
// pool before selecting Top-20; evaluation metrics never participate in ranking.
void write_analysis_csvs(std::ostream& rankings, std::ostream& topk, std::ostream& summary,
    const PoseRescoringResult& pool, const std::vector<PoseMetrics>& metrics);

} // namespace joint_diagnostics

#endif
