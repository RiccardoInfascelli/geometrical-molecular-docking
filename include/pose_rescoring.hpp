#ifndef POSE_RESCORING_HPP
#define POSE_RESCORING_HPP

#include "ligand_preparation.hpp"
#include "pose_generation.hpp"
#include "run_config.hpp"
#include "vinardo_scoring.hpp"

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <optional>
#include <vector>

// WeightedInteraction uses pocket features by default; Vinardo-like always uses the full protein.
// full_protein changes only the WeightedInteraction region, not the Vinardo receptor or clash.
// The owner keeps the loaded receptor available throughout the experiment.
struct RescoringTarget
{
    std::unique_ptr<RDKit::RWMol> protein;
    std::vector<interaction::interaction_area> interaction_areas;
    VinardoLikeScorer vinardo;

    explicit RescoringTarget(const RunConfig& config, bool full_protein = false);
};

struct RescoredPose
{
    // Zero-based position in the original clash-survivor sequence, before selection.
    std::size_t pose_id{};
    RankedPoseDiagnostic pose;
    VinardoLikeScore vinardo;
};

struct PoseRescoringMethods
{
    bool weighted_interaction = true;
    bool vinardo = true;
};

struct PoseRescoringResult
{
    std::size_t input_pose_count{};
    std::vector<RescoredPose> poses;
    double fast_scoring_seconds{};
    double selection_seconds{};
    double weighted_interaction_scoring_seconds{};
    double vinardo_scoring_seconds{};
    // Requested methods are marked calculated even for an empty pool.
    bool weighted_interaction_calculated = false;
    bool vinardo_calculated = false;
};

// Score the same transforms with Fast, WeightedInteraction and Vinardo-like, without RMSD use.
// No prefilter means all input poses; a positive top-N selects by decreasing Fast
// score, then increasing pose_id. Output rows retain their original input order.
// reuse_fast_scores preserves Fast values already computed by the GPU pipeline.
// This experiment rejects non-finite scores instead of silently dropping poses.
// Only requested rescoring loops run; omitted score fields must not be ranked.
PoseRescoringResult rescore_poses(
    const std::vector<RankedPoseDiagnostic>& poses,
    const RDKit::ROMol& ligand,
    const PreparedLigand& prepared,
    const ProteinGrid& grid,
    const std::vector<interaction::interaction_area>& protein_areas,
    const VinardoLikeScorer& vinardo,
    std::optional<std::size_t> fast_top_n = std::nullopt,
    bool reuse_fast_scores = false,
    PoseRescoringMethods methods = {});

// Write a complete CSV, including the header. Scores and transforms share one ID;
// no RMSD or ranking is exported. Fast/WeightedInteraction are maximized; Vinardo-like minimized.
// All score columns write stored values, even if a method was not requested.
void write_pose_scores_csv(
    std::ostream& output,
    const PoseRescoringResult& result,
    std::size_t record_index);

// Write shared scientific settings as key,value CSV. The driver appends backend,
// index policy, counts and timings for its particular experiment.
void write_rescoring_protocol(
    std::ostream& output,
    const RunConfig& config,
    bool full_protein,
    std::optional<std::size_t> fast_top_n);

#endif // POSE_RESCORING_HPP
