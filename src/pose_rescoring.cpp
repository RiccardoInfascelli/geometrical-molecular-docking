#include "pose_rescoring.hpp"

#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "interaction_scoring.hpp"
#include "molecule_io.hpp"
#include "protein_area_crop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require_finite(double score, const char* method, std::size_t pose_id)
{
    if (!std::isfinite(score)) {
        throw std::runtime_error(
            std::string("Non-finite ") + method + " score for pose " +
            std::to_string(pose_id));
    }
}

} // namespace

RescoringTarget::RescoringTarget(const RunConfig& config, bool full_protein)
    : protein(load_protein(config.protein_path.string())),
      vinardo(*protein)
{
    const auto crop_margin = protein_area_crop_margin();
    const auto pocket = (crop_margin || !full_protein)
        ? load_pocket(config.pocket_path.string()) : nullptr;
    interaction_areas = pocket
        ? prepare_protein_areas(config.protein_path, *protein, *pocket)
        : get_protein_areas(*protein);
    if (!full_protein) {
        interaction_areas = filter_protein_areas_in_pocket(interaction_areas, *pocket);
    }
}

PoseRescoringResult rescore_poses(
    const std::vector<RankedPoseDiagnostic>& poses,
    const RDKit::ROMol& ligand,
    const PreparedLigand& prepared,
    const ProteinGrid& grid,
    const std::vector<interaction::interaction_area>& protein_areas,
    const VinardoLikeScorer& vinardo,
    std::optional<std::size_t> fast_top_n,
    bool reuse_fast_scores,
    PoseRescoringMethods methods)
{
    if (fast_top_n && *fast_top_n == 0) {
        throw std::invalid_argument("Fast top-N must be positive");
    }

    PoseRescoringResult result;
    result.input_pose_count = poses.size();
    result.weighted_interaction_calculated = methods.weighted_interaction;
    result.vinardo_calculated = methods.vinardo;
    result.poses.reserve(poses.size());
    for (std::size_t i = 0; i < poses.size(); ++i) {
        result.poses.push_back({i, poses[i], {}});
    }

    // Fast values are validated on the whole input, before any optional prefilter.
    if (!reuse_fast_scores) {
        const auto start = Clock::now();
        for (auto& row : result.poses) {
            const auto matching = transform_matching_areas(
                prepared.matching_areas, row.pose.transform);
            row.pose.fast_weighted_score =
                score_matching_areas_weighted(matching, grid).weighted_score;
            require_finite(row.pose.fast_weighted_score, "Fast", row.pose_id);
        }
        result.fast_scoring_seconds = seconds_since(start);
    } else {
        for (const auto& row : result.poses) {
            require_finite(row.pose.fast_weighted_score, "Fast", row.pose_id);
        }
    }

    // The optional prefilter selects by decreasing Fast score, then increasing
    // pose_id (input survivor order). WeightedInteraction and Vinardo-like do not affect membership.
    if (fast_top_n) {
        const auto start = Clock::now();
        if (*fast_top_n < result.poses.size()) {
            std::sort(result.poses.begin(), result.poses.end(),
                [](const RescoredPose& a, const RescoredPose& b) {
                    if (a.pose.fast_weighted_score != b.pose.fast_weighted_score) {
                        return a.pose.fast_weighted_score > b.pose.fast_weighted_score;
                    }
                    return a.pose_id < b.pose_id;
                });
            result.poses.resize(*fast_top_n);
            std::sort(result.poses.begin(), result.poses.end(),
                [](const RescoredPose& a, const RescoredPose& b) {
                    return a.pose_id < b.pose_id;
                });
        }
        result.selection_seconds = seconds_since(start);
    }

    if (methods.weighted_interaction) {
        const auto weighted_interaction_start = Clock::now();
        for (auto& row : result.poses) {
            const auto areas = transform_interaction_areas(
                prepared.areas, row.pose.transform);
            row.pose.weighted_interaction_score = score_weighted_interactions(protein_areas, areas);
            require_finite(row.pose.weighted_interaction_score.weighted_score, "WeightedInteraction", row.pose_id);
        }
        result.weighted_interaction_scoring_seconds = seconds_since(weighted_interaction_start);
    }

    if (methods.vinardo) {
        const auto vinardo_start = Clock::now();
        for (auto& row : result.poses) {
            row.vinardo = vinardo.score(ligand, row.pose.transform);
            require_finite(row.vinardo.total, "Vinardo-like", row.pose_id);
            require_finite(row.vinardo.gaussian, "Vinardo-like gaussian", row.pose_id);
            require_finite(row.vinardo.repulsion, "Vinardo-like repulsion", row.pose_id);
            require_finite(row.vinardo.hydrophobic, "Vinardo-like hydrophobic", row.pose_id);
            require_finite(row.vinardo.hydrogen_bond, "Vinardo-like hydrogen bond", row.pose_id);
        }
        result.vinardo_scoring_seconds = seconds_since(vinardo_start);
    }

    return result;
}

void write_pose_scores_csv(
    std::ostream& output,
    const PoseRescoringResult& result,
    std::size_t record_index)
{
    output << "record_index,pose_id,fast_score,weighted_interaction_score,vinardo_like_score,"
              "occupied_atoms,clash_fraction,weighted_interaction_total_interactions,"
              "weighted_interaction_satisfied_ligand_areas,weighted_interaction_satisfied_protein_areas,"
              "vinardo_gaussian,vinardo_repulsion,vinardo_hydrophobic,vinardo_hbond,"
              "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
    const auto previous_precision = output.precision();
    output << std::setprecision(17);
    for (const auto& row : result.poses) {
        const auto& pose = row.pose;
        output << record_index << ',' << row.pose_id << ','
               << pose.fast_weighted_score << ',' << pose.weighted_interaction_score.weighted_score << ','
               << row.vinardo.total << ',' << pose.occupied_atoms << ',' << pose.clash_fraction
               << ',' << pose.weighted_interaction_score.total_interactions
               << ',' << pose.weighted_interaction_score.satisfied_ligand_areas
               << ',' << pose.weighted_interaction_score.satisfied_protein_areas
               << ',' << row.vinardo.gaussian << ',' << row.vinardo.repulsion
               << ',' << row.vinardo.hydrophobic << ',' << row.vinardo.hydrogen_bond;
        for (const double value : pose.transform.rotation) {
            output << ',' << value;
        }
        for (const double value : pose.transform.translation) {
            output << ',' << value;
        }
        output << '\n';
    }
    output.precision(previous_precision);
    if (!output) {
        throw std::runtime_error("Cannot write joint pose scores CSV");
    }
}

void write_rescoring_protocol(
    std::ostream& output,
    const RunConfig& config,
    bool full_protein,
    std::optional<std::size_t> fast_top_n)
{
    const auto write = [&](const char* key, const std::string& value) {
        output << key << ",\"";
        for (const char character : value) {
            if (character == '"') output << '"';
            output << character;
        }
        output << "\"\n";
    };
    const auto number = [](double value) {
        std::ostringstream text;
        text << std::setprecision(17) << value;
        return text.str();
    };
    const auto environment = [](const char* name, const char* fallback) {
        const char* value = std::getenv(name);
        return std::string(value ? value : fallback);
    };
    const auto crop_margin = protein_area_crop_margin();

    output << "key,value\n";
    write("complex_id", config.complex_id);
    write("protein_path", config.protein_path.string());
    write("pocket_path", config.pocket_path.string());
    write("ligand_path", config.ligand_path.string());
    write("grid_voxel_size_A", number(config.grid_voxel_size));
    write("grid_margin_A", number(config.grid_margin));
    write("triplet_bin_size_A", number(config.triplet_bin_size));
    write("pocket_area_max_distance_A", number(config.pocket_centroid_distance));
    write("operational_clash_fraction", number(config.operational_clash_threshold));
    for (std::size_t i = 0; i < config.clash_thresholds.size(); ++i) {
        const std::string key = "diagnostic_clash_fraction_" + std::to_string(i);
        write(key.c_str(), number(config.clash_thresholds[i]));
    }
    write("pose_pool", fast_top_n ? "fast_top_n" : "all_clash_survivors");
    write("fast_top_n", fast_top_n ? std::to_string(*fast_top_n) : "none");
    write("fast_score_direction", "maximize");
    write("weighted_interaction_score_direction", "maximize");
    write("vinardo_like_score_direction", "minimize");
    write("prefilter_order", "Fast score descending, then original pose_id ascending");
    write("export_order", "original clash-survivor order; no WeightedInteraction or Vinardo-like ranking");
    write("pose_id", "zero-based ordinal in the original clash-survivor sequence");
    write("fast_region", crop_margin
        ? "cropped-protein chemical masks clipped to the pocket grid and margin"
        : "whole-protein chemical masks clipped to the pocket grid and margin");
    write("interaction_region", full_protein
        ? (crop_margin ? "cropped-protein physical interaction areas" : "whole-protein physical interaction areas") :
        "physical areas whose every atom matches a pocket atom within 0.001 A");
    write("vinardo_like_region", "whole-protein supported heavy atoms; strict atom-pair distance < 8 A");
    write("chemical_area_crop", crop_margin
        ? "whole-residue PDB crop around pocket bounds before area detection; full protein for occupancy and clash"
        : "none; full protein used before pocket feature selection");
    write("chemical_area_crop_margin_A", crop_margin ? number(*crop_margin) : "off");
    write("DOCKING_AREA_CROP_MARGIN_A", environment("DOCKING_AREA_CROP_MARGIN_A", "unset"));
    write("vinardo_like_crop", "none");
    write("vinardo_like_units", "experimental score; not validated Vinardo or kcal/mol");
    write("score_weights", "unchanged default Fast, WeightedInteraction and Vinardo-like coefficients");
    write("nonfinite_scores", "fail the experiment; do not silently remove poses");
    write("rmsd", "not exported or used for selection or ranking");
    write("DOCKING_KABSCH_METHOD", environment("DOCKING_KABSCH_METHOD", "svd"));
    const std::string clash = environment("DOCKING_CPU_CLASH_METHOD", "original");
    write("DOCKING_CPU_CLASH_METHOD", clash);
    write("effective_cpu_clash_method", clash == "cached" ? "cached" : "original");
    write("DOCKING_QUERY_METHOD", environment("DOCKING_QUERY_METHOD", "reference"));
    write("DOCKING_CPU_SCREENING_ONLY", environment("DOCKING_CPU_SCREENING_ONLY", "unset"));
    write("DOCKING_STAGE_TIMINGS", environment("DOCKING_STAGE_TIMINGS", "unset"));
    write("DOCKING_KABSCH_TIMINGS", environment("DOCKING_KABSCH_TIMINGS", "unset"));
    if (!output) {
        throw std::runtime_error("Cannot write joint rescoring protocol CSV");
    }
}
