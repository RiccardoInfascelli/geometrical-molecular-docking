// Configurable CUDA search with per-record ranking and pose export. The target
// persists across records; the rescoring backend does not change the search backend.
#include "gpu_docking_config.hpp"
#include "molecule_io.hpp"
#include "protein_area_crop.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (const char c : value) {
        if (c == '"') result += '"';
        result += c;
    }
    return result + '"';
}

void write_summary(std::ostream& output, const GpuDockingCliOptions& options,
                   const GpuDockingScreen& screen, std::size_t processed,
                   double wall_before_summary) {
    const auto& config = options.config;
    const auto& settings = options.settings;
    const auto method = settings.ranking.method;
    const bool weighted_interaction = method == PoseScoreMethod::WeightedInteraction || method == PoseScoreMethod::Hybrid;
    const bool vinardo = method == PoseScoreMethod::Vinardo || method == PoseScoreMethod::Hybrid;
    const auto crop_margin = protein_area_crop_margin();
    output << std::setprecision(17) << "key,value\n";
    const auto write = [&](const std::string& key, const auto& value) {
        output << key << ',' << value << '\n';
    };
    const auto environment = [](const char* key) {
        const char* value = std::getenv(key);
        return csv_string(value ? value : "unset");
    };
    write("status", "complete");
    write("processed", processed);
    write("top_k", options.top_k);
    write("limit", options.limit == std::numeric_limits<std::size_t>::max()
        ? "all" : std::to_string(options.limit));
    write("protein", csv_string(config.protein_path.string()));
    write("pocket", csv_string(config.pocket_path.string()));
    write("ligands", csv_string(options.input.string()));
    write("reference_mol2", csv_string(options.references.string()));
    write("ranking_requested", options.ranking_requested);
    write("ranking_method", pose_score_method_name(method));
    write("rescoring_backend", method == PoseScoreMethod::Fast ? "not_calculated" :
        settings.rescoring_backend == GpuRescoringBackend::Cuda ? "cuda" : "cpu");
    write("survivor_download_policy", settings.rescoring_backend == GpuRescoringBackend::Cuda &&
        method != PoseScoreMethod::Fast ? "top-k only after device rescoring and ranking" : "all survivors after Fast");
    write("ranking_direction", method == PoseScoreMethod::Vinardo ? "ascending" : "descending");
    if (method == PoseScoreMethod::Fast)
        write("ranking", "Fast descending; ties alignment_index ascending; no clustering");
    else
        write("ranking", std::string(pose_score_method_name(method)) +
            (method == PoseScoreMethod::Vinardo ? " ascending" : " descending") +
            "; ties original pose_id ascending; no clustering");
    write("tie_break", "original alignment/survivor order ascending");
    write("pose_id", "zero-based original clash-survivor ordinal; alignment_index retained separately");
    write("pose_pool", settings.fast_prefilter ? "explicit_fast_prefilter" : "all_clash_survivors");
    write("fast_prefilter", settings.fast_prefilter ? std::to_string(*settings.fast_prefilter) : "none");
    write("index_policy", "shared_index_only");
    write("index_cutoff_A", settings.cutoff);
    write("index_records", screen.index_records());
    write("grid_voxel_size_A", config.grid_voxel_size);
    write("grid_margin_A", config.grid_margin);
    write("triplet_bin_size_A", config.triplet_bin_size);
    write("pocket_area_max_distance_A", config.pocket_centroid_distance);
    write("operational_clash_fraction", config.operational_clash_threshold);
    write("target_clearance_enabled", settings.min_target_clearance.has_value());
    if (settings.min_target_clearance) write("min_target_clearance_A", *settings.min_target_clearance);
    else write("min_target_clearance_A", "off");
    write("targets_before_pruning", screen.targets_before_pruning());
    write("targets_after_pruning", screen.targets_after_pruning());
    write("kabsch_method_effective", options.kabsch_method);
    write("kabsch_source", options.kabsch_source);
    write("query_method_effective", options.query_method);
    write("query_source", options.query_source);
    write("DOCKING_KABSCH_METHOD", environment("DOCKING_KABSCH_METHOD"));
    write("DOCKING_QUERY_METHOD", environment("DOCKING_QUERY_METHOD"));
    write("diagnostics_effective", settings.diagnostic_stages ? "stages" : "off");
    write("diagnostics_source", options.diagnostics_source);
    write("DOCKING_PROFILE_STAGES", environment("DOCKING_PROFILE_STAGES"));
    write("DOCKING_STAGE_TIMINGS", environment("DOCKING_STAGE_TIMINGS"));
    write("DOCKING_KABSCH_TIMINGS", environment("DOCKING_KABSCH_TIMINGS"));
    write("timings_enabled", settings.collect_timings);
    write("weighted_interaction_calculated", weighted_interaction);
    write("vinardo_calculated", vinardo);
    write("hybrid_calculated", method == PoseScoreMethod::Hybrid);
    write("interaction_region", weighted_interaction ? (settings.interaction_full_protein ? "protein" : "pocket") : "not_calculated");
    write("fast_region", crop_margin
        ? "cropped-protein chemical masks clipped to pocket grid and margin"
        : "whole-protein chemical masks clipped to pocket grid and margin");
    write("vinardo_region", vinardo ? "whole-protein supported heavy atoms; pair distance < 8 A" : "not_calculated");
    write("protein_crop", csv_string(crop_margin
        ? "whole-residue PDB crop around pocket bounds for chemical areas only; full protein for occupancy, clash and Vinardo"
        : "none; full protein used for chemical areas, occupancy, clash and Vinardo"));
    if (crop_margin) write("chemical_area_crop_margin_A", *crop_margin);
    else write("chemical_area_crop_margin_A", "off");
    write("DOCKING_AREA_CROP_MARGIN_A", environment("DOCKING_AREA_CROP_MARGIN_A"));
    write("clustering", "none; unavailable in shared API");
    if (vinardo) {
        write("vinardo_alpha", settings.ranking.vinardo_alpha);
        write("vinardo_adjustment", "V_alpha=V+(alpha-1)*repulsion");
        write("vinardo_skipped_receptor_atoms", screen.vinardo_skipped_receptor_atoms());
    } else write("vinardo_alpha", "not_calculated");
    if (method == PoseScoreMethod::Hybrid) {
        write("hybrid_lambda", settings.ranking.hybrid_lambda);
        write("hybrid_formula", "lambda*z(WeightedInteraction)+(1-lambda)*z(-V_alpha)");
        write("hybrid_normalization", "population std N; zero std gives zero z; no epsilon");
        write("hybrid_precision", settings.rescoring_backend == GpuRescoringBackend::Cuda
            ? "parallel scaled Welford in double; scalar reductions cross host; per-pose z and Hybrid on device"
            : "serial scaled Welford in host long double");
        write("hybrid_normalization_pool", "same ligand surviving explicit prefilter; before final top-k");
    } else write("hybrid_lambda", "not_calculated");
    write("nonfinite_scores", "error; no silent removal");
    write("vinardo_units", "experimental Vinardo-like; not validated affinity or kcal/mol");
    write("rmsd", options.references.empty() ? "disabled" :
          "direct heavy atoms; fixed atom order; no alignment or symmetry; never ranking");
    const FastScoreWeights fast{};
    write("fast_weight_hydrophobic", fast.hydrophobic);
    write("fast_weight_hydrogen_bond", fast.hydrogen_bond);
    write("fast_weight_halogen_bond", fast.halogen_bond);
    write("fast_weight_cation", fast.cation);
    write("fast_weight_anion", fast.anion);
    write("fast_weight_aromatic", fast.aromatic);
    write("fast_weight_metal", fast.metal);
    write("fast_weight_chelated", fast.chelated);
    if (weighted_interaction) {
        const WeightedInteractionWeights weights{};
        write("weighted_interaction_weight_hydrophobic", weights.hydrophobic);
        write("weighted_interaction_weight_pi_stacking", weights.pi_stacking);
        write("weighted_interaction_weight_hydrogen_bond", weights.hydrogen_bond);
        write("weighted_interaction_weight_halogen_bond", weights.halogen_bond);
        write("weighted_interaction_weight_cation_pi", weights.cation_pi);
        write("weighted_interaction_weight_ionic", weights.ionic);
        write("weighted_interaction_weight_metal", weights.metal);
    }
    if (settings.collect_timings) {
        write("startup_s", screen.startup_seconds());
        write("wall_before_summary_s", wall_before_summary);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            write_gpu_docking_usage(std::cout); return 0;
        }
        const auto options = parse_gpu_docking_options(argc, argv);
        const auto& settings = options.settings;
        const bool timed = settings.collect_timings;
        const bool extended = settings.ranking.method != PoseScoreMethod::Fast;
        for (const auto& path : {options.config.protein_path, options.config.pocket_path, options.input})
            if (!fs::is_regular_file(path)) throw std::runtime_error("Input not found: " + path.string());
        Mol2Reader reader(options.input);
        std::unique_ptr<Mol2Reader> reference_reader;
        if (!options.references.empty()) reference_reader = std::make_unique<Mol2Reader>(options.references);
        if (fs::exists(options.output)) throw std::runtime_error("Output already exists: " + options.output.string());
        apply_gpu_docking_environment(options);
        const auto run_start = timed ? Clock::now() : Clock::time_point{};
        GpuDockingScreen screen(options.config, settings);
        if (!fs::create_directories(options.output)) throw std::runtime_error("Cannot create output directory");
        std::ofstream poses(options.output / "poses.csv"), records(options.output / "records.csv");
        poses.exceptions(std::ios::badbit | std::ios::failbit);
        records.exceptions(std::ios::badbit | std::ios::failbit);
        write_gpu_docking_pose_header(poses, reference_reader != nullptr, extended);
        records << "record_index,molecule_id,status,diameter_A,diameter_gt_cutoff,ligand_triplets,"
                   "candidate_matches,alignments,valid_transforms,survivors,exported_poses,sdf";
        if (extended) records << ",candidate_pool";
        records << '\n';
        std::ofstream timings;
        if (timed) {
            timings.open(options.output / "timings.csv");
            timings.exceptions(std::ios::badbit | std::ios::failbit);
            timings << std::setprecision(17)
                << "record_index,read_parse_s,ligand_preparation_s,search_completed_s,"
                   "top_k_selection_s,reference_rmsd_export_s";
            if (extended) timings << ",fast_prefilter_s,weighted_interaction_scoring_s,vinardo_scoring_s,"
                "rescoring_chemical_preparation_s,rescoring_h2d_s,rescoring_score_preparation_s,"
                "hybrid_normalization_s,rescoring_top_k_d2h_s";
            timings << '\n';
        }
        std::size_t processed = 0;
        std::string block, name, reference_block, reference_name;
        while (processed < options.limit) {
            const auto parse_start = timed ? Clock::now() : Clock::time_point{};
            if (!reader.next(block, name)) break;
            const auto index = reader.index();
            try {
                auto molecule = parse_ligand_mol2_block(block);
                const double parse_s = timed ? elapsed(parse_start) : 0.0;
                auto result = screen.dock(*molecule, options.top_k);
                if (settings.diagnostic_stages)
                    write_stage_profile(std::cerr, "gpu_docking", index, result.stages);
                const auto export_start = timed ? Clock::now() : Clock::time_point{};
                std::unique_ptr<RDKit::RWMol> reference;
                // Pair references by record position. Ranking has already finished;
                // these coordinates are used only for atom mapping and RMSD export.
                if (reference_reader) {
                    if (!reference_reader->next(reference_block, reference_name))
                        throw std::runtime_error("Missing paired reference MOL2 record");
                    reference = parse_ligand_mol2_block(reference_block);
                    validate_gpu_docking_reference(*molecule, *reference);
                }
                const auto sdf_name = "record-" + std::to_string(index) + ".sdf";
                std::ofstream sdf(options.output / sdf_name);
                sdf.exceptions(std::ios::badbit | std::ios::failbit);
                write_gpu_docking_poses(poses, sdf, *molecule, result.poses, index, name,
                    reference_reader != nullptr, reference.get(), extended ? &result.scores : nullptr);
                sdf.close(); poses.flush();
                const double export_s = timed ? elapsed(export_start) : 0.0;
                const char* status = result.ligand_triplets == 0 ? "no_triplets" :
                                     result.poses.empty() ? "no_poses" : "ok";
                records << std::setprecision(17) << index << ',' << csv_string(name) << ','
                    << status << ',' << result.diameter << ',' << (result.diameter > settings.cutoff) << ','
                    << result.ligand_triplets << ',' << result.candidate_matches << ','
                    << result.alignments << ',' << result.valid_transforms << ','
                    << result.survivors << ',' << result.poses.size() << ',' << csv_string(sdf_name);
                if (extended) records << ',' << result.candidate_pool;
                records << '\n'; records.flush();
                if (timed) {
                    timings << index << ',' << parse_s << ',' << result.timings.preparation_seconds
                        << ',' << result.timings.search_seconds << ',' << result.timings.selection_seconds
                        << ',' << export_s;
                    if (extended) timings << ',' << result.timings.prefilter_seconds << ','
                        << result.timings.weighted_interaction_seconds << ',' << result.timings.vinardo_seconds << ','
                        << result.timings.chemical_preparation_seconds << ',' << result.timings.rescoring_upload_seconds << ','
                        << result.timings.score_preparation_seconds << ','
                        << result.timings.hybrid_seconds << ',' << result.timings.rescoring_download_seconds;
                    timings << '\n'; timings.flush();
                }
            } catch (const std::exception& error) {
                // Unlike screening, stop at the first failed record. Partial
                // output is retained, but no completed run summary is written.
                throw std::runtime_error("Record " + std::to_string(index) + " (" + name + "): " + error.what());
            }
            ++processed;
        }
        if (processed == 0) throw std::runtime_error("Input contains no MOL2 records");
        if (reference_reader && processed < options.limit && reference_reader->next(reference_block, reference_name))
            throw std::runtime_error("Reference MOL2 has more records than the input");
        poses.close(); records.close();
        if (timed) timings.close();
        const double wall_before_summary = timed ? elapsed(run_start) : 0.0;
        const auto summary_temporary = options.output / "summary.csv.tmp";
        std::ofstream summary(summary_temporary);
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        write_summary(summary, options, screen, processed, wall_before_summary);
        summary.close();
        fs::rename(summary_temporary, options.output / "summary.csv");
        std::cout << "Processed " << processed << " ligands; results: " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
