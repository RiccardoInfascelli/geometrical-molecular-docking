// GPU screening shares host preparation and the CPU fixed-cutoff index policy.
// Every ligand uses the single index uploaded at startup.
// Lookup, alignment expansion, Kabsch, clash, compaction and Fast run on device.
// Target buffers persist across ligands; default screening has no RMSD or WeightedInteraction.
// Optional joint export rescores the downloaded survivors on the CPU.

#include "run_config.hpp"
#include "molecule_io.hpp"
#include "ligand_preparation.hpp"
#include "gpu_pose_search.hpp"
#include "pose_rescoring.hpp"
#include "stage_profile.hpp"
#include "cuda_triplet_index.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += '"';
        escaped += ch;
    }
    return escaped + '"';
}

struct Options {
    fs::path target_complex;
    fs::path mol2;
    fs::path output_csv;
    std::size_t limit = 30;
    double cutoff = 24.0;
    std::optional<double> min_target_clearance;
    bool single = false;
    fs::path rescore_output;
    std::optional<std::size_t> rescore_prefilter;
    bool interaction_full_protein = false;
    bool profile = false;
    int query_threads = 1;
};

Options parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error(
            "Usage: multi_ligand_screen_gpu <target_complex_dir> <multi.mol2> <results.csv> "
            "[--limit 30] [--cutoff 24] [--min-target-clearance A] [--single] "
            "[--rescore-output <new-directory> [--rescore-prefilter N] "
            "[--interaction-region pocket|protein]]");
    }
    Options opts;
    opts.profile = stage_profiling_enabled();
    opts.query_threads = gpu_query_requested_threads();
    bool rescore_options_given = false;
    opts.target_complex = argv[1];
    opts.mol2 = argv[2];
    opts.output_csv = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--limit" && i + 1 < argc) {
            opts.limit = std::stoull(argv[++i]);
        } else if (flag == "--cutoff" && i + 1 < argc) {
            opts.cutoff = std::stod(argv[++i]);
        } else if (flag == "--min-target-clearance") {
            if (opts.min_target_clearance)
                throw std::runtime_error("--min-target-clearance may only be specified once");
            if (i + 1 >= argc)
                throw std::runtime_error("--min-target-clearance requires a value in angstroms");
            const std::string value = argv[++i];
            std::size_t parsed = 0;
            try {
                opts.min_target_clearance = std::stod(value, &parsed);
            } catch (const std::exception&) {
                throw std::runtime_error("--min-target-clearance must be finite and non-negative");
            }
            if (parsed != value.size() || !std::isfinite(*opts.min_target_clearance) ||
                *opts.min_target_clearance < 0.0)
                throw std::runtime_error("--min-target-clearance must be finite and non-negative");
        } else if (flag == "--single") {
            opts.single = true;
        } else if (flag == "--rescore-output" && i + 1 < argc) {
            opts.rescore_output = argv[++i];
        } else if (flag == "--rescore-prefilter" && i + 1 < argc) {
            rescore_options_given = true;
            const std::string value = argv[++i];
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("--rescore-prefilter requires a positive integer");
            opts.rescore_prefilter = std::stoull(value);
            if (*opts.rescore_prefilter == 0)
                throw std::runtime_error("--rescore-prefilter must be positive");
        } else if (flag == "--interaction-region" && i + 1 < argc) {
            rescore_options_given = true;
            const std::string value = argv[++i];
            if (value != "pocket" && value != "protein")
                throw std::runtime_error("--interaction-region must be pocket or protein");
            opts.interaction_full_protein = value == "protein";
        } else {
            throw std::runtime_error("Invalid argument: " + flag);
        }
    }
    if (opts.limit == 0 || !std::isfinite(opts.cutoff) || opts.cutoff <= 0.0) {
        throw std::runtime_error("--limit must be positive and --cutoff finite and > 0");
    }
    if (!fs::is_regular_file(opts.mol2)) {
        throw std::runtime_error("Input MOL2 not found: " + opts.mol2.string());
    }
    if (opts.rescore_output.empty() && rescore_options_given)
        throw std::runtime_error("Rescoring options require --rescore-output");
    if (!opts.rescore_output.empty() && fs::exists(opts.rescore_output))
        throw std::runtime_error("Rescoring output directory already exists: " +
                                 opts.rescore_output.string());
    if (opts.profile && !opts.rescore_output.empty())
        throw std::runtime_error("Stage profiling requires screening without rescoring export");
    return opts;
}

void write_target_pruning_summary(std::ostream& output, const PreparedGpuTarget& fixed,
                                 std::optional<double> minimum_clearance, char separator) {
    output << "min_target_clearance_A" << separator;
    if (minimum_clearance) {
        const auto precision = output.precision();
        output << std::setprecision(17) << *minimum_clearance << std::setprecision(precision);
    }
    else output << "disabled";
    output << '\n'
           << "targets_before_pruning" << separator << fixed.protein.targets_before_pruning << '\n'
           << "targets_after_pruning" << separator << fixed.protein.targets.size() << '\n'
           << "targets_removed" << separator << fixed.protein.targets_before_pruning - fixed.protein.targets.size() << '\n'
           << "target_pruning_s" << separator << fixed.protein.target_pruning_seconds << '\n'
           << "target_preparation_excluding_pruning_s" << separator
           << fixed.target_prep_seconds - fixed.protein.target_pruning_seconds << '\n'
           << "shared_index_records" << separator << fixed.index_records << '\n';
}


void write_header(std::ofstream& csv) {
    // Keep column names/order compatible with the CPU screening CSV, then
    // append GPU-specific counts to aid preliminary comparison.
    csv << "record_index,molecule_id,status,error,atoms,interaction_areas,"
           "matching_areas,diameter_A,ligand_triplets,protein_index_mode,"
           "protein_triplet_records,candidate_matches,transforms_generated,"
           "survivors,best_fast_score,ligand_prep_s,pose_search_s,score_s,"
           "total_ligand_s,valid_transforms,query_preparation_s,gpu_search_s,"
           "query_threads_requested,query_threads_used\n";
}

void process_one(std::unique_ptr<RDKit::RWMol> molecule,
                 std::size_t record_index, const std::string& name,
                 const PreparedGpuTarget& fixed, const RunConfig& config,
                 double cutoff, std::ofstream& csv,
                 std::size_t& passed, std::size_t& failed,
                 std::size_t& ligands_diameter_gt_cutoff,
                 double& accumulated_time, Clock::time_point ligand_start,
                 const Options& opts, const RescoringTarget* rescore_target) {
    std::size_t atoms = 0, physical = 0, logical = 0, ntriplets = 0;
    std::size_t matches = 0, transforms = 0, survivors = 0;
    std::size_t protein_records = 0, valid_transforms = 0;
    double diameter = 0.0, prep_s = 0.0, poses_s = 0.0;
    double best_fast_score = 0.0;
    bool has_score = false;
    bool has_search_timings = false;
    double query_s = 0.0, gpu_search_s = 0.0;
    int query_threads_used = 1;
    std::string status = "ok", error;
    constexpr const char* mode = "shared";
    StageProfile stages;

    // Release preparation data before measuring the total ligand time.
    {
        PreparedLigand ligand;
        try {
            if (!molecule) throw std::runtime_error("RDKit returned null");
            atoms = molecule->getNumAtoms();
            prepare_ligand(*molecule, ligand);
            if (ligand.diameter > cutoff) ++ligands_diameter_gt_cutoff;
            prep_s = seconds_since(ligand_start);
            GpuPosePipelineResult pose_result;
            GpuPosePipelineTimings download_times;
            // An empty index has no candidates; no device lookup is necessary.
            if (ligand.triplets.empty()) {
                status = "no_triplets";
            } else if (fixed.index_records != 0) {
                const auto pose_start = Clock::now();
                protein_records = fixed.index_records;
                auto search = run_gpu_ligand_search(*molecule, ligand, fixed, config,
                    opts.profile ? &stages : nullptr,
                    (rescore_target || opts.profile) ? &download_times : nullptr, nullptr, true);
                query_s = search.query_preparation_seconds;
                gpu_search_s = seconds_since(pose_start) - query_s;
                query_threads_used = search.query_threads_used;
                has_search_timings = true;
                matches = search.candidate_matches;
                transforms = search.alignments;
                pose_result = std::move(search.pipeline);
                valid_transforms = pose_result.valid_transform_count;
                survivors = pose_result.poses.size();
                // Kernel scoring is complete. This host pass selects only the
                // maximum Fast value; it performs no rescoring or RMSD evaluation.
                const auto best_start = opts.profile ? Clock::now() : Clock::time_point{};
                for (const auto& pose : pose_result.poses) {
                    if (pose.fast_score.valid &&
                        (!has_score || pose.fast_score.weighted_score > best_fast_score)) {
                        best_fast_score = pose.fast_score.weighted_score;
                        has_score = true;
                    }
                }
                if (opts.profile)
                    stages.push_back({"best_fast_selection", seconds_since(best_start) * 1000.0,
                                      survivors, "observed_host_survivors"});
                // Search now owns and releases device buffers before returning,
                // so this parent interval also includes their cleanup. Selection
                // stays here, after the common search has completed on device.
                poses_s = seconds_since(pose_start);
            }
            if (rescore_target) {
                // Preserve the device survivor sequence and its Fast values. Only
                // WeightedInteraction and Vinardo-like are evaluated on the CPU after download.
                const auto poses = make_rescoring_pose_pool(pose_result.poses);
                const auto scores = rescore_poses(
                    poses, *molecule, ligand, fixed.protein.grid,
                    rescore_target->interaction_areas, rescore_target->vinardo,
                    opts.rescore_prefilter, true);
                const auto prefix = opts.rescore_output /
                    ("record-" + std::to_string(record_index));
                std::ofstream joint_csv(prefix.string() + ".csv");
                joint_csv.exceptions(std::ios::badbit | std::ios::failbit);
                write_pose_scores_csv(joint_csv, scores, record_index);
                joint_csv.close();
                std::ofstream summary(prefix.string() + "-protocol.csv");
                summary.exceptions(std::ios::badbit | std::ios::failbit);
                const double download_s = download_times.survivor_download_seconds +
                                          download_times.fast_score_download_seconds;
                summary << std::setprecision(17)
                        << "key,value\nrecord_index," << record_index << '\n'
                        << "molecule_id," << csv_escape(name) << '\n'
                        << "status," << status << '\n'
                        << "protein_index_mode," << mode << '\n'
                        << "effective_index_cutoff_A," << cutoff << '\n'
                        << "protein_triplet_records," << protein_records << '\n'
                        << "candidate_matches," << matches << '\n'
                        << "transforms_generated," << transforms << '\n'
                        << "input_survivors," << scores.input_pose_count << '\n'
                        << "exported_poses," << scores.poses.size() << '\n'
                        << "ligand_preparation_s," << prep_s << '\n'
                        << "pose_search_including_download_s," << poses_s << '\n'
                        << "pose_search_excluding_download_s," << poses_s - download_s << '\n'
                        << "gpu_survivor_download_s," << download_times.survivor_download_seconds << '\n'
                        << "gpu_fast_score_download_s," << download_times.fast_score_download_seconds << '\n'
                        << "gpu_to_cpu_s," << download_s << '\n'
                        << "prefilter_s," << scores.selection_seconds << '\n'
                        << "weighted_interaction_scoring_s," << scores.weighted_interaction_scoring_seconds << '\n'
                        << "vinardo_scoring_s," << scores.vinardo_scoring_seconds << '\n';
                summary.close();
            }
            ++passed;
        } catch (const GpuLigandSearchError& e) {
            matches = e.candidate_matches;
            transforms = e.alignments;
            valid_transforms = e.valid_transforms;
            survivors = e.survivors;
            status = "error";
            error = e.what();
            ++failed;
        } catch (const std::exception& e) {
            status = "error";
            error = e.what();
            ++failed;
        }
        // Keep counts from completed stages even when preparation fails.
        physical = ligand.areas.size();
        logical = ligand.matching_areas.size();
        diameter = ligand.diameter;
        ntriplets = ligand.triplets.size();
    }
    // With rescoring enabled this also includes CPU rescoring and joint export.
    // The regular screening CSV and the RDKit molecule's destruction stay outside.
    const double total_s = seconds_since(ligand_start);
    accumulated_time += total_s;
    csv << record_index << ',' << csv_escape(name) << ','
        << csv_escape(status) << ',' << csv_escape(error) << ','
        << atoms << ',' << physical << ',' << logical << ','
        << std::setprecision(9) << diameter << ',' << ntriplets << ','
        << mode << ',' << protein_records << ',' << matches << ','
        << transforms << ',' << survivors << ',';
    if (has_score) csv << best_fast_score;
    // Scoring is included in cuda_run_pose_pipeline, i.e. in pose_search_s.
    csv << ',' << prep_s << ',' << poses_s << ",,"
        << total_s << ',' << valid_transforms << ',';
    if (has_search_timings) csv << query_s;
    csv << ',';
    if (has_search_timings) csv << gpu_search_s;
    csv << ',' << opts.query_threads << ',' << query_threads_used << '\n';
    csv.flush();
    if (opts.profile) {
        double measured_search_ms = 0.0;
        for (const auto& stage : stages) measured_search_ms += stage.milliseconds;
        if (status == "ok")
            stages.push_back({"pose_search_residual", poses_s * 1000.0 - measured_search_ms,
                              transforms, "observed_residual_not_kernel_time"});
        stages.push_back({"ligand_parse_preparation", prep_s * 1000.0,
                          atoms, "observed_host_atoms"});
        if (status != "error") {
            stages.push_back({"pose_search", poses_s * 1000.0,
                              transforms, "observed_parent"});
            stages.push_back({"ligand_total_residual", (total_s - prep_s - poses_s) * 1000.0,
                              1, "observed_cleanup_and_bookkeeping"});
        }
        stages.push_back({"ligand_total", total_s * 1000.0, 1, "observed_parent_before_csv"});
        write_stage_profile(std::cerr, "gpu", record_index, stages);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        char program[] = "multi_ligand_screen_gpu";
        const auto target = opts.target_complex.string();
        char* target_argv[] = {program, const_cast<char*>(target.c_str())};
        const RunConfig config = parse_run_config(2, target_argv);
        std::optional<RescoringTarget> rescore_target;
        double scoring_target_s = 0.0;
        if (!opts.rescore_output.empty()) {
            fs::create_directories(opts.rescore_output);
            const auto start = Clock::now();
            rescore_target.emplace(config, opts.interaction_full_protein);
            scoring_target_s = seconds_since(start);
        }
        if (!opts.output_csv.parent_path().empty())
            fs::create_directories(opts.output_csv.parent_path());
        std::ofstream csv(opts.output_csv);
        if (!csv) throw std::runtime_error("Cannot open output CSV");
        write_header(csv);
        std::cerr << "Preparing GPU target " << config.complex_id << " ...\n";
        const auto all_start = Clock::now();
        const PreparedGpuTarget fixed(config, opts.cutoff, opts.min_target_clearance);
        if (rescore_target) {
            std::ofstream summary(opts.rescore_output / "protocol.csv");
            summary.exceptions(std::ios::badbit | std::ios::failbit);
            auto scoring_config = config;
            scoring_config.ligand_path = fs::absolute(opts.mol2).lexically_normal();
            write_rescoring_protocol(summary, scoring_config, opts.interaction_full_protein,
                                     opts.rescore_prefilter);
            summary << std::setprecision(17)
                    << "backend,gpu_generation_cpu_rescoring\nfast_source,cuda_pose_pipeline\n"
                    << "generation_diagnostics,no RMSD or unweighted grid support\n"
                    << "input_mol2," << csv_escape(opts.mol2.string()) << '\n'
                    << "limit," << opts.limit << '\n'
                    << "single," << opts.single << '\n'
                    << "index_policy,shared_index_only\n"
                    << "index_cutoff_A," << opts.cutoff << '\n'
                    << "target_preparation_s," << fixed.target_prep_seconds << '\n'
                    << "shared_index_build_s," << fixed.index_seconds << '\n'
                    << "shared_gpu_upload_s," << fixed.gpu_upload_seconds << '\n'
                    << "scoring_target_preparation_s," << scoring_target_s << '\n'
                    << "vinardo_skipped_receptor_atoms,"
                    << rescore_target->vinardo.receptor_skipped_unsupported() << '\n';
            write_target_pruning_summary(summary, fixed, opts.min_target_clearance, ',');
            summary.close();
        }
        std::cerr << "Targets: " << fixed.protein.targets.size()
                  << "; index: " << fixed.index_records
                  << " records, " << fixed.index_keys << " keys\n"
                  << "Target preparation: " << fixed.target_prep_seconds
                  << " s; index build: " << fixed.index_seconds
                  << " s; GPU upload: " << fixed.gpu_upload_seconds << " s\n";
        if (opts.min_target_clearance) {
            std::cerr << "Target pruning: " << fixed.protein.targets_before_pruning
                      << " -> " << fixed.protein.targets.size()
                      << "; minimum clearance: " << *opts.min_target_clearance
                      << " A; " << fixed.protein.target_pruning_seconds << " s\n";
        }

        std::size_t processed = 0, passed = 0, failed = 0;
        std::size_t ligands_diameter_gt_cutoff = 0;
        double ligand_seconds = 0.0;
        if (opts.single) {
            const auto ligand_start = Clock::now();
            auto mol = load_ligand(opts.mol2.string());
            process_one(std::move(mol), 1, opts.mol2.filename().string(),
                        fixed, config, opts.cutoff, csv,
                        passed, failed, ligands_diameter_gt_cutoff, ligand_seconds, ligand_start,
                        opts, rescore_target ? &*rescore_target : nullptr);
            processed = 1;
        } else {
            Mol2Reader reader(opts.mol2);
            std::string block, name;
            while (processed < opts.limit) {
                const auto ligand_start = Clock::now();
                if (!reader.next(block, name)) break;
                std::unique_ptr<RDKit::RWMol> mol;
                try {
                    mol = parse_ligand_mol2_block(block);
                } catch (const std::exception& e) {
                    std::cerr << "MOL2 record " << reader.index()
                              << " parse error: " << e.what() << '\n';
                    mol.reset();
                }
                process_one(std::move(mol), reader.index(), name,
                            fixed, config, opts.cutoff, csv,
                            passed, failed, ligands_diameter_gt_cutoff, ligand_seconds, ligand_start,
                            opts, rescore_target ? &*rescore_target : nullptr);
                ++processed;
                if (processed % 10 == 0 || processed == opts.limit)
                    std::cerr << "Processed " << processed << "/" << opts.limit
                              << ", failures=" << failed
                              << ", elapsed=" << seconds_since(all_start) << " s\n";
            }
        }
        csv.close();
        std::cout << std::fixed << std::setprecision(6)
                  << "target=" << config.complex_id << '\n'
                  << "cutoff_A=" << opts.cutoff << '\n'
                  << "processed=" << processed << '\n'
                  << "ok_or_no_triplets=" << passed << '\n'
                  << "errors=" << failed << '\n'
                  << "ligands_diameter_gt_cutoff=" << ligands_diameter_gt_cutoff << '\n'
                  << "target_preparation_s=" << fixed.target_prep_seconds << '\n'
                  << "shared_index_build_s=" << fixed.index_seconds << '\n'
                  << "shared_gpu_upload_s=" << fixed.gpu_upload_seconds << '\n'
                  << "sum_ligand_s=" << ligand_seconds << '\n'
                  << "wall_total_s=" << seconds_since(all_start) << '\n'
                  << "results_csv=" << opts.output_csv << '\n';
        std::cout << "query_threads_requested=" << opts.query_threads << '\n'
                  << "query_openmp_available=" << gpu_query_openmp_available() << '\n'
                  << "query_parallel_min_triplets=" << gpu_query_parallel_min_triplets << '\n'
                  << "query_min_triplets_per_requested_thread=" << gpu_query_min_per_thread << '\n'
                  << "gpu_search_scope=completed search excluding query preparation; includes host packing, copies and CUDA waits\n"
                  << "cpu_gpu_overlap=off\n";
        // Additional startup metadata stays outside the ligand CSV and its timers.
        write_target_pruning_summary(std::cout, fixed, opts.min_target_clearance, '=');
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
