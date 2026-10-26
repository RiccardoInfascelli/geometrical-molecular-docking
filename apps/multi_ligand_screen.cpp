// CPU screening reuses one protein grid, target set and fixed-cutoff index.
// Every ligand uses that index, even when its diameter exceeds the cutoff.
// Reports the maximum Fast score among survivors; no WeightedInteraction ranking is performed.

#include "run_config.hpp"
#include "molecule_io.hpp"
#include "ligand_preparation.hpp"
#include "geometry.hpp"
#include "protein_preparation.hpp"
#include "triplet_generation.hpp"
#include "pose_generation.hpp"
#include "interaction_scoring.hpp"
#include "stage_profile.hpp"

#include <interaction/sanitize_mol.hpp>
#include <GraphMol/FileParsers/FileParsers.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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
    std::size_t limit = 500;
    double cutoff = 24.0;
    bool single = false;
    bool verify = false;
    bool profile = false;
};

Options parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error(
            "Usage: multi_ligand_screen <target_complex_dir> <multi.mol2> <results.csv> "
            "[--limit 500] [--cutoff 24] [--single] [--verify]\n"
            "--single: input is one ordinary MOL2 file (verification enabled)\n"
            "--verify: validate the shared index once at startup");
    }
    Options opts;
    opts.profile = stage_profiling_enabled();
    opts.target_complex = argv[1];
    opts.mol2 = argv[2];
    opts.output_csv = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--limit" && i + 1 < argc) {
            opts.limit = std::stoull(argv[++i]);
        } else if (flag == "--cutoff" && i + 1 < argc) {
            opts.cutoff = std::stod(argv[++i]);
        } else if (flag == "--single") {
            opts.single = true;
            // Unlike the GPU application's --single, this also enables verification.
            opts.verify = true;
        } else if (flag == "--verify") {
            opts.verify = true;
        } else {
            throw std::runtime_error("Invalid argument: " + flag);
        }
    }
    if (opts.limit == 0 || !std::isfinite(opts.cutoff) || opts.cutoff <= 0.0) {
        throw std::runtime_error("--limit must be > 0 and --cutoff finite and > 0");
    }
    if (!fs::is_regular_file(opts.mol2)) {
        throw std::runtime_error("Input MOL2 not found: " + opts.mol2.string());
    }
    if (opts.profile && opts.verify)
        throw std::runtime_error("Stage profiling requires screening without --verify or --single");
    return opts;
}

struct FixedTarget {
    explicit FixedTarget(const RunConfig& config, double shared_cutoff, bool verify) {
        const auto target_start = Clock::now();
        protein = prepare_protein(config);
        target_prep_seconds = seconds_since(target_start);

        const auto index_start = Clock::now();
        const auto shared_hash = build_protein_triplet_hash_table(
            protein.targets, shared_cutoff, config.triplet_bin_size);
        shared_index = flatten_protein_triplet_index(shared_hash);
        index_seconds = seconds_since(index_start);
        index_records = shared_index.records.size();
        if (verify && !validate_flat_triplet_index(shared_hash, shared_index))
            throw std::runtime_error("Shared protein index validation failed");
    }

    PreparedProtein protein;
    FlatTripletIndex shared_index;
    std::size_t index_records = 0;
    double target_prep_seconds = 0.0;
    double index_seconds = 0.0;
};


void write_header(std::ofstream& output) {
    output << "record_index,molecule_id,status,error,atoms,interaction_areas,"
              "matching_areas,diameter_A,ligand_triplets,protein_index_mode,"
              "protein_triplet_records,candidate_matches,transforms_generated,"
              "survivors,best_fast_score,ligand_prep_s,pose_search_s,score_s,total_ligand_s\n";
}

void process_one(std::unique_ptr<RDKit::RWMol> molecule,
                 std::size_t record_index, const std::string& name,
                 const FixedTarget& fixed, const RunConfig& config,
                 double cutoff, bool profile, std::ofstream& csv,
                 std::size_t& passed, std::size_t& failed,
                 std::size_t& ligands_diameter_gt_cutoff,
                 double& accumulated_time, Clock::time_point total_start) {
    std::size_t atoms = 0, physical = 0, logical = 0, ntriplets = 0;
    std::size_t matches = 0, transforms = 0, survivors = 0, protein_records = 0;
    std::size_t valid_transforms = 0;
    double diameter = 0.0, prep_s = 0.0, poses_s = 0.0, scoring_s = 0.0;
    double observed_clash_s = 0.0;
    double best_fast_score = 0.0;
    std::string status = "ok", error;
    constexpr const char* mode = "shared";
    // Opt out here, not globally: full ranking callers need support and RMSD.
    const char* screening_mode = std::getenv("DOCKING_CPU_SCREENING_ONLY");
    const bool screening_only = screening_mode && std::strcmp(screening_mode, "1") == 0;

    // Release preparation data before measuring the total ligand time.
    {
        PreparedLigand ligand;
        try {
            if (!molecule) throw std::runtime_error("RDKit returned null");
            atoms = molecule->getNumAtoms();
            prepare_ligand(*molecule, ligand);
            if (ligand.diameter > cutoff) ++ligands_diameter_gt_cutoff;
            // Includes streaming read, RDKit MOL2 parsing, sanitization and ligand prep.
            prep_s = seconds_since(total_start);
            if (ligand.triplets.empty()) {
                status = "no_triplets";
            } else {
                const auto pose_start = Clock::now();
                // The shared index may admit extra neighboring-bin records beyond
                // this ligand's diameter; pose generation does not reapply that cutoff.
                const auto generated = generate_candidate_poses_from_index(
                    *molecule, ligand.matching_areas, ligand.triplets,
                    fixed.protein.targets, fixed.protein.grid,
                    fixed.shared_index, config.triplet_bin_size,
                    config.operational_clash_threshold, config.clash_thresholds,
                    screening_only);

                poses_s = seconds_since(pose_start);
                protein_records = generated.protein_triplet_records;
                matches = generated.total_candidate_matches;
                transforms = generated.transforms_generated;
                survivors = generated.surviving_poses.size();
                if (profile) {
                    valid_transforms = generated.transforms_valid;
                    observed_clash_s = generated.timings.grid_clash_seconds;
                }
                // This is only a maximum over survivors, not the legacy WeightedInteraction
                // ranking or Hybrid; an equal score does not replace the maximum.
                const auto rank_start = Clock::now();
                bool has_score = false;
                for (const auto& pose : generated.surviving_poses) {
                    const auto transformed = transform_matching_areas(
                        ligand.matching_areas, pose.transform);
                    const auto scored = score_matching_areas_weighted(
                        transformed, fixed.protein.grid);
                    if (!has_score || scored.weighted_score > best_fast_score) {
                        best_fast_score = scored.weighted_score;
                        has_score = true;
                    }
                }
                scoring_s = seconds_since(rank_start);
            }
            ++passed;
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
    // CSV write/flush and destruction of the RDKit molecule are outside this timer.
    const double total_s = seconds_since(total_start);
    accumulated_time += total_s;
    csv << record_index << ',' << csv_escape(name) << ','
        << csv_escape(status) << ',' << csv_escape(error) << ','
        << atoms << ',' << physical << ',' << logical << ','
        << std::setprecision(9) << diameter << ',' << ntriplets << ','
        << mode << ',' << protein_records << ',' << matches << ','
        << transforms << ',' << survivors << ',';
    if (survivors > 0) csv << best_fast_score;
    csv << ',' << prep_s << ',' << poses_s << ',' << scoring_s
        << ',' << total_s << '\n';
    csv.flush(); // Records remain available even if a long run is interrupted.

    if (profile) {
        StageProfile stages{
            {"ligand_parse_preparation", prep_s * 1000.0, atoms, "observed_host_atoms"},
            {"pose_search", poses_s * 1000.0, transforms, "observed_fused_parent"},
            {"fast_scoring", scoring_s * 1000.0, survivors, "observed_host_survivors"},
            {"ligand_total_residual", (total_s - prep_s - poses_s - scoring_s) * 1000.0,
             1, "observed_cleanup_and_bookkeeping"},
            {"ligand_total", total_s * 1000.0, 1, "observed_parent_before_csv"}
        };
        if (status == "error")
            stages = {{"ligand_total", total_s * 1000.0, 1, "observed_failed_parent"}};
        if (status == "ok") {
            // Rebuild only diagnostic inputs after the original result and timer
            // are finalized. The normal path keeps its original object lifetimes.
            const auto rebuild_start = Clock::now();
            PreparedLigand prepared;
            prepare_ligand(*molecule, prepared);
            stages.push_back({"profile_input_rebuild", seconds_since(rebuild_start) * 1000.0,
                              atoms, "diagnostic_only_host"});
            const auto replay = profile_cpu_pose_stages(
                *molecule, prepared.matching_areas, prepared.triplets,
                fixed.protein.targets, fixed.protein.grid, fixed.shared_index,
                config.triplet_bin_size, config.operational_clash_threshold,
                config.clash_thresholds);
            if (replay.candidate_matches != matches || replay.transforms_generated != transforms ||
                replay.transforms_valid != valid_transforms ||
                replay.survivors != survivors || replay.has_fast_score != (survivors != 0) ||
                (replay.has_fast_score && replay.best_fast_score != best_fast_score)) {
                throw std::runtime_error("CPU profiling replay differs from the screening result");
            }
            stages.push_back({"grid_clash_existing_timer", observed_clash_s * 1000.0,
                              valid_transforms, "observed_nested_per_fit_timer"});
            stages.insert(stages.end(), replay.stages.begin(), replay.stages.end());
            double replay_children_ms = 0.0, replay_total_ms = 0.0;
            for (const auto& stage : replay.stages) {
                if (stage.phase == "replay_total") replay_total_ms = stage.milliseconds;
                else replay_children_ms += stage.milliseconds;
            }
            stages.push_back({"replay_setup_residual", replay_total_ms - replay_children_ms,
                              transforms, "replay_allocations_and_bookkeeping"});
        }
        write_stage_profile(std::cerr, "cpu", record_index, stages);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        char program[] = "multi_ligand_screen";
        const auto target = opts.target_complex.string();
        // parse_run_config() expects char* argv[], but does not modify args.
        char* target_argv[] = {program, const_cast<char*>(target.c_str())};
        const RunConfig config = parse_run_config(2, target_argv);
        if (!opts.output_csv.parent_path().empty())
            fs::create_directories(opts.output_csv.parent_path());
        std::ofstream csv(opts.output_csv);
        if (!csv) throw std::runtime_error("Cannot open output CSV");
        write_header(csv);

        std::cerr << "Preparing protein target " << config.complex_id << " ...\n";
        const auto all_start = Clock::now();
        const FixedTarget fixed(config, opts.cutoff, opts.verify);
        std::cerr << "Targets: " << fixed.protein.targets.size()
                  << "; index: " << fixed.index_records
                  << " records, " << fixed.shared_index.keys.size() << " keys\n"
                  << "Target preparation: " << fixed.target_prep_seconds
                  << " s; index build: " << fixed.index_seconds << " s\n";

        std::size_t processed = 0, passed = 0, failed = 0;
        std::size_t ligands_diameter_gt_cutoff = 0;
        double ligand_seconds = 0.0;
        if (opts.single) {
            const auto ligand_start = Clock::now();
            auto mol = load_ligand(opts.mol2.string());
            process_one(std::move(mol), 1, opts.mol2.filename().string(),
                        fixed, config, opts.cutoff, opts.profile, csv,
                        passed, failed, ligands_diameter_gt_cutoff, ligand_seconds, ligand_start);
            processed = 1;
        } else {
            Mol2Reader reader(opts.mol2);
            std::string block, name;
            while (processed < opts.limit) {
                const auto ligand_start = Clock::now();
                if (!reader.next(block, name)) break;
                std::unique_ptr<RDKit::RWMol> mol;
                try {
                    mol.reset(RDKit::Mol2BlockToMol(block, false, false));
                    if (mol) interaction::sanitize_molecule(mol);
                } catch (const std::exception& e) {
                    std::cerr << "MOL2 record " << reader.index()
                              << " parse error: " << e.what() << '\n';
                    mol.reset();
                }
                process_one(std::move(mol), reader.index(), name,
                            fixed, config, opts.cutoff, opts.profile, csv,
                            passed, failed, ligands_diameter_gt_cutoff, ligand_seconds, ligand_start);
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
                  << "sum_ligand_s=" << ligand_seconds << '\n'
                  << "wall_total_s=" << seconds_since(all_start) << '\n'
                  << "results_csv=" << opts.output_csv << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
