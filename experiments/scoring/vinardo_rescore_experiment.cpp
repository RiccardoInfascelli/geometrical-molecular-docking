#include "vinardo_scoring.hpp"
#include "molecule_io.hpp"
#include "pipeline.hpp"
#include "ligand_preparation.hpp"
#include "pose_rescoring.hpp"
#include "protein_preparation.hpp"
#include "joint_diagnostics.hpp"
#include "gpu_docking_output.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct VinardoRankedPose {
    std::size_t fast_rank{};
    RankedPoseDiagnostic pose;
    VinardoLikeScore energy;
};

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void write_diagnostic_status(const fs::path& output, const std::string& status,
    const std::string& stage, const std::string& message,
    std::optional<std::size_t> survivors, double wall_seconds) {
    std::ofstream csv(output / "status.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "key,value\n";
    write_gpu_docking_summary_entry(csv, "status", status);
    write_gpu_docking_summary_entry(csv, "stage", stage);
    write_gpu_docking_summary_entry(csv, "message", message);
    write_gpu_docking_summary_entry(csv, "input_survivors", survivors ? std::to_string(*survivors) : "");
    write_gpu_docking_summary_entry(csv, "completed", status == "ok" || status == "no_poses" ? "1" : "0");
    std::ostringstream seconds;
    seconds << std::setprecision(17) << wall_seconds;
    write_gpu_docking_summary_entry(csv, "wall_before_status_s", seconds.str());
}

void write_joint_protocol(std::ostream& csv, const RunConfig& config,
    bool full_protein, std::optional<std::size_t> prefilter, bool diagnostics) {
    if (!diagnostics) {
        write_rescoring_protocol(csv, config, full_protein, prefilter);
        return;
    }
    std::ostringstream original;
    write_rescoring_protocol(original, config, full_protein, prefilter);
    std::istringstream rows(original.str());
    std::string row;
    while (std::getline(rows, row)) {
        if (row.starts_with("rmsd,")) {
            write_gpu_docking_summary_entry(csv, "rmsd",
                "direct heavy atoms; verified fixed atom indices; no alignment or symmetry correction; evaluation only");
        } else {
            csv << row << '\n';
        }
    }
}

// This explicit mode leaves the positional top-N experiment below unchanged.
int run_joint_scoring(int argc, char* argv[]) {
    if (argc < 4) {
        throw std::runtime_error(
            "Usage: vinardo_rescore_experiment <complex> --joint-output <new-directory> "
            "[--prefilter N] [--interaction-region pocket|protein] [--cutoff 24] "
            "[--diagnostics [--reference crystal.mol2]]");
    }
    char* config_args[]{argv[0], argv[1]};
    const auto config = parse_run_config(2, config_args);
    const fs::path output = argv[3];
    std::optional<std::size_t> prefilter;
    bool full_protein = false;
    double cutoff = 24.0;
    bool diagnostics = false;
    fs::path reference_path;
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--diagnostics") {
            diagnostics = true;
            continue;
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--prefilter") {
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("--prefilter requires a positive integer");
            prefilter = std::stoull(value);
            if (*prefilter == 0) throw std::runtime_error("--prefilter must be positive");
        } else if (flag == "--interaction-region") {
            if (value != "pocket" && value != "protein")
                throw std::runtime_error("--interaction-region must be pocket or protein");
            full_protein = value == "protein";
        } else if (flag == "--reference") {
            reference_path = value;
        } else if (flag == "--cutoff") {
            std::size_t consumed = 0;
            cutoff = std::stod(value, &consumed);
            if (consumed != value.size() || !std::isfinite(cutoff) || cutoff <= 0.0)
                throw std::runtime_error("--cutoff must be finite and positive");
        } else {
            throw std::runtime_error("Invalid argument: " + flag);
        }
    }
    if (diagnostics && prefilter)
        throw std::runtime_error("--diagnostics evaluates the full pool and Fast Top-500; omit --prefilter");
    if (!diagnostics && !reference_path.empty())
        throw std::runtime_error("--reference requires --diagnostics");
    if (fs::exists(output))
        throw std::runtime_error("Output directory already exists: " + output.string());
    fs::create_directories(output);

    const auto wall_start = Clock::now();
    std::string stage = "preparation";
    bool invalid_reference = false;
    std::optional<std::size_t> survivors;
    try {
        std::unique_ptr<RDKit::RWMol> reference;
        double reference_s = 0.0;
        if (diagnostics) {
            stage = "reference_loading";
            const auto reference_start = Clock::now();
            try {
                if (reference_path.empty()) reference_path = config.ligand_path;
                reference = load_ligand(reference_path.string());
                validate_gpu_docking_reference(*reference, *reference);
            } catch (...) {
                invalid_reference = true;
                throw;
            }
            reference_s = elapsed(reference_start);
        }

        stage = "protein_preparation";
        auto start = Clock::now();
        const auto protein = prepare_protein(config);
        const double target_s = elapsed(start);

        stage = "ligand_preparation";
        start = Clock::now();
        auto molecule = load_ligand(config.ligand_path.string());
        PreparedLigand ligand;
        prepare_ligand(*molecule, ligand);
        const double ligand_s = elapsed(start);

        if (diagnostics) {
            stage = "reference_validation";
            start = Clock::now();
            try {
                validate_gpu_docking_reference(*molecule, *reference);
            } catch (...) {
                invalid_reference = true;
                throw;
            }
            reference_s += elapsed(start);
        }

        stage = "index_build";
        start = Clock::now();
        const bool shared = ligand.diameter <= cutoff;
        const auto hash = build_protein_triplet_hash_table(
            protein.targets, shared ? cutoff : ligand.diameter, config.triplet_bin_size);
        const auto index = flatten_protein_triplet_index(hash);
        const double index_s = elapsed(start);

        stage = "pose_generation";
        start = Clock::now();
        // Only evaluation diagnostics are skipped: every operational clash survivor
        // is retained before the optional, explicitly requested scoring prefilter.
        const auto generated = generate_candidate_poses_from_index(
            *molecule, ligand.matching_areas, ligand.triplets, protein.targets,
            protein.grid, index, config.triplet_bin_size,
            config.operational_clash_threshold, config.clash_thresholds, true);
        const double generation_s = elapsed(start);
        survivors = generated.surviving_poses.size();

        stage = "scoring";
        start = Clock::now();
        const RescoringTarget target(config, full_protein);
        const double scoring_target_s = elapsed(start);
        const auto scores = rescore_poses(
            generated.surviving_poses, *molecule, ligand, protein.grid,
            target.interaction_areas, target.vinardo, prefilter);

        std::vector<joint_diagnostics::PoseMetrics> metrics;
        double metrics_s = 0.0;
        if (diagnostics) {
            stage = "pose_metrics";
            start = Clock::now();
            metrics = joint_diagnostics::evaluate_pose_metrics(scores, *molecule, *reference);
            metrics_s = elapsed(start);
        }

        stage = "export";
        start = Clock::now();
        std::ofstream csv(output / "poses.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        if (diagnostics) joint_diagnostics::write_pose_csv(csv, scores, metrics);
        else write_pose_scores_csv(csv, scores, 1);
        csv.close();
        const double pose_export_s = elapsed(start);
        double ranking_export_s = 0.0;
        if (diagnostics) {
            stage = "ranking_analysis_export";
            start = Clock::now();
            std::ofstream rankings(output / "rankings.csv"), topk(output / "topk_metrics.csv"),
                analysis(output / "summary.csv");
            for (auto* file : {&rankings, &topk, &analysis})
                file->exceptions(std::ios::badbit | std::ios::failbit);
            joint_diagnostics::write_analysis_csvs(rankings, topk, analysis, scores, metrics);
            rankings.close();
            topk.close();
            analysis.close();
            ranking_export_s = elapsed(start);
        }
        stage = "protocol_export";
        std::ofstream summary(output / "protocol.csv");
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        write_joint_protocol(summary, config, full_protein, prefilter, diagnostics);
        summary << std::setprecision(17)
                << "backend,cpu\nfast_source,cpu_rescoring\n"
                << "cpu_generation_diagnostics,screening_only forced true; RMSD and unweighted grid support skipped\n"
                << "index_policy,shared_cutoff_with_ligand_diameter_fallback\n"
                << "index_cutoff_A," << cutoff << '\n'
                << "protein_index_mode," << (shared ? "shared" : "fallback") << '\n'
                << "effective_index_cutoff_A," << (shared ? cutoff : ligand.diameter) << '\n'
                << "protein_triplet_records," << index.records.size() << '\n'
                << "candidate_matches," << generated.total_candidate_matches << '\n'
                << "transforms_generated," << generated.transforms_generated << '\n'
                << "input_survivors," << scores.input_pose_count << '\n'
                << "exported_poses," << scores.poses.size() << '\n'
                << "target_preparation_s," << target_s << '\n'
                << "ligand_preparation_s," << ligand_s << '\n'
                << "index_build_s," << index_s << '\n'
                << "pose_generation_s," << generation_s << '\n'
                << "gpu_to_cpu_s,0\n"
                << "scoring_target_preparation_s," << scoring_target_s << '\n'
                << "fast_scoring_s," << scores.fast_scoring_seconds << '\n'
                << "prefilter_s," << scores.selection_seconds << '\n'
                << "weighted_interaction_scoring_s," << scores.weighted_interaction_scoring_seconds << '\n'
                << "vinardo_scoring_s," << scores.vinardo_scoring_seconds << '\n'
                << "vinardo_skipped_receptor_atoms," << target.vinardo.receptor_skipped_unsupported() << '\n';
        if (diagnostics) {
            write_gpu_docking_summary_entry(summary, "diagnostics", "full and fast_top500; one search and one atomic rescoring");
            write_gpu_docking_summary_entry(summary, "reference_path", reference_path.string());
            write_gpu_docking_summary_entry(summary, "reference_atom_mapping", "fixed indices; complete atom and bond graph checked including explicit H; no symmetry permutations");
            write_gpu_docking_summary_entry(summary, "ranking_tie_break", "score then original pose_id ascending; shared rank_rescored_poses, not legacy rank_poses ties");
            write_gpu_docking_summary_entry(summary, "hybrid_normalization_pool", "separate population z-scores for full_pool and fast_top500 before Top-20 selection");
            summary << "vinardo_standard_alpha,1\nvinardo_rep50_alpha,0.5\nhybrid_alpha,0.5\nhybrid_lambda,0.3\n"
                    << "fast_subset_size,500\nexport_top_k,20\n"
                    << "success_rmsd_2_A,strict less than 2\nsuccess_rmsd_3_A,strict less than 3\n"
                    << "success_center_5_A,strict less than 5\n"
                    << "protein_targets," << protein.targets.size() << '\n'
                    << "ligand_diameter_A," << ligand.diameter << '\n'
                    << "reference_preparation_s," << reference_s << '\n'
                    << "pose_metrics_s," << metrics_s << '\n'
                    << "poses_export_s," << pose_export_s << '\n'
                    << "ranking_analysis_export_s," << ranking_export_s << '\n';
        }
        summary.close();
        if (diagnostics) write_diagnostic_status(output, scores.poses.empty() ? "no_poses" : "ok",
            "complete", "", survivors, elapsed(wall_start));
        std::cout << "Joint scores: " << scores.poses.size() << " / "
                  << scores.input_pose_count << " clash survivors; output " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (diagnostics) {
            try {
                write_diagnostic_status(output, invalid_reference ? "invalid_reference" : "error",
                    stage, error.what(), survivors, elapsed(wall_start));
            } catch (const std::exception& status_error) {
                std::cerr << "Cannot write diagnostic status: " << status_error.what() << '\n';
            }
        }
        throw;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc >= 3 && std::string(argv[2]) == "--joint-output")
            return run_joint_scoring(argc, argv);
        if (argc < 2 || argc > 4) {
            std::cerr << "Uso: " << argv[0]
                      << " <directory_complesso> [top_N=10] [out.csv]\n";
            return 2;
        }

        RunConfig config;
        config.complex_directory = fs::absolute(argv[1]).lexically_normal();
        if (config.complex_directory.filename().empty())
            config.complex_directory = config.complex_directory.parent_path();
        config.complex_id = config.complex_directory.filename().string();
        const auto prefix = config.complex_directory / config.complex_id;
        config.protein_path = prefix.string() + "_protein.pdb";
        config.ligand_path = prefix.string() + "_ligand.mol2";
        config.pocket_path = prefix.string() + "_pocket.pdb";
        const std::size_t top_n = argc >= 3
            ? static_cast<std::size_t>(std::stoul(argv[2])) : 10;
        if (top_n == 0 || top_n > 5000)
            throw std::runtime_error("top_N deve essere tra 1 e 5000");
        config.fast_prefilter_top_n = top_n;

        const fs::path out = argc >= 4
            ? fs::path(argv[3])
            : fs::path("vinardo_like_" + config.complex_id + ".csv");
        if (!out.parent_path().empty())
            fs::create_directories(out.parent_path());

        std::cout << "Eseguo pipeline CPU per " << config.complex_id
                  << ", fast top-" << top_n << "...\n";
        // This already computes WeightedInteraction scores on the Fast top-N candidate pool.
        const auto run = run_pipeline(config);
        std::cout << "CPU search: candidate_matches=" << run.pose_generation.total_candidate_matches
                  << " transforms_generated=" << run.pose_generation.transforms_generated
                  << " survivors=" << run.pose_generation.surviving_poses.size() << '\n'
                  << "Target preparation: " << run.timings.offline_target_preparation_seconds
                  << " s; index + pose search: " << run.timings.online_pose_search_seconds
                  << " s; CPU pipeline total: " << run.timings.total_seconds << " s\n";
        const auto& candidates = run.pose_ranking.weighted_interaction_candidates;
        if (candidates.empty()) {
            std::cerr << "Nessuna posa sopravvissuta al fast prefilter.\n";
            return 3;
        }

        const auto t1 = std::chrono::steady_clock::now();
        const auto receptor = load_protein(config.protein_path.string());
        const double receptor_load_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        const auto t2 = std::chrono::steady_clock::now();
        const auto ligand = load_ligand(config.ligand_path.string());
        const double ligand_load_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();

        const auto t3 = std::chrono::steady_clock::now();
        const VinardoLikeScorer scorer(*receptor);
        const double receptor_prep_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count();

        std::vector<VinardoRankedPose> rows;
        rows.reserve(candidates.size());

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < candidates.size(); ++i)
            rows.push_back({i + 1, candidates[i],
                            scorer.score(*ligand, candidates[i].transform)});
        const double scoring_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::cout << std::fixed << std::setprecision(6)
                  << "Caricamento proteina: " << receptor_load_s << " s\n"
                  << "Caricamento ligando: " << ligand_load_s << " s\n"
                  << "Preparazione recettore Vinardo-like: " << receptor_prep_s << " s\n"
                  << "Vinardo-like preparazione + scoring: "
                  << (receptor_prep_s + scoring_seconds) << " s\n"
                  << "Vinardo-like incluso caricamento file: "
                  << (receptor_load_s + ligand_load_s +
                      receptor_prep_s + scoring_seconds) << " s\n";


        // The sums below compare scoring intervals, not the full executable time.
        const double fast_s = run.pose_ranking.fast_scoring_seconds;

        std::cout << std::fixed << std::setprecision(6)
                  << "Tempo SOLO fast scoring: " << fast_s << " s\n"
                  << "RMSD Top-1 fast-only: "
                  << run.pose_ranking.fast_weighted_ranking.front().rmsd << " A\n"
                  << "Tempo FAST + Vinardo-like: "
                  << fast_s + scoring_seconds << " s\n"
                  << "Tempo FAST + WeightedInteraction: "
                  << fast_s + run.pose_ranking.weighted_interaction_scoring_seconds << " s\n";
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            if (a.energy.total != b.energy.total)
                return a.energy.total < b.energy.total;
            return a.fast_rank < b.fast_rank;
        });

        std::ofstream csv(out);
        if (!csv) throw std::runtime_error("Impossibile scrivere " + out.string());
        csv << "vinardo_like_rank,fast_rank,rmsd_A,weighted_interaction_score,"
               "vinardo_like_score,gaussian,repulsion,hydrophobic,hbond\n";
        csv << std::setprecision(12);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            csv << i + 1 << ',' << r.fast_rank << ',' << r.pose.rmsd << ','
                << r.pose.weighted_interaction_score.weighted_score << ','
                << r.energy.total << ',' << r.energy.gaussian << ','
                << r.energy.repulsion << ',' << r.energy.hydrophobic << ','
                << r.energy.hydrogen_bond << '\n';
        }

        const auto& old_rank = run.pose_ranking.weighted_interaction_ranking;
        std::cout << std::fixed << std::setprecision(4)
                  << "Pose confrontate: " << rows.size() << '\n'
                  << "RMSD Top-1 attuale: " << old_rank.front().rmsd << " A\n"
                  << "RMSD Top-1 Vinardo-like: " << rows.front().pose.rmsd << " A\n"
                  << "Tempo SOLO rescoring Vinardo-like: " << scoring_seconds << " s\n"
                  << "Tempo/posa: " << 1000.0 * scoring_seconds / rows.size() << " ms\n"
                  << "Atomi recettore ignorati (elementi non supportati): "
                  << scorer.receptor_skipped_unsupported() << '\n'
                  << "CSV: " << out << '\n'
                  << "NOTA: score sperimentale NON equivalente a Vinardo/PDBQT, "
                     "NON kcal/mol.\n";

        const double weighted_interaction_s = run.pose_ranking.weighted_interaction_scoring_seconds;
        std::cout << std::fixed << std::setprecision(6)
                  << "Tempo scoring WeightedInteraction attuale: " << weighted_interaction_s << " s\n"
                  << "Tempo rescoring Vinardo-like: " << scoring_seconds << " s\n"
                  << "Rapporto tempi (attuale / Vinardo-like): "
                  << (weighted_interaction_s / scoring_seconds) << "x\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERRORE: " << e.what() << '\n';
        return 1;
    }
}
