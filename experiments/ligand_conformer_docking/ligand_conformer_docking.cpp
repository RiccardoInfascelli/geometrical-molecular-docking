#include "conformer_generation.hpp"
#include "conformer_docking.hpp"
#include "gpu_docking_output.hpp"
#include "molecule_io.hpp"
#include "protein_area_crop.hpp"

#include <RDGeneral/versions.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using conformer_experiment::ConformerEnsemble;
using conformer_experiment::ConformerOptions;

namespace {
struct Options {
    fs::path complex, output, ligand, reference;
    ConformerOptions generation;
    std::optional<std::size_t> prefilter;
    double cutoff = 24.0;
    bool crop = false, sdf = false, original_only = false, generate_only = false;
};

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string quoted(const std::string& value) {
    std::string result = "\"";
    for (char c : value) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}

std::size_t positive(const std::string& value) {
    std::size_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
        throw std::invalid_argument("Expected a positive integer: " + value);
    return result;
}

double number(const std::string& value) {
    std::size_t used{};
    const double result = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(result))
        throw std::invalid_argument("Expected a finite number: " + value);
    return result;
}

void usage(std::ostream& out) {
    out << "Usage: ligand_conformer_docking <complex-directory> <new-output> [options]\n"
        "  --num-conformers N        ETKDGv3 candidates requested (default 10)\n"
        "  --seed N                  Nonnegative reproducible seed (default 20261027)\n"
        "  --min-rmsd A              Aligned heavy-atom diversity threshold (default 0.5)\n"
        "  --mmff                    Minimize with MMFF94 when supported\n"
        "  --max-iterations N        MMFF94 iteration limit (default 500)\n"
        "  --fast-prefilter none|N   Per-conformer rescoring pool (default none)\n"
        "  --crop-margin off|8       Chemical-area crop only (default off)\n"
        "  --cutoff A                Shared triplet-index cutoff (default 24)\n"
        "  --ligand MOL2             Generation input; default <id>_ligand.mol2\n"
        "  --reference MOL2          Crystal reference/baseline; same default\n"
        "  --original-only           Dock only the unchanged crystallographic baseline\n"
        "  --generate-only           Generate conformers without receptor preparation or CUDA\n"
        "  --write-sdf               Export conformers and global Top-20 structures (default CSV only)\n"
        "Quaternion is mandatory. Search uses CUDA; rescoring/global ranking use CPU.\n";
}

Options parse(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("Supply a complex and a new output directory; see --help");
    Options options;
    options.complex = fs::absolute(argv[1]);
    options.output = fs::absolute(argv[2]);
    std::set<std::string> seen;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (!seen.insert(flag).second) throw std::invalid_argument("Duplicate option: " + flag);
        if (flag == "--mmff") { options.generation.optimize_mmff = true; continue; }
        if (flag == "--write-sdf") { options.sdf = true; continue; }
        if (flag == "--original-only") { options.original_only = true; continue; }
        if (flag == "--generate-only") { options.generate_only = true; continue; }
        if (++i == argc) throw std::invalid_argument("Missing option value: " + flag);
        const std::string value = argv[i];
        if (flag == "--num-conformers") options.generation.count = positive(value);
        else if (flag == "--seed") {
            std::size_t seed{};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), seed);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                seed > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Seed must fit a nonnegative int");
            options.generation.seed = static_cast<int>(seed);
        } else if (flag == "--min-rmsd") options.generation.min_rmsd = number(value);
        else if (flag == "--max-iterations") {
            const auto n = positive(value);
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Iteration limit is too large");
            options.generation.max_iterations = static_cast<int>(n);
        } else if (flag == "--fast-prefilter") {
            if (value != "none") options.prefilter = positive(value);
        } else if (flag == "--crop-margin") {
            if (value != "off" && value != "8") throw std::invalid_argument("Crop must be off or 8");
            options.crop = value == "8";
        } else if (flag == "--cutoff") options.cutoff = number(value);
        else if (flag == "--ligand") options.ligand = fs::absolute(value);
        else if (flag == "--reference") options.reference = fs::absolute(value);
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (options.generation.min_rmsd < 0 || options.cutoff <= 0)
        throw std::invalid_argument("Diversity must be nonnegative and cutoff positive");
#ifndef CONFORMER_WITH_CUDA
    if (!options.generate_only) throw std::invalid_argument("Build with ENABLE_CUDA=ON for docking; this build supports --generate-only");
#endif
    return options;
}

std::unique_ptr<RDKit::RWMol> single_ligand(const fs::path& path) {
    Mol2Reader reader(path);
    std::string block, name;
    if (!reader.next(block, name)) throw std::runtime_error("No MOL2 record in " + path.string());
    auto molecule = parse_ligand_mol2_block(block);
    if (reader.next(block, name)) throw std::runtime_error("Expected one ligand MOL2 record: " + path.string());
    return molecule;
}

std::ofstream output_file(const fs::path& path) {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path);
    output << std::setprecision(17);
    return output;
}

template<class T> void optional_value(std::ostream& out, const std::optional<T>& value) {
    if (value) out << *value;
}

struct Record {
    std::size_t id{}; // 0 is reserved for the separate crystal baseline.
    RDKit::RWMol molecule;
    std::optional<double> energy, diversity;
    ConformerDockingResult docking;
};

void write_conformer_structures(const fs::path& path, const std::vector<Record>& records) {
    auto output = output_file(path);
    RDKit::SDWriter writer(&output, false);
    writer.setProps({"conformer_id", "population", "mmff_energy_kcal_mol"});
    for (const auto& row : records) {
        RDKit::RWMol molecule(row.molecule);
        molecule.setProp("_Name", row.id ? "conformer-" + std::to_string(row.id) : "crystal-baseline");
        molecule.setProp("conformer_id", std::to_string(row.id));
        molecule.setProp("population", row.id ? std::string("ensemble") : std::string("baseline"));
        if (molecule.hasProp("mmff_energy_kcal_mol")) molecule.clearProp("mmff_energy_kcal_mol");
        if (row.energy) molecule.setProp("mmff_energy_kcal_mol", *row.energy);
        writer.write(molecule);
    }
    writer.flush();
}

void write_conformers(const fs::path& path, const std::vector<Record>& records, bool generated_only) {
    auto out = output_file(path);
    out << "population,conformer_id,etkdg_candidate_id,mmff_energy_kcal_mol,min_aligned_heavy_rmsd_A,status,diameter_A,"
        "ligand_triplets,candidate_matches,alignments,valid_transforms,survivors,rescored_poses,"
        "oracle_all_survivors_A,oracle_after_prefilter_A,ligand_preparation_s,search_completed_s,"
        "query_preparation_s,rescoring_s,fast_prefilter_s,weighted_interaction_s,vinardo_s,rmsd_evaluation_s\n";
    for (const auto& row : records) {
        out << (row.id ? "ensemble" : "baseline") << ',' << row.id << ',';
        if (row.id) out << row.id - 1;
        out << ',';
        optional_value(out, row.energy); out << ','; optional_value(out, row.diversity);
        if (generated_only) {
            out << ',' << (row.id ? "generated" : "crystal_original");
            for (int i = 0; i < 17; ++i) out << ',';
            out << '\n'; continue;
        }
        const auto& d = row.docking;
        out << ',' << (d.survivors ? "ok" : "no_poses") << ',' << d.diameter << ',' << d.triplets << ','
            << d.matches << ',' << d.alignments << ',' << d.valid_transforms << ',' << d.survivors << ','
            << d.rescored_ids.size() << ',';
        optional_value(out, d.oracle_all_survivors); out << ',';
        optional_value(out, d.oracle_after_prefilter);
        out << ',' << d.preparation_s << ',' << d.search_s << ',' << d.query_s << ',' << d.rescoring_s
            << ',' << d.prefilter_s << ',' << d.weighted_interaction_s << ',' << d.vinardo_s << ',' << d.rmsd_s << '\n';
    }
}

#ifdef CONFORMER_WITH_CUDA
struct RankingResult {
    std::string population;
    PoseScoreMethod method;
    std::vector<EnsemblePose> pool;
    std::vector<RankedEnsemblePose> ranked;
    EnsembleRankingSummary summary;
    std::optional<double> oracle_all_survivors, oracle_after_prefilter;
    double ranking_s{}, hybrid_s{};
};

std::vector<RankingResult> rank_populations(const std::vector<Record>& records) {
    std::vector<RankingResult> results;
    for (bool baseline : {true, false}) {
        std::vector<EnsemblePose> all, selected;
        for (const auto& record : records) if ((record.id == 0) == baseline) {
            all.insert(all.end(), record.docking.poses.begin(), record.docking.poses.end());
            for (auto id : record.docking.rescored_ids) selected.push_back(record.docking.poses.at(id));
        }
        const auto order = [](const auto& a, const auto& b) {
            return std::tie(a.conformer_id, a.pose_id) < std::tie(b.conformer_id, b.pose_id);
        };
        std::sort(all.begin(), all.end(), order);
        std::sort(selected.begin(), selected.end(), order);
        const auto oracle = [](const auto& pool) -> std::optional<double> {
            std::optional<double> best;
            for (const auto& pose : pool) if (!best || pose.rmsd < *best) best = pose.rmsd;
            return best;
        };
        for (auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                            PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid}) {
            RankingResult result;
            result.population = baseline ? "baseline" : "ensemble";
            result.method = method;
            result.pool = method == PoseScoreMethod::Fast ? all : selected;
            result.oracle_all_survivors = oracle(all);
            result.oracle_after_prefilter = oracle(selected);
            const auto start = Clock::now();
            result.ranked = rank_ensemble_poses(result.pool, method, &result.hybrid_s);
            result.ranking_s = seconds(start);
            result.summary = summarize_ensemble_ranking(result.pool, result.ranked);
            results.push_back(std::move(result));
        }
    }
    return results;
}

void write_rankings(const fs::path& directory, const std::vector<Record>& records,
                    const std::vector<RankingResult>& rankings, bool structures) {
    auto out = output_file(directory / "poses.csv");
    out << "population,method,conformer_id,pose_id,alignment_index,local_rank,global_rank,ranking_score,fast_score,"
        "weighted_interaction_score,vinardo_rep50,hybrid_score,rmsd_heavy_A,success_lt_2A,"
        "clash_fraction,occupied_atoms,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
    std::map<std::size_t, const Record*> by_id;
    for (const auto& row : records) by_id.emplace(row.id, &row);
    for (const auto& result : rankings) {
        for (const auto& rank : result.ranked) {
            const auto& row = result.pool.at(rank.source_index);
            const auto& pose = row.scored.pose;
            const auto& selected = by_id.at(row.conformer_id)->docking.rescored_ids;
            const bool scored = std::binary_search(selected.begin(), selected.end(), row.pose_id);
            out << result.population << ',' << pose_score_method_name(result.method) << ','
                << row.conformer_id << ',' << row.pose_id << ',' << row.alignment_index << ',' << rank.local_rank << ',' << rank.global_rank
                << ',' << rank.score << ',' << pose.fast_weighted_score << ',';
            if (scored) out << pose.weighted_interaction_score.weighted_score;
            out << ',';
            if (scored) out << vinardo_adjusted_score(row.scored.vinardo, 0.5);
            out << ','; optional_value(out, rank.hybrid);
            out << ',' << row.rmsd << ',' << (row.rmsd < 2.0) << ',' << pose.clash_fraction << ',' << pose.occupied_atoms;
            for (double value : pose.transform.rotation) out << ',' << value;
            for (double value : pose.transform.translation) out << ',' << value;
            out << '\n';
        }
        if (!structures) continue;
        std::map<std::size_t, std::vector<const RankedEnsemblePose*>> groups;
        for (std::size_t i = 0; i < std::min<std::size_t>(20, result.ranked.size()); ++i)
            groups[result.pool.at(result.ranked[i].source_index).conformer_id].push_back(&result.ranked[i]);
        fs::create_directories(directory / "structures");
        for (const auto& [id, group] : groups) {
            std::vector<GpuPipelinePose> poses;
            std::vector<GpuDockingPoseScores> scores;
            for (const auto* rank : group) {
                const auto& row = result.pool.at(rank->source_index);
                GpuPipelinePose pose{};
                pose.alignment_index = row.alignment_index;
                std::copy(row.scored.pose.transform.rotation.begin(), row.scored.pose.transform.rotation.end(), pose.transform.rotation);
                std::copy(row.scored.pose.transform.translation.begin(), row.scored.pose.transform.translation.end(), pose.transform.translation);
                pose.transform.valid = pose.clash.valid = pose.fast_score.valid = 1;
                pose.clash.fraction = row.scored.pose.clash_fraction;
                pose.clash.occupied_atoms = row.scored.pose.occupied_atoms;
                pose.clash.total_atoms = by_id.at(id)->molecule.getNumAtoms();
                pose.fast_score.weighted_score = row.scored.pose.fast_weighted_score;
                poses.push_back(pose);
                GpuDockingPoseScores score;
                score.pose_id = row.pose_id;
                if (result.method != PoseScoreMethod::Fast) {
                    score.weighted_interaction = row.scored.pose.weighted_interaction_score;
                    score.vinardo = row.scored.vinardo;
                    score.vinardo_adjusted = vinardo_adjusted_score(row.scored.vinardo, 0.5);
                    score.hybrid = rank->hybrid;
                }
                scores.push_back(score);
            }
            auto sdf = output_file(directory / "structures" / (result.population + "-" +
                pose_score_method_name(result.method) + "-conformer-" + std::to_string(id) + ".sdf"));
            std::ostringstream discarded_csv;
            write_gpu_docking_poses(discarded_csv, sdf, by_id.at(id)->molecule, poses, id,
                                    "conformer-" + std::to_string(id), false, nullptr, &scores);
        }
    }
}

void write_summary(const fs::path& path, const std::vector<RankingResult>& rankings,
                   const ConformerEnsemble& ensemble, const std::vector<Record>& records,
                   double generation_s, double wall) {
    auto out = output_file(path);
    out << "population,method,status,requested_conformers,embedded_conformers,distinct_conformers,pool_size,top_k,available_top_k,"
        "top1_conformer_id,top1_pose_id,top1_rmsd_A,best_top_k_rmsd_A,success_lt_2A,oracle_rank_pool_A,"
        "oracle_all_survivors_A,oracle_after_prefilter_A,ranking_s,hybrid_normalization_s,conformer_generation_s,"
        "ligand_preparation_s,search_completed_s,query_preparation_s,rescoring_s,weighted_interaction_s,"
        "vinardo_s,rmsd_evaluation_s,wall_before_summary_s\n";
    for (const auto& result : rankings) for (std::size_t i = 0; i < ensemble_summary_top_k.size(); ++i) {
        const bool baseline = result.population == "baseline";
        std::array<double, 7> times{};
        for (const auto& record : records) if ((record.id == 0) == baseline) {
            const auto& d = record.docking;
            const std::array<double, 7> components{d.preparation_s, d.search_s, d.query_s,
                d.rescoring_s, d.weighted_interaction_s, d.vinardo_s, d.rmsd_s};
            for (std::size_t t = 0; t < times.size(); ++t) times[t] += components[t];
        }
        out << result.population << ',' << pose_score_method_name(result.method) << ','
            << (result.pool.empty() ? "no_poses" : "ok") << ','
            << (baseline ? 1 : ensemble.requested) << ',' << (baseline ? 0 : ensemble.embedded) << ','
            << (baseline ? 1 : ensemble.distinct) << ','
            << result.pool.size() << ',' << ensemble_summary_top_k[i] << ','
            << std::min(ensemble_summary_top_k[i], result.pool.size()) << ',';
        if (!result.ranked.empty()) {
            const auto& top = result.pool.at(result.ranked.front().source_index);
            out << top.conformer_id << ',' << top.pose_id << ',' << top.rmsd;
        } else out << ",,";
        out << ','; optional_value(out, result.summary.top_k_best_rmsd[i]);
        out << ',';
        if (result.summary.top_k_best_rmsd[i]) out << (*result.summary.top_k_best_rmsd[i] < 2.0);
        out << ','; optional_value(out, result.summary.oracle_rmsd);
        out << ','; optional_value(out, result.oracle_all_survivors);
        out << ','; optional_value(out, result.oracle_after_prefilter);
        out << ',' << result.ranking_s << ',' << result.hybrid_s << ',' << (baseline ? 0.0 : generation_s);
        for (double time : times) out << ',' << time;
        out << ',' << wall << '\n';
    }
}
#endif
} // namespace

int main(int argc, char** argv) {
    fs::path created;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(std::cout); return 0; }
        const auto options = parse(argc, argv);
        if (fs::exists(options.output)) throw std::runtime_error("Output directory already exists");
        std::string complex = options.complex.string();
        char* config_args[]{argv[0], complex.data()};
        auto config = parse_run_config(2, config_args);
        if (!options.ligand.empty()) config.ligand_path = options.ligand;
        const auto reference_path = options.reference.empty() ?
            options.complex / (config.complex_id + "_ligand.mol2") : options.reference;
        const auto wall = Clock::now();
        auto start = Clock::now();
        const auto input = single_ligand(config.ligand_path);
        const auto reference = single_ligand(reference_path);
        validate_gpu_docking_reference(*input, *reference);
        const double parsing_s = seconds(start);
        if (!fs::create_directories(options.output)) throw std::runtime_error("Cannot create output directory");
        created = options.output;
        auto protocol = output_file(created / "protocol.csv");
        protocol << "key,value\n";
        const auto setting = [&](const char* key, const auto& value) { protocol << key << ',' << value << '\n'; };
        setting("rdkit_version", RDKit::rdkitVersion);
        setting("ligand", quoted(config.ligand_path.string()));
        setting("reference", quoted(reference_path.string()));
        setting("protein", quoted(config.protein_path.string()));
        setting("pocket", quoted(config.pocket_path.string()));
        setting("seed", options.generation.seed); setting("min_aligned_heavy_rmsd_A", options.generation.min_rmsd);
        setting("mmff_requested", options.generation.optimize_mmff);
        setting("mmff_max_iterations", options.generation.max_iterations);
        setting("embedding", "ETKDGv3; one thread; chirality enforced; no internal pruning");
        setting("rmsd", "direct heavy atoms; verified fixed indices; no alignment or symmetry correction; evaluation only");
        setting("diversity_rmsd", "aligned heavy atoms; fixed indices; no reflection or symmetry correction");
        setting("stereochemistry", "input tags and defined input-3D geometry constrained on working copies; original graph retained");
        setting("kabsch", "quaternion"); setting("cutoff_A", options.cutoff);
        setting("grid_voxel_A", config.grid_voxel_size); setting("grid_margin_A", config.grid_margin);
        setting("triplet_bin_A", config.triplet_bin_size);
        setting("pocket_area_distance_A", config.pocket_centroid_distance);
        setting("clash_fraction", config.operational_clash_threshold); setting("target_clearance_pruning", "off");
        setting("index_policy", "shared-only; prepared once for baseline and all conformers");
        setting("crop_margin_A", options.crop ? "8" : "off");
        setting("fast_prefilter_per_conformer", options.prefilter ? std::to_string(*options.prefilter) : "none");
        setting("search_backend", options.generate_only ? "not_run" : "cuda");
        setting("rescoring_backend", options.generate_only ? "not_run" : "cpu");
        setting("vinardo_alpha", 0.5); setting("hybrid_lambda", 0.3);
        setting("hybrid_pool", "global generated-conformer rescoring pool; baseline normalized separately");
        setting("tie_break", "conformer_id then original survivor pose_id ascending; no RMSD or MMFF energy");
        setting("oracle_scope", "clash survivors and explicit prefilter subset; rejected alignments not evaluated");
        setting("score_regions", "WeightedInteraction pocket areas; Vinardo full protein; crop affects chemical areas only");
        setting("score_directions", "Fast/WeightedInteraction/Hybrid maximize; Vinardo rep50 minimize");
        setting("timing_scope", "search_completed includes query preparation and transfers; scoring subintervals included in rescoring; do not add parents and children");
        setting("sdf_export", options.sdf); setting("parse_s", parsing_s);
        start = Clock::now();
        ConformerEnsemble ensemble;
        if (!options.original_only) ensemble = conformer_experiment::generate_conformers(*input, options.generation);
        const double generation_s = seconds(start);
        setting("conformer_generation_s", generation_s);
        setting("requested_conformers", ensemble.requested); setting("embedded_conformers", ensemble.embedded);
        setting("distinct_conformers", ensemble.distinct); setting("mmff_used", ensemble.mmff_used);
        setting("mmff_support", !options.generation.optimize_mmff || options.original_only ? "not_checked" :
            ensemble.mmff_supported ? "supported" : "unsupported");
        setting("mmff_converged", ensemble.mmff_converged);
        setting("mmff_failed", ensemble.mmff_failed); setting("mmff_fallback", ensemble.mmff_fallback_reason);
        std::vector<Record> records;
        records.push_back({0, *reference, {}, {}, {}});
        for (auto& conformer : ensemble.conformers) {
            validate_gpu_docking_reference(conformer.molecule, *reference);
            records.push_back({conformer.id + 1, std::move(conformer.molecule), conformer.energy,
                               conformer.minimum_aligned_rmsd, {}});
        }
        if (options.sdf) write_conformer_structures(created / "conformers.sdf", records);
        if (options.generate_only) {
            write_conformers(created / "conformers.csv", records, true);
            setting("wall_before_summary_s", seconds(wall)); setting("status", "generation_only");
            std::cout << "Generated " << ensemble.embedded << "; distinct " << ensemble.distinct << '\n';
            return 0;
        }
#ifdef CONFORMER_WITH_CUDA
        if (::setenv("DOCKING_KABSCH_METHOD", "quaternion", 1) ||
            ::setenv("DOCKING_QUERY_METHOD", "reference", 1) ||
            ::setenv("DOCKING_QUERY_THREADS", "1", 1) || ::setenv("DOCKING_PROFILE_STAGES", "0", 1) ||
            ::unsetenv("DOCKING_STAGE_TIMINGS") || ::unsetenv("DOCKING_KABSCH_TIMINGS") ||
            (options.crop ? ::setenv("DOCKING_AREA_CROP_MARGIN_A", "8", 1) : ::unsetenv("DOCKING_AREA_CROP_MARGIN_A")))
            throw std::runtime_error("Cannot set the explicit experiment environment");
        start = Clock::now();
        const PreparedGpuTarget target(config, options.cutoff, std::nullopt, true);
        const RescoringTarget scoring(config, false);
        setting("receptor_preparation_s", seconds(start));
        setting("target_preparation_s", target.target_prep_seconds);
        setting("index_preparation_s", target.index_seconds);
        setting("initial_upload_s", target.gpu_upload_seconds);
        setting("index_records", target.index_records);
        setting("targets", target.protein.targets.size());
        for (auto& record : records)
            record.docking = dock_conformer(record.id, record.molecule, *reference, target, scoring, config, options.prefilter);
        const auto rankings = rank_populations(records);
        write_conformers(created / "conformers.csv", records, false);
        write_rankings(created, records, rankings, options.sdf);
        const double wall_before_summary = seconds(wall);
        write_summary(created / "summary.csv", rankings, ensemble, records, generation_s, wall_before_summary);
        setting("wall_before_summary_s", wall_before_summary); setting("status", "complete");
        std::cout << "Requested " << ensemble.requested << "; embedded " << ensemble.embedded
            << "; distinct " << ensemble.distinct << "; baseline separate; output " << created << '\n';
#endif
        return 0;
    } catch (const std::exception& error) {
        if (!created.empty()) {
            std::ofstream failure(created / "FAILED.txt"); failure << error.what() << '\n';
            std::ofstream protocol(created / "protocol.csv", std::ios::app); protocol << "status,error\n";
        }
        std::cerr << "ligand_conformer_docking: " << error.what() << '\n';
        return 1;
    }
}
