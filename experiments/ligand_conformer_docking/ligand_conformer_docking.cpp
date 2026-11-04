#include "conformer_generation.hpp"
#include "conformer_docking.hpp"
#include "ensemble_diagnostics.hpp"
#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "molecule_io.hpp"
#include "pose_export.hpp"
#include "protein_area_crop.hpp"
#ifdef CONFORMER_WITH_CUDA
#include "cuda_pose_rescoring.hpp"
#endif

#include <RDGeneral/versions.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
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
    ConformerExecutionOptions execution;
    std::optional<std::size_t> prefilter;
    double cutoff = 24.0;
    bool crop = false, sdf = false, original_only = false, generate_only = false;
    PoseExportOptions pose_export{PoseExportMode::All, 20, std::nullopt};
    bool pose_output_explicit = false;
    bool rmsd_oracle = false;
    bool pose_center_distance = false;
    bool diagnostics = false;

    Options() {
        execution.batch_size = 4;
        execution.query_method = "optimized";
    }
};

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
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
        "  --gpu-batch off|on        Aggregate conformer searches (default off)\n"
        "  --batch-size N           Maximum conformers per batch, 1..64 (default 4)\n"
        "  --batch-memory-mib N     Explicit CUDA buffer budget (default 2048; not total VRAM)\n"
        "  --cpu-gpu-overlap off|on Bounded CPU preparation with one CUDA consumer\n"
        "  --query-method METHOD   reference|optimized (default optimized; CLI overrides DOCKING_QUERY_METHOD)\n"
        "  --rescoring-backend NAME cpu|cuda for raw scores (default cpu)\n"
        "  --ranking all|rmsd       Four energy rankings; rmsd adds an ensemble-only oracle diagnostic\n"
        "  --pose-center-distance on|off  Export heavy-atom centroid distance; default off\n"
        "  --ligand MOL2             Generation input; default <id>_ligand.mol2\n"
        "  --reference MOL2          Crystal reference/baseline; same default\n"
        "  --original-only           Dock only the unchanged crystallographic baseline\n"
        "  --generate-only           Generate conformers without receptor preparation or CUDA\n"
        "  --write-sdf               Optional structures; legacy Top-20 unless pose-output is explicit\n"
        "  --pose-output none|topk|all  Pose export only (default all); metrics always use full pools\n"
        "  --top-k N                 Rows per population/method with topk (default 20)\n"
        "  --pose-detail compact|full  Compact omits R/t; default preserves the full CSV\n"
        "  --diagnostics            Aggregate original-included/excluded Full/Fast-500 pools for accepted prefixes 1/5/10/20; no pose files\n"
        "Quaternion is mandatory. Hybrid is normalized globally on the host; the baseline is separate.\n"
        "Diagnostics uses 20 candidates, seed 20261027, diversity 0.5 A, MMFF off and CUDA rescoring; the original participates only in original-included analyses.\n"
        "GPU batching requires --rescoring-backend cuda.\n"
        "Batch, explicit overlap, CUDA rescoring and RMSD oracle require --fast-prefilter none.\n";
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
        if (flag == "--diagnostics") { options.diagnostics = true; continue; }
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
        else if (flag == "--gpu-batch") {
            if (value != "off" && value != "on") throw std::invalid_argument("GPU batch must be off or on");
            options.execution.gpu_batch = value == "on";
        } else if (flag == "--batch-size") {
            options.execution.batch_size = positive(value);
            if (options.execution.batch_size > 64) throw std::invalid_argument("Batch size must be at most 64");
        } else if (flag == "--batch-memory-mib") {
            const auto mib = positive(value);
            constexpr std::size_t bytes_per_mib = 1024 * 1024;
            if (mib > std::numeric_limits<std::size_t>::max() / bytes_per_mib)
                throw std::invalid_argument("Batch memory exceeds addressable capacity");
            options.execution.batch_memory_bytes = mib * bytes_per_mib;
        } else if (flag == "--cpu-gpu-overlap") {
            if (value != "off" && value != "on") throw std::invalid_argument("CPU/GPU overlap must be off or on");
            options.execution.cpu_gpu_overlap = value == "on";
        } else if (flag == "--query-method") {
            if (value != "reference" && value != "optimized") throw std::invalid_argument("Unknown query method: " + value);
            options.execution.query_method = value;
        } else if (flag == "--rescoring-backend") {
            if (value != "cpu" && value != "cuda") throw std::invalid_argument("Rescoring backend must be cpu or cuda");
            options.execution.rescoring_backend = value == "cuda" ? GpuRescoringBackend::Cuda : GpuRescoringBackend::Cpu;
        } else if (flag == "--ranking") {
            if (value != "all" && value != "rmsd") throw std::invalid_argument("Ranking must be all or rmsd");
            options.rmsd_oracle = value == "rmsd";
        } else if (flag == "--pose-center-distance") {
            if (value != "off" && value != "on") throw std::invalid_argument("Pose center distance must be off or on");
            options.pose_center_distance = value == "on";
        } else if (flag == "--ligand") options.ligand = fs::absolute(value);
        else if (flag == "--reference") options.reference = fs::absolute(value);
        else if (flag == "--pose-output") {
            options.pose_export.mode = parse_pose_export_mode(value);
            options.pose_output_explicit = true;
        } else if (flag == "--top-k") options.pose_export.top_k = parse_pose_top_k(value);
        else if (flag == "--pose-detail") options.pose_export.detail = parse_pose_export_detail(value);
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (!seen.contains("--query-method")) {
        if (const char* method = std::getenv("DOCKING_QUERY_METHOD")) {
            if (std::string(method) != "reference" && std::string(method) != "optimized")
                throw std::invalid_argument("DOCKING_QUERY_METHOD must be reference or optimized");
            options.execution.query_method = method;
        }
    }
    if (options.generation.min_rmsd < 0 || options.cutoff <= 0)
        throw std::invalid_argument("Diversity must be nonnegative and cutoff positive");
    if (options.diagnostics) {
        if (!seen.contains("--num-conformers")) options.generation.count = 20;
        if (options.generation.count != 20 || options.generation.seed != 20261027 ||
            options.generation.min_rmsd != 0.5 || options.generation.optimize_mmff)
            throw std::invalid_argument("--diagnostics requires 20 ETKDGv3 candidates, seed 20261027, diversity 0.5 A and MMFF off");
        if (options.generate_only || options.original_only || options.rmsd_oracle || options.prefilter)
            throw std::invalid_argument("--diagnostics requires baseline plus generated conformers and no per-conformer prefilter or RMSD ranking");
        if (options.sdf || (options.pose_output_explicit && options.pose_export.mode != PoseExportMode::None))
            throw std::invalid_argument("--diagnostics does not export poses; omit --write-sdf and use --pose-output none");
        if (options.execution.rescoring_backend != GpuRescoringBackend::Cuda ||
            options.execution.query_method != "optimized")
            throw std::invalid_argument("--diagnostics requires --rescoring-backend cuda and --query-method optimized");
        options.pose_export.mode = PoseExportMode::None;
    }
    if (!options.execution.gpu_batch && (seen.contains("--batch-size") || seen.contains("--batch-memory-mib")))
        throw std::invalid_argument("Batch size and memory controls require --gpu-batch on");
    if (options.execution.gpu_batch && options.execution.rescoring_backend != GpuRescoringBackend::Cuda)
        throw std::invalid_argument("Conformer GPU batching requires --rescoring-backend cuda");
    if (options.prefilter && (options.execution.gpu_batch || options.execution.cpu_gpu_overlap.has_value() ||
        options.execution.rescoring_backend == GpuRescoringBackend::Cuda || options.rmsd_oracle))
        throw std::invalid_argument("Batch, explicit overlap, CUDA rescoring and RMSD oracle require --fast-prefilter none");
#if !defined(CONFORMER_WITH_CUDA) && !defined(CONFORMER_EXPORT_TEST)
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

struct PoseCenterCache {
    explicit PoseCenterCache(const RDKit::ROMol& reference)
        : reference_center(heavy_atom_centroid(reference)) {}

    double distance(const Record& source, const EnsemblePose& pose) {
        const auto key = std::pair{pose.conformer_id, pose.pose_id};
        if (const auto found = distances.find(key); found != distances.end()) return found->second;
        auto center = ligand_centers.find(source.id);
        if (center == ligand_centers.end())
            center = ligand_centers.emplace(source.id, heavy_atom_centroid(source.molecule)).first;
        const auto value = pose_center_distance(center->second, pose.scored.pose.transform, reference_center);
        return distances.emplace(key, value).first->second;
    }

    std::array<double, 3> reference_center;
    std::map<std::size_t, std::array<double, 3>> ligand_centers;
    std::map<std::pair<std::size_t, std::size_t>, double> distances;
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

void write_conformers(const fs::path& path, const std::vector<Record>& records, bool generated_only,
                      bool diagnostics = false) {
    auto out = output_file(path);
    out << "population,conformer_id,etkdg_candidate_id,mmff_energy_kcal_mol,min_aligned_heavy_rmsd_A,status,diameter_A,"
        "ligand_triplets,candidate_matches,alignments,valid_transforms,survivors,rescored_poses,"
        "oracle_all_survivors_A,oracle_after_prefilter_A,ligand_preparation_s,search_completed_s,"
        "query_preparation_s,rescoring_s,fast_prefilter_s,weighted_interaction_s,vinardo_s,rmsd_evaluation_s";
    if (diagnostics) out << ",accepted_order";
    out << '\n';
    std::size_t accepted_order = 0;
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
        out << ',' << d.preparation_s << ',';
        if (!d.shared_search) out << d.search_s;
        out << ',' << d.query_s << ',';
        if (!d.shared_rescoring) out << d.rescoring_s;
        out << ',' << d.prefilter_s << ',';
        if (!d.shared_rescoring) out << d.weighted_interaction_s;
        out << ',';
        if (!d.shared_rescoring) out << d.vinardo_s;
        out << ',' << d.rmsd_s;
        if (diagnostics) {
            out << ',';
            if (row.id) out << ++accepted_order;
        }
        out << '\n';
    }
}

struct DiagnosticProgress {
    std::optional<std::size_t> generated, accepted, not_generated, discarded;
    std::optional<std::size_t> valid, failed, with_survivors, original_survivors, total_survivors;
};

void write_diagnostic_status(const fs::path& directory, const std::string& status,
    const std::string& stage, const std::string& message, const DiagnosticProgress& counts, double wall) {
    auto out = output_file(directory / "status.csv");
    out << "key,value\n";
    const auto write = [&](const std::string& key, const auto& value) {
        std::ostringstream text;
        text << std::setprecision(17) << value;
        write_gpu_docking_summary_entry(out, key, text.str());
    };
    const auto count = [&](const char* key, const std::optional<std::size_t>& value) {
        write(key, value ? std::to_string(*value) : "");
    };
    write("status", status); write("stage", stage); write("message", message);
    write("completed", status == "ok" || status == "no_poses" ? 1 : 0);
    write("requested_conformers", 20);
    count("generated_conformers", counts.generated); count("accepted_conformers", counts.accepted);
    count("not_generated_conformers", counts.not_generated); count("diversity_discarded_conformers", counts.discarded);
    count("valid_conformers", counts.valid); count("failed_conformers", counts.failed);
    count("conformers_with_survivors", counts.with_survivors);
    count("original_survivors", counts.original_survivors); count("total_survivors", counts.total_survivors);
    write("wall_before_status_s", wall);
}

#if defined(CONFORMER_WITH_CUDA) || defined(CONFORMER_EXPORT_TEST)
conformer_diagnostics::Timings write_diagnostic_outputs(const fs::path& directory,
    const std::string& complex_id, const std::vector<Record>& records, const RDKit::ROMol& reference) {
    std::vector<conformer_diagnostics::Input> inputs;
    for (const auto& record : records) inputs.push_back({record.id, &record.molecule, &record.docking});
    auto summary = output_file(directory / "summary.csv");
    auto topk = output_file(directory / "topk_metrics.csv");
    conformer_diagnostics::Timings timings;
    conformer_diagnostics::write_diagnostics(summary, topk, complex_id, inputs, reference, &timings);
    summary.close(); topk.close();
    write_conformers(directory / "conformers.csv", records, false, true);
    return timings;
}

struct RankingResult {
    std::string population;
    PoseScoreMethod method = PoseScoreMethod::Fast;
    bool rmsd_oracle = false;
    std::vector<EnsemblePose> pool;
    std::vector<RankedEnsemblePose> ranked;
    EnsembleRankingSummary summary;
    std::optional<double> oracle_all_survivors, oracle_after_prefilter;
    double ranking_s{}, hybrid_s{};
};

const char* ranking_name(const RankingResult& result) {
    return result.rmsd_oracle ? "rmsd" : pose_score_method_name(result.method);
}

std::vector<RankingResult> rank_populations(const std::vector<Record>& records, bool rmsd_oracle = false) {
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
        if (!baseline && rmsd_oracle) {
            RankingResult result;
            result.population = "ensemble";
            result.rmsd_oracle = true;
            result.pool = std::move(all);
            result.oracle_all_survivors = oracle(result.pool);
            result.oracle_after_prefilter = oracle(selected);
            const auto start = Clock::now();
            result.ranked = rank_ensemble_oracle(result.pool);
            result.ranking_s = seconds(start);
            result.summary = summarize_ensemble_ranking(result.pool, result.ranked);
            results.push_back(std::move(result));
        }
    }
    return results;
}

void write_rankings(const fs::path& directory, const std::vector<Record>& records,
                    const std::vector<RankingResult>& rankings, const Options& options,
                    PoseCenterCache* centers = nullptr) {
    if (options.pose_export.mode == PoseExportMode::None) return;
    if (options.pose_center_distance && !centers)
        throw std::invalid_argument("Pose center distance requires a validated reference centroid");
    const bool full = options.pose_export.detail != PoseExportDetail::Compact;
    auto out = output_file(directory / "poses.csv");
    out << "population,method,conformer_id,pose_id,alignment_index,local_rank,global_rank,ranking_score,fast_score,"
        "weighted_interaction_score,vinardo_rep50,hybrid_score,rmsd_heavy_A,success_lt_2A,"
        "clash_fraction,occupied_atoms";
    if (full) out << ",r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz";
    if (options.pose_center_distance) out << ",center_distance_A";
    out << '\n';
    std::map<std::size_t, const Record*> by_id;
    for (const auto& row : records) by_id.emplace(row.id, &row);
    for (const auto& result : rankings) {
        const auto rows = pose_export_count(options.pose_export, result.ranked.size());
        for (std::size_t i = 0; i < rows; ++i) {
            const auto& rank = result.ranked[i];
            const auto& row = result.pool.at(rank.source_index);
            const auto& pose = row.scored.pose;
            const auto& selected = by_id.at(row.conformer_id)->docking.rescored_ids;
            const bool scored = std::binary_search(selected.begin(), selected.end(), row.pose_id);
            out << result.population << ',' << ranking_name(result) << ','
                << row.conformer_id << ',' << row.pose_id << ',' << row.alignment_index << ',' << rank.local_rank << ',' << rank.global_rank
                << ',' << rank.score << ',' << pose.fast_weighted_score << ',';
            if (scored) out << pose.weighted_interaction_score.weighted_score;
            out << ',';
            if (scored) out << vinardo_adjusted_score(row.scored.vinardo, 0.5);
            out << ','; optional_value(out, rank.hybrid);
            out << ',' << row.rmsd << ',' << (row.rmsd < 2.0) << ',' << pose.clash_fraction << ',' << pose.occupied_atoms;
            if (full) {
                for (double value : pose.transform.rotation) out << ',' << value;
                for (double value : pose.transform.translation) out << ',' << value;
            }
            if (options.pose_center_distance)
                out << ',' << centers->distance(*by_id.at(row.conformer_id), row);
            out << '\n';
        }
        if (!options.sdf) continue;
        std::map<std::size_t, std::vector<const RankedEnsemblePose*>> groups;
        // Keep historical SDF Top-20 unless an output policy was explicitly chosen.
        const auto structures = options.pose_output_explicit ? rows :
            std::min<std::size_t>(20, result.ranked.size());
        for (std::size_t i = 0; i < structures; ++i)
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
                if (result.rmsd_oracle || result.method != PoseScoreMethod::Fast) {
                    score.weighted_interaction = row.scored.pose.weighted_interaction_score;
                    score.vinardo = row.scored.vinardo;
                    score.vinardo_adjusted = vinardo_adjusted_score(row.scored.vinardo, 0.5);
                    score.hybrid = rank->hybrid;
                }
                scores.push_back(score);
            }
            auto sdf = output_file(directory / "structures" / (result.population + "-" +
                ranking_name(result) + "-conformer-" + std::to_string(id) + ".sdf"));
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
        "vinardo_s,rmsd_evaluation_s,wall_before_summary_s,top1_minus_oracle_all_A\n";
    for (const auto& result : rankings) for (std::size_t i = 0; i < ensemble_summary_top_k.size(); ++i) {
        const bool baseline = result.population == "baseline";
        bool shared_search = false, shared_rescoring = false;
        std::array<double, 7> times{};
        for (const auto& record : records) if ((record.id == 0) == baseline) {
            const auto& d = record.docking;
            shared_search = shared_search || d.shared_search;
            shared_rescoring = shared_rescoring || d.shared_rescoring;
            const std::array<double, 7> components{d.preparation_s, d.search_s, d.query_s,
                d.rescoring_s, d.weighted_interaction_s, d.vinardo_s, d.rmsd_s};
            for (std::size_t t = 0; t < times.size(); ++t) times[t] += components[t];
        }
        out << result.population << ',' << ranking_name(result) << ','
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
        for (std::size_t t = 0; t < times.size(); ++t) {
            out << ',';
            if (!(shared_search && t == 1) && !(shared_rescoring && (t == 3 || t == 4 || t == 5)))
                out << times[t];
        }
        out << ',' << wall << ',';
        if (!result.ranked.empty() && result.oracle_all_survivors)
            out << result.pool.at(result.ranked.front().source_index).rmsd - *result.oracle_all_survivors;
        out << '\n';
    }
}
#endif
} // namespace

int main(int argc, char** argv) {
    fs::path created;
    bool diagnostics = false;
    std::string stage = "configuration";
    DiagnosticProgress counts;
    Clock::time_point diagnostic_start;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(std::cout); return 0; }
        const auto options = parse(argc, argv);
        diagnostics = options.diagnostics;
        if (diagnostics) diagnostic_start = Clock::now();
        const bool shared_execution = options.execution.gpu_batch || options.execution.cpu_gpu_overlap.has_value() ||
            options.execution.rescoring_backend == GpuRescoringBackend::Cuda;
        if (fs::exists(options.output)) throw std::runtime_error("Output directory already exists");
        std::string complex = options.complex.string();
        char* config_args[]{argv[0], complex.data()};
        auto config = parse_run_config(2, config_args);
        if (!options.ligand.empty()) config.ligand_path = options.ligand;
        const auto reference_path = options.reference.empty() ?
            options.complex / (config.complex_id + "_ligand.mol2") : options.reference;
        if (diagnostics) {
            if (!fs::create_directories(options.output)) throw std::runtime_error("Cannot create output directory");
            created = options.output;
            write_diagnostic_status(created, "running", "input_loading", "", counts, seconds(diagnostic_start));
        }
        const auto wall = Clock::now();
        auto start = Clock::now();
        stage = "ligand_loading";
        const auto input = single_ligand(config.ligand_path);
        stage = "reference_loading";
        const auto reference = single_ligand(reference_path);
        stage = "reference_validation";
        validate_gpu_docking_reference(*input, *reference);
        std::optional<PoseCenterCache> pose_centers;
        if (options.pose_center_distance) pose_centers.emplace(*reference);
        const double parsing_s = seconds(start);
        if (!diagnostics) {
            if (!fs::create_directories(options.output)) throw std::runtime_error("Cannot create output directory");
            created = options.output;
        }
        auto protocol = output_file(created / "protocol.csv");
        protocol << "key,value\n";
        const auto setting = [&](const std::string& key, const auto& value) {
            std::ostringstream text;
            text << std::setprecision(17) << value;
            write_gpu_docking_summary_entry(protocol, key, text.str());
        };
        setting("rdkit_version", RDKit::rdkitVersion);
        setting("ligand", config.ligand_path.string());
        setting("reference", reference_path.string());
        setting("protein", config.protein_path.string());
        setting("pocket", config.pocket_path.string());
        setting("seed", options.generation.seed); setting("min_aligned_heavy_rmsd_A", options.generation.min_rmsd);
        setting("mmff_requested", options.generation.optimize_mmff);
        setting("mmff_max_iterations", options.generation.max_iterations);
        setting("embedding", "ETKDGv3; one thread; chirality enforced; no internal pruning");
        setting("rmsd", "direct heavy atoms; verified fixed indices; no alignment or symmetry correction; excluded from energy rankings");
        if (options.pose_center_distance || diagnostics) {
            setting("pose_center_distance", "on");
            setting("center_distance_definition", diagnostics
                ? "direct heavy-atom centroids after R*c+t; angstroms; evaluated once for every survivor in memory; excluded from ranking"
                : "distance between direct heavy-atom geometric centroids after R*c+t; angstroms; export diagnostic only");
        }
        setting("ranking_requested", diagnostics ? "diagnostics" : options.rmsd_oracle ? "rmsd" : "all");
        setting("rmsd_oracle_diagnostic", options.rmsd_oracle);
        if (options.rmsd_oracle) {
            setting("rmsd_oracle_pool", "all generated-conformer clash survivors; crystallographic baseline excluded");
            setting("rmsd_oracle_ranking", "minimum reference RMSD; diagnostic upper bound requiring the crystal reference; not predictive scoring");
        }
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
        setting("rescoring_backend", options.generate_only ? "not_run" :
            options.execution.rescoring_backend == GpuRescoringBackend::Cuda ? "cuda" : "cpu");
        setting("gpu_batch", options.execution.gpu_batch ? "on" : "off");
        if (options.execution.gpu_batch) {
            setting("batch_size", options.execution.batch_size);
            setting("batch_memory_budget_bytes", options.execution.batch_memory_bytes);
        }
        setting("cpu_gpu_overlap", !options.execution.cpu_gpu_overlap.has_value() ? "legacy" :
            *options.execution.cpu_gpu_overlap ? "on" : "off");
        setting("query_method", options.execution.query_method);
        setting("execution_pipeline", options.generate_only ? "not_run" :
            shared_execution ? "shared prepared CUDA workflow" : "legacy CPU rescoring workflow");
        if (diagnostics) {
            setting("vinardo_standard_alpha", 1); setting("vinardo_rep50_alpha", 0.5); setting("hybrid_alpha", 0.5);
        } else setting("vinardo_alpha", 0.5);
        setting("hybrid_lambda", 0.3);
        setting("hybrid_pool", diagnostics
            ? "independent normalization for each accepted-prefix budget, original-inclusion condition and full/global-Fast-500 pool"
            : "global generated-conformer rescoring pool; baseline normalized separately");
        setting("tie_break", "conformer_id then original survivor pose_id ascending; no RMSD or MMFF energy");
        setting("oracle_scope", "clash survivors and explicit prefilter subset; rejected alignments not evaluated");
        setting("score_regions", "WeightedInteraction pocket areas; Vinardo full protein; crop affects chemical areas only");
        setting("score_directions", diagnostics
            ? "Fast/WeightedInteraction/Hybrid maximize; Vinardo standard and rep50 minimize"
            : "Fast/WeightedInteraction/Hybrid maximize; Vinardo rep50 minimize");
        if (diagnostics) {
            setting("diagnostics", "on"); setting("accepted_prefix_budgets", "1;5;10;20");
            setting("budget_definition", "first min(B,accepted) conformers in the one retained ETKDG sequence; no regeneration to fill shortages");
            setting("conditions", "original_included=0 or 1; full or fast_top500 selected globally within each condition");
            setting("original_geometry", "docked crystallographic reference (conformer_id 0); no untransformed reference pose is inserted");
            setting("native_information", "original geometry is known; ETKDG working stereo constraints may be inferred from input crystal coordinates before coordinates are cleared");
            setting("global_pose_id", "zero-based ordinal of (conformer_id,pose_id) across all survivors of this execution; unchanged across budgets and pools");
            setting("fast_subset_order", "Fast descending then global_pose_id ascending; at most 500 for each full condition");
            setting("metric_ties", "Oracle minima use global_pose_id; cumulative minima retain earliest ranking position");
            setting("success_thresholds", "strict RMSD < 2 A; RMSD < 3 A; centroid distance < 5 A");
            setting("timing_budget_scope", "one maximum-budget execution; prefix statistics are not separate measured timing runs");
            setting("failure_policy", "abort complex on generation/preparation/device errors; unknown failure counts stay empty, never silently drop conformers");
        }
        setting("timing_scope", options.execution.gpu_batch
            ? "shared batch search/scoring are in execution timings; per-conformer shared intervals are blank; do not add nested clocks"
            : "search_completed includes query preparation and transfers; scoring subintervals included in rescoring; do not add parents and children");
        setting("pose_output", pose_export_mode_name(options.pose_export.mode));
        setting("pose_top_k", options.pose_export.top_k);
        setting("pose_detail", options.pose_export.detail == PoseExportDetail::Compact ? "compact" : "full");
        setting("pose_output_scope", "serialization only; full ranking pools and RMSD metrics unchanged");
        setting("sdf_requested", options.sdf);
        setting("sdf_export", options.sdf && options.pose_export.mode != PoseExportMode::None);
        setting("sdf_pose_policy", options.pose_output_explicit ? "same as pose_output" : "historical top-20 per population/method");
        setting("parse_s", parsing_s);
        stage = "conformer_generation";
        start = Clock::now();
        ConformerEnsemble ensemble;
        if (!options.original_only) ensemble = conformer_experiment::generate_conformers(*input, options.generation);
        const double generation_s = seconds(start);
        if (diagnostics) {
            counts.generated = ensemble.embedded; counts.accepted = ensemble.distinct;
            counts.not_generated = ensemble.requested - ensemble.embedded;
            counts.discarded = ensemble.embedded - ensemble.distinct;
            setting("not_generated_conformers", *counts.not_generated);
            setting("diversity_discarded_conformers", *counts.discarded);
        }
        setting("conformer_generation_s", generation_s);
        setting("requested_conformers", ensemble.requested); setting("embedded_conformers", ensemble.embedded);
        setting("distinct_conformers", ensemble.distinct); setting("mmff_used", ensemble.mmff_used);
        setting("mmff_support", !options.generation.optimize_mmff || options.original_only ? "not_checked" :
            ensemble.mmff_supported ? "supported" : "unsupported");
        setting("mmff_converged", ensemble.mmff_converged);
        setting("mmff_failed", ensemble.mmff_failed); setting("mmff_fallback", ensemble.mmff_fallback_reason);
        std::vector<Record> records;
        records.push_back({0, *reference, {}, {}, {}});
        stage = "conformer_reference_validation";
        for (auto& conformer : ensemble.conformers) {
            validate_gpu_docking_reference(conformer.molecule, *reference);
            records.push_back({conformer.id + 1, std::move(conformer.molecule), conformer.energy,
                               conformer.minimum_aligned_rmsd, {}});
        }
        const auto structures_start = Clock::now();
        if (options.sdf && options.pose_export.mode != PoseExportMode::None)
            write_conformer_structures(created / "conformers.sdf", records);
        const double conformer_export_s = seconds(structures_start);
        if (options.generate_only) {
            write_conformers(created / "conformers.csv", records, true);
            setting("conformer_structure_export_s", conformer_export_s);
            setting("wall_before_summary_s", seconds(wall)); setting("status", "generation_only");
            std::cout << "Generated " << ensemble.embedded << "; distinct " << ensemble.distinct << '\n';
            return 0;
        }
#ifdef CONFORMER_WITH_CUDA
        stage = "receptor_preparation";
        if (::setenv("DOCKING_KABSCH_METHOD", "quaternion", 1) ||
            ::setenv("DOCKING_QUERY_METHOD", options.execution.query_method.c_str(), 1) ||
            ((!shared_execution || !std::getenv("DOCKING_QUERY_THREADS")) && ::setenv("DOCKING_QUERY_THREADS", "1", 1)) ||
            ::setenv("DOCKING_PROFILE_STAGES", "0", 1) ||
            ::unsetenv("DOCKING_STAGE_TIMINGS") || ::unsetenv("DOCKING_KABSCH_TIMINGS") ||
            (options.crop ? ::setenv("DOCKING_AREA_CROP_MARGIN_A", "8", 1) : ::unsetenv("DOCKING_AREA_CROP_MARGIN_A")))
            throw std::runtime_error("Cannot set the explicit experiment environment");
        setting("query_threads_requested", gpu_query_requested_threads());
        setting("query_thread_source", shared_execution ? "DOCKING_QUERY_THREADS or default 1" : "legacy fixed 1");
        start = Clock::now();
        const PreparedGpuTarget target(config, options.cutoff, std::nullopt, true);
        const RescoringTarget scoring(config, false);
        const auto scoring_upload_start = Clock::now();
        std::unique_ptr<CudaRescoringTarget> device_scoring;
        if (shared_execution && options.execution.rescoring_backend == GpuRescoringBackend::Cuda)
            device_scoring = std::make_unique<CudaRescoringTarget>(scoring, PoseRescoringMethods{true, true});
        setting("rescoring_receptor_upload_s", device_scoring ? seconds(scoring_upload_start) : 0.0);
        setting("receptor_preparation_s", seconds(start));
        setting("target_preparation_s", target.target_prep_seconds);
        setting("index_preparation_s", target.index_seconds);
        setting("initial_upload_s", target.gpu_upload_seconds);
        setting("index_records", target.index_records);
        setting("targets", target.protein.targets.size());
        if (shared_execution) {
            const auto dock_population = [&](std::size_t first, std::size_t last, const std::string& population) {
                if (first == last) return;
                stage = population + "_docking";
                std::vector<ConformerDockingInput> inputs;
                inputs.reserve(last - first);
                for (auto i = first; i < last; ++i) inputs.push_back({records[i].id, &records[i].molecule});
                ConformerExecutionTimings execution;
                auto results = dock_conformers(inputs, *reference, target, scoring, config, options.prefilter,
                    options.execution, &execution, device_scoring.get());
                if (results.size() != inputs.size()) throw std::runtime_error("Conformer execution lost input records");
                for (auto i = first; i < last; ++i) records[i].docking = std::move(results[i - first]);
                const auto prefix = population + "_execution_";
                setting(prefix + "wall_s", execution.wall_s);
                setting(prefix + "preparation_s", execution.preparation_s);
                setting(prefix + "query_s", execution.query_s);
                setting(prefix + "search_s", execution.search_s);
                setting(prefix + "rescoring_s", execution.rescoring_s);
                setting(prefix + "consumer_wait_s", execution.consumer_wait_s);
                setting(prefix + "producer_wait_s", execution.producer_wait_s);
                setting(prefix + "adaptive_splits", execution.adaptive_splits);
                setting(prefix + "memory_retries", execution.memory_retries);
                setting(prefix + "queue_capacity", execution.queue_capacity);
                setting(prefix + "queue_high_water_records", execution.queue_high_water_records);
                setting(prefix + "queue_high_water_bytes", execution.queue_high_water_bytes);
            };
            dock_population(0, 1, "baseline");
            dock_population(1, records.size(), "ensemble");
            setting("execution_timing_scope", "baseline and ensemble measured separately; one shared geometric target/index; host-completed wrappers; not isolated kernel times");
        } else {
            for (auto& record : records)
                record.docking = dock_conformer(record.id, record.molecule, *reference, target, scoring, config, options.prefilter);
        }
        if (diagnostics) {
            stage = "diagnostic_analysis";
            counts.valid = records.size() - 1; counts.failed = 0;
            counts.original_survivors = records.front().docking.survivors;
            counts.with_survivors = 0; counts.total_survivors = 0;
            for (const auto& record : records) {
                *counts.total_survivors += record.docking.survivors;
                if (record.id && record.docking.survivors) ++*counts.with_survivors;
            }
            const auto timings = write_diagnostic_outputs(created, config.complex_id, records, *reference);
            setting("valid_conformers", *counts.valid); setting("failed_conformers", *counts.failed);
            setting("conformers_with_survivors", *counts.with_survivors);
            setting("original_survivors", *counts.original_survivors); setting("total_survivors", *counts.total_survivors);
            setting("diagnostic_metrics_s", timings.metrics_s); setting("diagnostic_analysis_export_s", timings.analysis_s);
            setting("exported_pose_rows", 0); setting("pose_csv_bytes", 0);
            setting("conformer_sdf_bytes", 0); setting("pose_sdf_bytes", 0);
            setting("wall_before_summary_s", seconds(wall)); setting("status", "complete");
            protocol.close();
            write_diagnostic_status(created, *counts.total_survivors ? "ok" : "no_poses",
                "complete", "", counts, seconds(diagnostic_start));
            std::cout << "Requested " << ensemble.requested << "; generated " << ensemble.embedded
                      << "; accepted " << ensemble.distinct << "; diagnostic prefixes 1/5/10/20; output " << created << '\n';
            return 0;
        }
        const auto rankings = rank_populations(records, options.rmsd_oracle);
        write_conformers(created / "conformers.csv", records, false);
        const auto pose_export_start = Clock::now();
        write_rankings(created, records, rankings, options, pose_centers ? &*pose_centers : nullptr);
        setting("pose_export_s", seconds(pose_export_start));
        setting("conformer_structure_export_s", conformer_export_s);
        std::size_t exported_rows = 0;
        for (const auto& result : rankings)
            exported_rows += pose_export_count(options.pose_export, result.ranked.size());
        setting("exported_pose_rows", exported_rows);
        setting("pose_csv_bytes", fs::exists(created / "poses.csv") ? fs::file_size(created / "poses.csv") : 0);
        setting("conformer_sdf_bytes", fs::exists(created / "conformers.sdf") ? fs::file_size(created / "conformers.sdf") : 0);
        std::uintmax_t pose_sdf_bytes = 0;
        if (fs::exists(created / "structures"))
            for (const auto& entry : fs::directory_iterator(created / "structures"))
                if (entry.is_regular_file()) pose_sdf_bytes += entry.file_size();
        setting("pose_sdf_bytes", pose_sdf_bytes);
        const double wall_before_summary = seconds(wall);
        write_summary(created / "summary.csv", rankings, ensemble, records, generation_s, wall_before_summary);
        setting("wall_before_summary_s", wall_before_summary); setting("status", "complete");
        std::cout << "Requested " << ensemble.requested << "; embedded " << ensemble.embedded
            << "; distinct " << ensemble.distinct << "; baseline separate; output " << created << '\n';
#endif
        return 0;
    } catch (const std::exception& error) {
        if (!created.empty()) {
            if (diagnostics) {
                try {
                    const auto status = stage == "reference_loading" || stage == "reference_validation" ||
                        stage == "conformer_reference_validation" ? "invalid_reference" : "error";
                    write_diagnostic_status(created, status, stage, error.what(), counts, seconds(diagnostic_start));
                    const bool header = !fs::exists(created / "protocol.csv");
                    std::ofstream protocol(created / "protocol.csv", std::ios::app);
                    protocol.exceptions(std::ios::badbit | std::ios::failbit);
                    if (header) protocol << "key,value\n";
                    write_gpu_docking_summary_entry(protocol, "status", status);
                    write_gpu_docking_summary_entry(protocol, "failure_stage", stage);
                    write_gpu_docking_summary_entry(protocol, "failure_message", error.what());
                } catch (const std::exception& output_error) {
                    std::cerr << "Cannot record diagnostic failure: " << output_error.what() << '\n';
                }
            } else {
                std::ofstream failure(created / "FAILED.txt"); failure << error.what() << '\n';
                std::ofstream protocol(created / "protocol.csv", std::ios::app); protocol << "status,error\n";
            }
        }
        std::cerr << "ligand_conformer_docking: " << error.what() << '\n';
        return 1;
    }
}
