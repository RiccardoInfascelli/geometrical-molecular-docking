#include "ablation.hpp"

#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "interaction_areas.hpp"
#include "molecule_io.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace scoring_ablation;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    fs::path complex, ligand, poses, output, reference;
    std::string pool_kind, interaction_region;
    std::optional<std::size_t> pool_size;
    std::size_t record = 1;
    double alpha = 1.0, lambda = 0.3, tolerance = 1e-6;
};

std::size_t integer(const std::string& value)
{
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("Expected a non-negative integer: " + value);
    return result;
}

double number(const std::string& value)
{
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result))
        throw std::invalid_argument("Expected a finite number: " + value);
    return result;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--complex") options.complex = value;
        else if (flag == "--ligand") options.ligand = value;
        else if (flag == "--poses") options.poses = value;
        else if (flag == "--output") options.output = value;
        else if (flag == "--reference") options.reference = value;
        else if (flag == "--record") options.record = integer(value);
        else if (flag == "--pool-size") options.pool_size = integer(value);
        else if (flag == "--pool-kind") options.pool_kind = value;
        else if (flag == "--interaction-region") options.interaction_region = value;
        else if (flag == "--vinardo-alpha") options.alpha = number(value);
        else if (flag == "--hybrid-lambda") options.lambda = number(value);
        else if (flag == "--score-tolerance") options.tolerance = number(value);
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (options.complex.empty() || options.poses.empty() || options.output.empty() ||
        !options.pool_size || options.record == 0)
        throw std::invalid_argument(
            "Required: --complex DIR --poses CSV --output NEW_DIR --pool-size N "
            "--pool-kind all-survivors|fast-prefilter|exported --interaction-region pocket|protein");
    if (options.pool_kind != "all-survivors" && options.pool_kind != "fast-prefilter" &&
        options.pool_kind != "exported")
        throw std::invalid_argument("Declare --pool-kind: all-survivors, fast-prefilter or exported");
    if (options.interaction_region != "pocket" && options.interaction_region != "protein")
        throw std::invalid_argument("Declare --interaction-region: pocket or protein");
    if (options.alpha < 0.0 || options.lambda < 0.0 || options.lambda > 1.0 || options.tolerance < 0.0)
        throw std::invalid_argument("Invalid alpha, lambda or score tolerance");
    if (fs::exists(options.output)) throw std::invalid_argument("Output directory already exists");
    return options;
}

RunConfig run_config(const Options& options)
{
    RunConfig config;
    config.complex_directory = fs::absolute(options.complex).lexically_normal();
    if (config.complex_directory.filename().empty())
        config.complex_directory = config.complex_directory.parent_path();
    config.complex_id = config.complex_directory.filename().string();
    const auto prefix = config.complex_directory / config.complex_id;
    config.protein_path = prefix.string() + "_protein.pdb";
    config.pocket_path = prefix.string() + "_pocket.pdb";
    config.ligand_path = options.ligand.empty() ? fs::path(prefix.string() + "_ligand.mol2") : options.ligand;
    return config;
}

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

GpuRigidTransform gpu_transform(const RigidTransform& transform)
{
    GpuRigidTransform result{};
    std::copy(transform.rotation.begin(), transform.rotation.end(), result.rotation);
    std::copy(transform.translation.begin(), transform.translation.end(), result.translation);
    result.valid = 1;
    return result;
}

void require_saved_score(std::optional<double> saved, double recomputed,
                         double tolerance, const char* name, std::size_t pose_id)
{
    if (!std::isfinite(recomputed)) throw std::runtime_error(std::string("Nonfinite ") + name);
    if (saved && std::abs(*saved - recomputed) > tolerance * (1.0 + std::abs(*saved)))
        throw std::runtime_error(std::string(name) + " differs from the saved score at pose_id " +
            std::to_string(pose_id) + "; check ligand, receptor region, crop, weights and backend precision");
}

std::ofstream csv_file(const fs::path& path)
{
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path);
    output << std::setprecision(17);
    return output;
}

void optional_number(std::ostream& output, std::optional<double> value)
{
    if (value) output << *value;
}

double ranking_score(const RankedPoseScore& row, PoseScoreMethod method)
{
    switch (method) {
    case PoseScoreMethod::Fast: return row.scored.pose.fast_weighted_score;
    case PoseScoreMethod::WeightedInteraction: return row.scored.pose.weighted_interaction_score.weighted_score;
    case PoseScoreMethod::Vinardo: return row.vinardo_adjusted.value();
    case PoseScoreMethod::Hybrid: return row.hybrid.value();
    }
    throw std::invalid_argument("Unknown ranking method");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: scoring_ablation --complex DIR --poses CSV --output NEW_DIR\n"
                "  --pool-size N --pool-kind all-survivors|fast-prefilter|exported\n"
                "  --interaction-region pocket|protein [--ligand MOL2] [--record 1]\n"
                "  [--reference MOL2] [--vinardo-alpha 1] [--hybrid-lambda 0.3]\n"
                "  [--score-tolerance 1e-6]\n";
            return 0;
        }
        const auto options = parse_options(argc, argv);
        const auto config = run_config(options);
        const auto start = Clock::now();
        std::ifstream input(options.poses);
        if (!input) throw std::runtime_error("Cannot open input pose CSV");
        const auto imported = read_pose_pool(input, options.record);
        if (imported.size() != *options.pool_size)
            throw std::runtime_error("Imported pose count differs from --pool-size; possible truncated pool");

        const auto ligand = load_ligand(config.ligand_path.string());
        const auto ligand_areas = get_ligand_areas(*ligand);
        const RescoringTarget target(config, options.interaction_region == "protein");
        const auto reference = options.reference.empty() ? std::unique_ptr<RDKit::RWMol>{}
            : load_ligand(options.reference.string());
        if (reference) validate_gpu_docking_reference(*ligand, *reference);
        const double preparation_seconds = seconds_since(start);
        std::vector<PoseAblation> poses;
        poses.reserve(imported.size());
        const auto scoring_start = Clock::now();
        std::size_t checked_weighted_interaction = 0, checked_vinardo = 0;
        for (const auto& input_pose : imported) {
            PoseAblation pose;
            pose.baseline = input_pose.scored;
            auto& baseline = pose.baseline;
            const auto areas = transform_interaction_areas(ligand_areas, baseline.pose.transform);
            baseline.pose.weighted_interaction_score = score_weighted_interactions(target.interaction_areas, areas);
            baseline.vinardo = target.vinardo.score(*ligand, baseline.pose.transform);
            require_saved_score(input_pose.saved_weighted_interaction, baseline.pose.weighted_interaction_score.weighted_score,
                                options.tolerance, "WeightedInteraction", baseline.pose_id);
            require_saved_score(input_pose.saved_vinardo, baseline.vinardo.total,
                                options.tolerance, "Vinardo-like", baseline.pose_id);
            checked_weighted_interaction += input_pose.saved_weighted_interaction.has_value();
            checked_vinardo += input_pose.saved_vinardo.has_value();
            for (std::size_t i = 0; i < weighted_interaction_families().size(); ++i)
                pose.weighted_interaction_without[i] = score_weighted_interactions(
                    target.interaction_areas, areas, weighted_interaction_weights_without(i));
            if (reference)
                pose.rmsd_heavy = gpu_pose_heavy_atom_rmsd(
                    *ligand, gpu_transform(baseline.pose.transform), *reference);
            poses.push_back(pose);
        }
        const double scoring_seconds = seconds_since(scoring_start);

        if (!options.output.parent_path().empty()) fs::create_directories(options.output.parent_path());
        if (!fs::create_directory(options.output)) throw std::runtime_error("Cannot create new output directory");
        auto contributions = csv_file(options.output / "contributions.csv");
        contributions << "record_index,pose_id,fast_score,weighted_interaction_score,weighted_interaction_total_interactions,"
            "weighted_interaction_satisfied_ligand_areas,weighted_interaction_satisfied_protein_areas,vinardo_like_score,"
            "vinardo_adjusted_score,vinardo_gaussian,vinardo_repulsion,vinardo_hydrophobic,"
            "vinardo_hydrogen_bond,rmsd_heavy_A\n";
        auto weighted_interaction_terms = csv_file(options.output / "weighted_interaction_ablation.csv");
        weighted_interaction_terms << "record_index,pose_id,family,contacts,default_weight,weighted_interaction_full,weighted_interaction_without,"
            "marginal_loss_not_additive\n";
        std::map<std::size_t, std::optional<double>> rmsds;
        for (const auto& pose : poses) {
            const auto& row = pose.baseline;
            const auto& weighted_interaction = row.pose.weighted_interaction_score;
            const auto& v = row.vinardo;
            rmsds.emplace(row.pose_id, pose.rmsd_heavy);
            contributions << options.record << ',' << row.pose_id << ',' << row.pose.fast_weighted_score
                << ',' << weighted_interaction.weighted_score << ',' << weighted_interaction.total_interactions << ','
                << weighted_interaction.satisfied_ligand_areas << ',' << weighted_interaction.satisfied_protein_areas << ','
                << v.total << ',' << vinardo_adjusted_score(v, options.alpha) << ',' << v.gaussian << ','
                << v.repulsion << ',' << v.hydrophobic << ',' << v.hydrogen_bond << ',';
            optional_number(contributions, pose.rmsd_heavy);
            contributions << '\n';
            const WeightedInteractionWeights weights;
            for (std::size_t i = 0; i < weighted_interaction_families().size(); ++i) {
                const auto& family = weighted_interaction_families()[i];
                const auto without = pose.weighted_interaction_without[i].weighted_score;
                weighted_interaction_terms << options.record << ',' << row.pose_id << ',' << family.name << ','
                    << weighted_interaction.*(family.count) << ',' << weights.*(family.weight) << ','
                    << weighted_interaction.weighted_score << ',' << without << ',' << weighted_interaction.weighted_score - without << '\n';
            }
        }

        auto rankings = csv_file(options.output / "rankings.csv");
        rankings << "record_index,variant,ranking,rank,pose_id,score,rmsd_heavy_A,success_lt_2A\n";
        auto changes = csv_file(options.output / "top1_changes.csv");
        changes << "record_index,variant,ranking,pool_size,status,baseline_top1_id,top1_id,"
            "top1_changed,baseline_top1_new_rank,baseline_top1_score,top1_score,"
            "baseline_rmsd_heavy_A,rmsd_heavy_A,baseline_success_lt_2A,success_lt_2A\n";
        const auto baseline = make_pool(poses);
        // RMSD is looked up only after the existing score-only ranker returns.
        const auto write_ranking = [&](const std::string& variant, PoseScoreMethod method,
                                       const PoseRescoringResult& pool) {
            const PoseScoreRankingOptions ranking_options{method, options.alpha, options.lambda};
            const auto ranked = rank_rescored_poses(pool, ranking_options, std::max<std::size_t>(1, poses.size()));
            const auto original = rank_rescored_poses(baseline, ranking_options, 1);
            const char* name = pose_score_method_name(method);
            for (std::size_t i = 0; i < ranked.size(); ++i) {
                const auto& row = ranked[i];
                const auto rmsd = rmsds.at(row.scored.pose_id);
                rankings << options.record << ',' << variant << ',' << name << ',' << i + 1 << ','
                    << row.scored.pose_id << ',' << ranking_score(row, method) << ',';
                optional_number(rankings, rmsd);
                rankings << ',';
                if (rmsd) rankings << (*rmsd < 2.0);
                rankings << '\n';
            }
            changes << options.record << ',' << variant << ',' << name << ',' << poses.size() << ',';
            if (ranked.empty()) {
                changes << "no_poses,,,,,,,,,,\n";
                return;
            }
            const auto old_id = original.front().scored.pose_id;
            const auto new_id = ranked.front().scored.pose_id;
            const auto old_rmsd = rmsds.at(old_id), new_rmsd = rmsds.at(new_id);
            const auto old_position = std::find_if(ranked.begin(), ranked.end(),
                [&](const auto& row) { return row.scored.pose_id == old_id; });
            changes << (reference ? "ok" : "no_reference") << ',' << old_id << ',' << new_id << ','
                << (old_id != new_id) << ',' << std::distance(ranked.begin(), old_position) + 1 << ','
                << ranking_score(original.front(), method) << ',' << ranking_score(ranked.front(), method) << ',';
            optional_number(changes, old_rmsd);
            changes << ',';
            optional_number(changes, new_rmsd);
            changes << ',';
            if (old_rmsd) changes << (*old_rmsd < 2.0);
            changes << ',';
            if (new_rmsd) changes << (*new_rmsd < 2.0);
            changes << '\n';
        };
        for (const auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                                   PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
            write_ranking("baseline", method, baseline);
        for (std::size_t i = 0; i < weighted_interaction_families().size(); ++i) {
            const auto pool = make_pool(poses, RemovedScore::WeightedInteraction, i);
            const auto variant = std::string("without_weighted_interaction_") + weighted_interaction_families()[i].name;
            write_ranking(variant, PoseScoreMethod::WeightedInteraction, pool);
            write_ranking(variant, PoseScoreMethod::Hybrid, pool);
        }
        for (std::size_t i = 0; i < vinardo_terms().size(); ++i) {
            const auto pool = make_pool(poses, RemovedScore::Vinardo, i);
            const auto variant = std::string("without_vinardo_") + vinardo_terms()[i];
            write_ranking(variant, PoseScoreMethod::Vinardo, pool);
            write_ranking(variant, PoseScoreMethod::Hybrid, pool);
        }
        auto protocol = csv_file(options.output / "protocol.csv");
        const auto setting = [&](const char* key, const auto& value) {
            protocol << key << ",\"";
            for (const char character : std::string(value)) {
                if (character == '"') protocol << '"';
                protocol << character;
            }
            protocol << "\"\n";
        };
        protocol << "key,value\n";
        setting("complex_id", config.complex_id);
        setting("input_pose_csv", fs::absolute(options.poses).string());
        setting("ligand_path", fs::absolute(config.ligand_path).string());
        setting("protein_path", config.protein_path.string());
        setting("pocket_path", config.pocket_path.string());
        setting("reference_path", options.reference.empty() ? "none" : fs::absolute(options.reference).string());
        setting("pool_kind_declared", options.pool_kind);
        setting("pose_id_source", "input pose_id; alignment_index only when pose_id column is absent");
        setting("pool_validation", "row count checked against declared size; upstream completeness must be checked in generation protocol");
        setting("interaction_region", options.interaction_region);
        const char* crop_margin = std::getenv("DOCKING_AREA_CROP_MARGIN_A");
        setting("chemical_area_crop", crop_margin
            ? "whole-residue pocket bounding-box crop before RescoringTarget area detection"
            : "none; standard RescoringTarget pocket selection or full protein");
        setting("DOCKING_AREA_CROP_MARGIN_A", crop_margin ? crop_margin : "unset");
        setting("vinardo_region", "full supported heavy-atom receptor; original strict 8 A pair cutoff");
        setting("weighted_interaction_ablation", "zero one reward; recompute max per physical ligand area with the existing scorer");
        setting("weighted_interaction_counts", "geometric contacts; retained when reward is zero; not additive weighted contributions");
        setting("vinardo_components", "signed and weighted; removing repulsion removes its alpha-scaled contribution");
        setting("ranking", "existing rank_rescored_poses; WeightedInteraction/Hybrid/Fast maximize; Vinardo minimize");
        setting("ties", "pose_id ascending, then input offset; not the legacy rank_poses coverage/clash ties");
        setting("hybrid_normalization", "recomputed separately per variant on the same complete supplied pool; population SD; zero variance gives zero z-score");
        setting("rmsd", "optional heavy atoms by verified atom index in protein frame; no alignment or symmetry correction; never used in ranking");
        setting("success", "RMSD strictly less than 2 A; blank without reference; no_poses is distinct");
        setting("backend", "CPU scoring replay only; no geometric search, grid, triplet index or clash recomputation");
        const WeightedInteractionWeights weighted_interaction_weights;
        for (const auto& family : weighted_interaction_families())
            protocol << "weighted_interaction_weight_" << family.name << ',' << weighted_interaction_weights.*(family.weight) << '\n';
        protocol << "record_index," << options.record << "\npool_size," << poses.size()
            << "\nvinardo_alpha," << options.alpha << "\nhybrid_lambda," << options.lambda
            << "\nscore_tolerance_relative_plus_absolute," << options.tolerance
            << "\nsaved_weighted_interaction_scores_checked," << checked_weighted_interaction << "\nsaved_vinardo_scores_checked," << checked_vinardo
            << "\nunsupported_receptor_atoms_skipped," << target.vinardo.receptor_skipped_unsupported()
            << "\npreparation_s," << preparation_seconds << "\nscoring_ablation_and_rmsd_s," << scoring_seconds
            << "\nwall_to_protocol_s," << seconds_since(start) << '\n';
        contributions.close(); weighted_interaction_terms.close(); rankings.close(); changes.close(); protocol.close();
        std::cout << "Scoring ablation complete: " << poses.size() << " poses, record " << options.record
            << "; output " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Scoring ablation: " << error.what() << '\n';
        return 1;
    }
}
