#include "hybrid_ranking.hpp"

#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "interaction_areas.hpp"
#include "molecule_io.hpp"
#include "pose_rescoring.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace conformer_ensemble;

namespace {

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t integer(const std::string& text) {
    std::size_t result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Expected an unsigned integer: " + text);
    return result;
}

std::string csv_string(const std::string& text) {
    std::string result = "\"";
    for (char c : text) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}

std::ofstream output_file(const fs::path& path) {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path);
    output << std::setprecision(17);
    return output;
}

GpuRigidTransform rmsd_transform(const RigidTransform& transform) {
    GpuRigidTransform result{};
    std::copy(transform.rotation.begin(), transform.rotation.end(), result.rotation);
    std::copy(transform.translation.begin(), transform.translation.end(), result.translation);
    result.valid = 1;
    return result;
}

struct Conformer {
    std::string name;
    std::unique_ptr<RDKit::RWMol> molecule;
    std::vector<interaction::interaction_area> areas;
};

void write_rankings(const fs::path& path, const std::vector<RankedEnsemblePose>& rows) {
    auto output = output_file(path);
    output << "conformer_id,record_index,pose_id,local_rank,global_rank,weighted_interaction_score,"
        "vinardo_rep50,hybrid,rmsd_heavy_A,success_lt_2A\n";
    for (const auto& row : rows)
        output << csv_string(row.pose.conformer_id) << ',' << row.pose.record_index << ','
            << row.pose.scored.pose_id << ',' << row.local_rank << ',' << row.global_rank << ','
            << row.pose.scored.pose.weighted_interaction_score.weighted_score << ',' << row.vinardo_rep50 << ','
            << row.hybrid << ',' << row.pose.rmsd_heavy << ',' << (row.pose.rmsd_heavy < 2.0) << '\n';
    output.close();
}

void write_topk(std::ostream& output, const char* population, const HybridEnsembleRanking& ranked) {
    double best_available = 0.0;
    for (std::size_t i = 0; i < ranked.all.size(); ++i)
        if (i == 0 || ranked.all[i].pose.rmsd_heavy < best_available)
            best_available = ranked.all[i].pose.rmsd_heavy;
    for (std::size_t k : {1, 5, 10}) {
        const auto n = std::min(k, ranked.global.size());
        output << population << ',' << k << ',' << ranked.all.size() << ',' << ranked.local.size()
            << ',' << n << ',';
        if (n == 0) {
            output << "no_poses,,,0,\n";
            continue;
        }
        double best = ranked.global[0].pose.rmsd_heavy;
        for (std::size_t i = 1; i < n; ++i) best = std::min(best, ranked.global[i].pose.rmsd_heavy);
        output << "ok," << ranked.global[0].pose.rmsd_heavy << ',' << best << ','
            << (best < 2.0) << ',' << best_available << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6 || (argc - 6) % 2)
            throw std::runtime_error("Usage: conformer_hybrid <complex> <conformers.mol2> <reference.mol2> "
                "<poses.tsv> <new-output> [--interaction-region pocket|protein] [--local-k 10] [--top-x 10]");
        bool full_protein = false;
        std::size_t local_k = 10, top_x = 10;
        for (int i = 6; i < argc; i += 2) {
            const std::string option = argv[i], value = argv[i + 1];
            if (option == "--local-k") local_k = integer(value);
            else if (option == "--top-x") top_x = integer(value);
            else if (option == "--interaction-region" && (value == "pocket" || value == "protein"))
                full_protein = value == "protein";
            else throw std::runtime_error("Invalid option or value: " + option);
        }
        if (top_x < 10 || local_k < top_x)
            throw std::runtime_error("Require local K >= global X >= 10 to report Top-1/5/10");
        const fs::path output = argv[5];
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists");
        const auto wall_start = Clock::now();
        char* config_args[]{argv[0], argv[1]};
        const auto config = parse_run_config(2, config_args);
        auto start = Clock::now();
        const RescoringTarget target(config, full_protein);
        const auto reference = load_ligand(argv[3]);
        std::vector<Conformer> conformers;
        Mol2Reader molecules(argv[2]);
        std::set<std::string> names;
        std::string block, name;
        while (molecules.next(block, name)) {
            if (!names.insert(name).second) throw std::runtime_error("Duplicate conformer name");
            auto molecule = parse_ligand_mol2_block(block);
            validate_gpu_docking_reference(*molecule, *reference);
            auto areas = get_ligand_areas(*molecule);
            conformers.push_back({name, std::move(molecule), std::move(areas)});
        }
        if (conformers.size() < 2 || conformers[0].name != "baseline")
            throw std::runtime_error("Expected a separate first baseline and at least one generated conformer");
        if (gpu_pose_heavy_atom_rmsd(*conformers[0].molecule, rmsd_transform({}), *reference) != 0.0)
            throw std::runtime_error("Baseline coordinates differ from crystal reference");
        const double preparation_s = elapsed(start);

        start = Clock::now();
        std::ifstream input(argv[4]);
        if (!input) throw std::runtime_error("Cannot open numeric pose input");
        std::string line;
        if (!std::getline(input, line) || line != "record_index\tpose_id\tfast_score\tclash_fraction\toccupied_atoms\trmsd_heavy_A\tr00\tr01\tr02\tr10\tr11\tr12\tr20\tr21\tr22\ttx\tty\ttz")
            throw std::runtime_error("Unexpected numeric pose input header");
        std::vector<EnsemblePose> poses;
        std::set<std::pair<std::size_t, std::size_t>> ids;
        while (std::getline(input, line)) {
            if (line.empty()) throw std::runtime_error("Empty pose input row");
            std::istringstream row(line);
            std::string record_text, id_text, occupied_text;
            EnsemblePose pose;
            double saved_rmsd;
            auto& scored = pose.scored;
            if (!(row >> record_text >> id_text >> scored.pose.fast_weighted_score >>
                    scored.pose.clash_fraction >> occupied_text >> saved_rmsd))
                throw std::runtime_error("Malformed pose input row");
            pose.record_index = integer(record_text);
            scored.pose_id = integer(id_text);
            scored.pose.occupied_atoms = integer(occupied_text);
            if (pose.record_index == 0 || pose.record_index > conformers.size() ||
                !ids.emplace(pose.record_index, scored.pose_id).second)
                throw std::runtime_error("Unexpected record or duplicate pose ID");
            for (auto& value : scored.pose.transform.rotation)
                if (!(row >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid rotation");
            for (auto& value : scored.pose.transform.translation)
                if (!(row >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid translation");
            row >> std::ws;
            if (!row.eof() || !std::isfinite(saved_rmsd) || saved_rmsd < 0.0 ||
                !std::isfinite(scored.pose.fast_weighted_score) || !std::isfinite(scored.pose.clash_fraction) ||
                scored.pose.clash_fraction < 0.0 || scored.pose.clash_fraction > 1.0)
                throw std::runtime_error("Invalid numeric pose data");
            const auto& conformer = conformers[pose.record_index - 1];
            pose.conformer_id = conformer.name;
            pose.rmsd_heavy = gpu_pose_heavy_atom_rmsd(
                *conformer.molecule, rmsd_transform(scored.pose.transform), *reference);
            if (std::abs(saved_rmsd - pose.rmsd_heavy) > 1e-9 * (1.0 + std::abs(saved_rmsd)))
                throw std::runtime_error("Saved RMSD disagrees with conformer/transform/reference identity");
            poses.push_back(std::move(pose));
        }
        if (input.bad()) throw std::runtime_error("Could not read pose input");
        const double validation_s = elapsed(start);

        start = Clock::now();
        for (auto& pose : poses) {
            const auto& conformer = conformers[pose.record_index - 1];
            pose.scored.pose.weighted_interaction_score = score_weighted_interactions(target.interaction_areas,
                transform_interaction_areas(conformer.areas, pose.scored.pose.transform));
        }
        const double weighted_interaction_s = elapsed(start);
        std::cout << "WeightedInteraction: " << poses.size() << " saved survivors scored\n" << std::flush;
        start = Clock::now();
        std::size_t previous_record = 0;
        for (auto& pose : poses) {
            if (pose.record_index != previous_record) {
                std::cout << "Vinardo: " << pose.conformer_id << '\n' << std::flush;
                previous_record = pose.record_index;
            }
            pose.scored.vinardo = target.vinardo.score(
                *conformers[pose.record_index - 1].molecule, pose.scored.pose.transform);
        }
        const double vinardo_s = elapsed(start);
        std::vector<EnsemblePose> baseline, ensemble;
        for (auto& pose : poses) {
            if (pose.record_index == 1) baseline.push_back(std::move(pose));
            else ensemble.push_back(std::move(pose));
        }
        start = Clock::now();
        const auto ranked = rank_ensemble_poses(std::move(ensemble), local_k, top_x);
        const auto native = rank_ensemble_poses(std::move(baseline), local_k, top_x);
        const double ranking_s = elapsed(start);

        if (!fs::create_directories(output)) throw std::runtime_error("Cannot create new output directory");
        write_rankings(output / "global_rankings.csv", ranked.global);
        write_rankings(output / "local_rankings.csv", ranked.local);
        write_rankings(output / "baseline_rankings.csv", native.global);
        auto all = output_file(output / "all_pose_scores.csv");
        all << "population,conformer_id,record_index,pose_id,local_rank,full_pool_rank,weighted_interaction_score,"
            "vinardo_standard,vinardo_rep50,hybrid,gaussian,repulsion,hydrophobic,hydrogen_bond,rmsd_heavy_A\n";
        for (const auto* population : {&ranked, &native}) {
            const char* label = population == &ranked ? "ensemble" : "baseline";
            for (const auto& row : population->all) {
                const auto& v = row.pose.scored.vinardo;
                all << label << ',' << csv_string(row.pose.conformer_id) << ',' << row.pose.record_index << ','
                    << row.pose.scored.pose_id << ',' << row.local_rank << ',' << row.global_rank << ','
                    << row.pose.scored.pose.weighted_interaction_score.weighted_score << ',' << v.total << ',' << row.vinardo_rep50
                    << ',' << row.hybrid << ',' << v.gaussian << ',' << v.repulsion << ',' << v.hydrophobic << ','
                    << v.hydrogen_bond << ',' << row.pose.rmsd_heavy << '\n';
            }
        }
        all.close();
        auto summary = output_file(output / "topk_summary.csv");
        summary << "population,top_k,pool_size,local_candidates,returned_poses,status,top1_rmsd_heavy_A,"
            "best_rmsd_in_top_k_A,success_lt_2A,best_available_rmsd_heavy_A\n";
        write_topk(summary, "ensemble", ranked);
        write_topk(summary, "baseline", native);
        summary.close();
        auto protocol = output_file(output / "protocol.csv");
        protocol << "key,value\nalpha,0.5\nlambda,0.3\nlocal_k," << local_k << "\nglobal_x," << top_x << '\n';
        const auto setting = [&](const char* key, const std::string& value) {
            protocol << key << ',' << csv_string(value) << '\n';
        };
        setting("ranking", "existing rank_rescored_poses Hybrid; maximize; no Fast prefilter");
        setting("normalization", "one call on all generated-conformer survivors before local Top-K; population SD; zero variance gives zero z-score");
        setting("baseline", "excluded from ensemble statistics and local/global selection; own complete-pool normalization");
        setting("tie_break", "record_index then original pose_id; never RMSD, MMFF energy or Fast");
        setting("interaction_region", full_protein ? "full protein areas" : "standard pocket area membership");
        setting("vinardo_region", "full supported heavy-atom receptor; unchanged strict 8 A pair cutoff; no crop");
        setting("rmsd", "recomputed with existing heavy-atom helper; fixed atom mapping in protein frame; no alignment or symmetry correction; evaluation only");
        setting("scope", "rescoring saved transforms only; no docking, Kabsch, clash, grid or index reconstruction");
        protocol << "ensemble_poses," << ranked.all.size() << "\nbaseline_poses," << native.all.size()
            << "\nconformers," << conformers.size() - 1 << "\nscoring_preparation_s," << preparation_s
            << "\ninput_and_rmsd_validation_s," << validation_s << "\nweighted_interaction_scoring_s," << weighted_interaction_s
            << "\nvinardo_scoring_s," << vinardo_s << "\nranking_s," << ranking_s
            << "\nwall_through_export_s," << elapsed(wall_start) << '\n';
        protocol.close();
        std::cout << "Global Hybrid complete: " << ranked.all.size() << " ensemble poses, "
            << native.all.size() << " separate baseline poses; " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conformer_hybrid: " << error.what() << '\n';
        return 1;
    }
}
