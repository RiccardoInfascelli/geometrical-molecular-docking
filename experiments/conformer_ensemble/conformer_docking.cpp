#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "interaction_scoring.hpp"
#include "ligand_preparation.hpp"
#include "molecule_io.hpp"
#include "pose_generation.hpp"
#include "pose_score_ranking.hpp"
#include "protein_preparation.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    return result + '"';
}

std::ofstream output_file(const fs::path& path) {
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
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

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 5 && argc != 7) {
            throw std::runtime_error(
                "Usage: conformer_docking <complex-directory> <conformers.mol2> "
                "<reference.mol2> <new-output-directory> [--cutoff 24]");
        }
        double cutoff = 24.0;
        if (argc == 7) {
            if (std::string(argv[5]) != "--cutoff")
                throw std::runtime_error("The only optional argument is --cutoff");
            std::size_t consumed = 0;
            cutoff = std::stod(argv[6], &consumed);
            if (consumed != std::string(argv[6]).size() ||
                !std::isfinite(cutoff) || cutoff <= 0.0)
                throw std::runtime_error("Cutoff must be finite and positive");
        }
        const fs::path output = argv[4];
        if (!fs::create_directories(output))
            throw std::runtime_error("Output directory already exists: " + output.string());
        const auto wall_start = Clock::now();
        char* config_args[]{argv[0], argv[1]};
        const auto config = parse_run_config(2, config_args);
        const auto reference = load_ligand(argv[3]);

        auto start = Clock::now();
        const auto protein = prepare_protein(config);
        const double protein_s = seconds_since(start);
        start = Clock::now();
        const auto index = flatten_protein_triplet_index(build_protein_triplet_hash_table(
            protein.targets, cutoff, config.triplet_bin_size));
        const double index_s = seconds_since(start);
        const double startup_s = seconds_since(wall_start);

        auto poses_csv = output_file(output / "poses.csv");
        poses_csv << "record_index,molecule_id,pose_id,fast_score,rmsd_heavy_A,"
            "clash_fraction,occupied_atoms,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
        auto records = output_file(output / "records.csv");
        records << "record_index,molecule_id,status,ligand_triplets,candidate_matches,"
            "transforms_generated,survivors,top1_pose_id,top1_fast_score,top1_rmsd_heavy_A,"
            "success_lt_2A,best_available_rmsd_heavy_A,read_parse_preparation_s,"
            "pose_generation_s,fast_scoring_s,rmsd_evaluation_s\n";

        Mol2Reader reader(argv[2]);
        std::string block, molecule_id;
        std::set<std::string> names;
        std::size_t record_count = 0;
        while (true) {
            start = Clock::now();
            if (!reader.next(block, molecule_id)) break;
            if (!names.insert(molecule_id).second)
                throw std::runtime_error("Duplicate conformer name: " + molecule_id);
            if (record_count == 0 && molecule_id != "baseline")
                throw std::runtime_error("The first MOL2 record must be named baseline");
            auto molecule = parse_ligand_mol2_block(block);
            validate_gpu_docking_reference(*molecule, *reference);
            if (record_count == 0 &&
                gpu_pose_heavy_atom_rmsd(*molecule, rmsd_transform({}), *reference) != 0.0)
                throw std::runtime_error("Baseline heavy-atom coordinates differ from the reference");
            PreparedLigand ligand;
            prepare_ligand(*molecule, ligand);
            const double preparation_s = seconds_since(start);

            start = Clock::now();
            // The supplied index is shared and never expanded to a conformer's diameter.
            const auto generated = generate_candidate_poses_from_index(
                *molecule, ligand.matching_areas, ligand.triplets, protein.targets,
                protein.grid, index, config.triplet_bin_size,
                config.operational_clash_threshold, config.clash_thresholds, true);
            const double generation_s = seconds_since(start);

            start = Clock::now();
            std::vector<PoseScoreEntry> scores;
            scores.reserve(generated.surviving_poses.size());
            for (std::size_t id = 0; id < generated.surviving_poses.size(); ++id) {
                const auto areas = transform_matching_areas(
                    ligand.matching_areas, generated.surviving_poses[id].transform);
                scores.push_back({id, score_matching_areas_weighted(areas, protein.grid).weighted_score});
            }
            const auto selected = top_pose_score_indices(scores, 1);
            const double fast_s = seconds_since(start);

            // Select by Fast first; evaluate every survivor against the fixed crystal reference.
            start = Clock::now();
            std::vector<double> rmsds;
            rmsds.reserve(scores.size());
            double best_rmsd = std::numeric_limits<double>::infinity();
            for (const auto& pose : generated.surviving_poses) {
                const double rmsd = gpu_pose_heavy_atom_rmsd(
                    *molecule, rmsd_transform(pose.transform), *reference);
                rmsds.push_back(rmsd);
                best_rmsd = std::min(best_rmsd, rmsd);
            }
            const double rmsd_s = seconds_since(start);
            ++record_count;

            for (std::size_t id = 0; id < scores.size(); ++id) {
                const auto& pose = generated.surviving_poses[id];
                poses_csv << record_count << ',' << csv_string(molecule_id) << ',' << id << ','
                    << scores[id].score << ',' << rmsds[id] << ',' << pose.clash_fraction
                    << ',' << pose.occupied_atoms;
                for (double value : pose.transform.rotation) poses_csv << ',' << value;
                for (double value : pose.transform.translation) poses_csv << ',' << value;
                poses_csv << '\n';
            }
            const char* status = ligand.triplets.empty() ? "no_triplets" :
                selected.empty() ? "no_poses" : "ok";
            records << record_count << ',' << csv_string(molecule_id) << ',' << status << ','
                << ligand.triplets.size() << ',' << generated.total_candidate_matches << ','
                << generated.transforms_generated << ',' << scores.size() << ',';
            if (selected.empty()) {
                records << ",,,0,";
            } else {
                const auto id = selected.front();
                records << scores[id].pose_id << ',' << scores[id].score << ',' << rmsds[id]
                    << ',' << (rmsds[id] < 2.0) << ',' << best_rmsd;
            }
            records << ',' << preparation_s << ',' << generation_s << ',' << fast_s
                << ',' << rmsd_s << '\n';
            std::cout << molecule_id << ": " << status << ", " << scores.size()
                << " survivors\n";
        }
        if (record_count == 0) throw std::runtime_error("Conformer MOL2 contains no records");
        poses_csv.close();
        records.close();
        auto protocol = output_file(output / "protocol.csv");
        protocol << "key,value\n";
        const auto setting = [&](const std::string& key, const std::string& value) {
            protocol << key << ',' << csv_string(value) << '\n';
        };
        setting("backend", "CPU; existing engine functions only");
        setting("complex_directory", config.complex_directory.string());
        setting("protein_path", config.protein_path.string());
        setting("pocket_path", config.pocket_path.string());
        setting("conformers_mol2", fs::absolute(argv[2]).string());
        setting("reference_mol2", fs::absolute(argv[3]).string());
        setting("index_policy", "shared_index_only; fixed cutoff; no fallback or pruning");
        setting("scoring", "unchanged Fast default weights; maximize");
        setting("tie_break", "pose_id ascending within record; ensemble ties use record_index then pose_id");
        setting("pose_pool", "all operational clash survivors; no prefilter or deduplication");
        setting("rmsd", "heavy atoms by validated fixed index in protein frame; no alignment or symmetry correction; never ranked");
        setting("baseline", "first record named baseline; heavy-atom coordinates equal reference");
        setting("success", "Top-1 RMSD strictly < 2 A; no poses is failure with blank RMSD");
        setting("screening_only", "forced true; native RMSD and unweighted support skipped");
        setting("timer_scope", "record stage times exclude CSV writes; wall includes preparation, all records, RMSD and pose/record CSV writes");
        const FastScoreWeights weights;
        protocol << "fast_weight_hydrophobic," << weights.hydrophobic
            << "\nfast_weight_hydrogen_bond," << weights.hydrogen_bond
            << "\nfast_weight_halogen_bond," << weights.halogen_bond
            << "\nfast_weight_cation," << weights.cation << "\nfast_weight_anion," << weights.anion
            << "\nfast_weight_aromatic," << weights.aromatic
            << "\nfast_weight_metal," << weights.metal << "\nfast_weight_chelated," << weights.chelated << '\n';
        for (const char* key : {"DOCKING_KABSCH_METHOD", "DOCKING_CPU_CLASH_METHOD",
                "DOCKING_QUERY_METHOD", "DOCKING_CPU_SCREENING_ONLY", "DOCKING_PROFILE_STAGES",
                "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"}) {
            const char* value = std::getenv(key);
            setting(key, value ? value : "unset");
        }
        protocol << "cutoff_A," << cutoff << "\ngrid_voxel_size_A," << config.grid_voxel_size
            << "\ngrid_margin_A," << config.grid_margin
            << "\npocket_centroid_distance_A," << config.pocket_centroid_distance
            << "\ntriplet_bin_size_A," << config.triplet_bin_size
            << "\noperational_clash_fraction," << config.operational_clash_threshold;
        for (std::size_t i = 0; i < config.clash_thresholds.size(); ++i)
            protocol << "\ndiagnostic_clash_fraction_" << i << ',' << config.clash_thresholds[i];
        protocol << "\nprotein_target_count," << protein.targets.size()
            << "\nprotein_triplet_records," << index.records.size()
            << "\nprotein_preparation_s," << protein_s << "\nindex_build_s," << index_s
            << "\nstartup_s," << startup_s << "\nrecord_count," << record_count
            << "\nwall_s," << seconds_since(wall_start) << '\n';
        protocol.close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conformer_docking: " << error.what()
            << "\nAny partial output is incomplete and must not be treated as a successful run.\n";
        return 1;
    }
}
