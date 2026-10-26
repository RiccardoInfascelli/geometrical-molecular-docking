#include "gpu_docking.hpp"
#include "ligand_preparation.hpp"
#include "molecule_io.hpp"
#include "pipeline.hpp"
#include "pose_rescoring.hpp"
#include "pose_score_ranking.hpp"
#include "protein_preparation.hpp"

#include <GraphMol/Conformer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Actual, class Expected>
void same(Actual actual, Expected expected, const std::string& field) {
    if (actual != expected) {
        std::ostringstream message;
        message << field << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(message.str());
    }
}

// Existing real-pipeline tolerances cover independent double-precision fits.
// Rescoring tolerances match cuda_pose_rescoring_test; discrete decisions remain exact.
struct Errors {
    double transform = 0.0;
    double point = 0.0;
    double fast = 0.0;
    double vinardo = 0.0;
    double hybrid = 0.0;

    void close(double actual, double expected, double absolute, double relative,
               double& maximum, const std::string& field) {
        require(std::isfinite(actual) && std::isfinite(expected), field + ": nonfinite value");
        const double error = std::abs(actual - expected);
        maximum = std::max(maximum, error);
        if (error > absolute + relative * std::abs(expected)) {
            std::ostringstream message;
            message << std::setprecision(17) << field << ": expected " << expected
                    << ", got " << actual << ", absolute error " << error
                    << ", allowed " << absolute + relative * std::abs(expected);
            throw std::runtime_error(message.str());
        }
    }
};

std::string selected_method(const char* variable, const char* fallback,
                            const char* first, const char* second) {
    const char* value = std::getenv(variable);
    const std::string method = value ? value : fallback;
    require(method == first || method == second,
            std::string(variable) + " must be " + first + " or " + second);
    return method;
}

RunConfig configuration(const char* directory) {
    RunConfig config;
    config.complex_directory = std::filesystem::absolute(directory).lexically_normal();
    if (config.complex_directory.filename().empty())
        config.complex_directory = config.complex_directory.parent_path();
    config.complex_id = config.complex_directory.filename().string();
    require(!config.complex_id.empty(), "Cannot determine the complex ID");
    config.protein_path = config.complex_directory / (config.complex_id + "_protein.pdb");
    config.pocket_path = config.complex_directory / (config.complex_id + "_pocket.pdb");
    config.ligand_path = config.complex_directory / (config.complex_id + "_ligand.mol2");
    for (const auto& path : {config.protein_path, config.pocket_path, config.ligand_path})
        require(std::filesystem::is_regular_file(path), "Missing molecular input: " + path.string());
    config.output_mode = OutputMode::Quiet;
    return config;
}

void compare_transform(const RigidTransform& cpu, const GpuRigidTransform& gpu,
                       const RDKit::ROMol& ligand, const std::string& label,
                       Errors& errors) {
    require(gpu.valid != 0, label + ": invalid GPU transform");
    for (std::size_t i = 0; i < 9; ++i)
        errors.close(gpu.rotation[i], cpu.rotation[i], 1e-6, 0.0, errors.transform,
                     label + " rotation[" + std::to_string(i) + "]");
    for (std::size_t i = 0; i < 3; ++i)
        errors.close(gpu.translation[i], cpu.translation[i], 1e-6, 0.0, errors.transform,
                     label + " translation[" + std::to_string(i) + "]");

    // Compare physical placement as well as coefficients: a small rotation error
    // can have a larger effect on atoms far from the coordinate origin.
    const auto& conformer = ligand.getConformer();
    for (unsigned int atom = 0; atom < ligand.getNumAtoms(); ++atom) {
        const auto& point = conformer.getAtomPos(atom);
        const std::array<double, 3> p{point.x, point.y, point.z};
        for (std::size_t row = 0; row < 3; ++row) {
            double expected = cpu.translation[row];
            double actual = gpu.translation[row];
            for (std::size_t col = 0; col < 3; ++col) {
                expected += cpu.rotation[3 * row + col] * p[col];
                actual += gpu.rotation[3 * row + col] * p[col];
            }
            errors.close(actual, expected, 1e-5, 0.0, errors.point,
                         label + " atom " + std::to_string(atom) + " coordinate " +
                         std::to_string(row));
        }
    }
}

void compare_weighted_interaction(const WeightedInteractionScore& cpu, const WeightedInteractionScore& gpu,
                   const std::string& label) {
    same(gpu.total_interactions, cpu.total_interactions, label + " contacts");
    same(gpu.satisfied_ligand_areas, cpu.satisfied_ligand_areas, label + " ligand coverage");
    same(gpu.satisfied_protein_areas, cpu.satisfied_protein_areas, label + " protein coverage");
    same(gpu.hydrophobic, cpu.hydrophobic, label + " hydrophobic");
    same(gpu.hydrogen_bonds, cpu.hydrogen_bonds, label + " hydrogen bonds");
    same(gpu.halogen_bonds, cpu.halogen_bonds, label + " halogen bonds");
    same(gpu.ionic, cpu.ionic, label + " ionic");
    same(gpu.cation_pi, cpu.cation_pi, label + " cation-pi");
    same(gpu.pi_stacking, cpu.pi_stacking, label + " pi stacking");
    same(gpu.metal, cpu.metal, label + " metal");
    require(std::isfinite(cpu.weighted_score) && std::isfinite(gpu.weighted_score),
            label + ": nonfinite WeightedInteraction score");
    // The default WeightedInteraction weights are binary-exact fractions; contact decisions
    // and the maximum per area must agree, rather than merely have close totals.
    same(gpu.weighted_score, cpu.weighted_score, label + " weighted score");
}

void compare_vinardo(const VinardoLikeScore& cpu, const VinardoLikeScore& gpu,
                     const std::string& label, Errors& errors) {
    const std::array<double, 5> expected{
        cpu.total, cpu.gaussian, cpu.repulsion, cpu.hydrophobic, cpu.hydrogen_bond};
    const std::array<double, 5> actual{
        gpu.total, gpu.gaussian, gpu.repulsion, gpu.hydrophobic, gpu.hydrogen_bond};
    const std::array<const char*, 5> names{
        "total", "gaussian", "repulsion", "hydrophobic", "hydrogen bond"};
    for (std::size_t i = 0; i < expected.size(); ++i)
        errors.close(actual[i], expected[i], 1e-10, 1e-8, errors.vinardo,
                     label + " Vinardo " + names[i]);
}

std::vector<std::size_t> survivor_offsets(const GpuDockingResult& gpu) {
    same(gpu.poses.size(), gpu.survivors, "Full GPU survivor download");
    same(gpu.scores.size(), gpu.poses.size(), "GPU score/pose pairing");
    std::vector<std::size_t> offsets(gpu.survivors, std::numeric_limits<std::size_t>::max());
    for (std::size_t offset = 0; offset < gpu.scores.size(); ++offset) {
        const auto id = gpu.scores[offset].pose_id;
        require(id < offsets.size(), "GPU pose_id outside the survivor pool: " + std::to_string(id));
        require(offsets[id] == std::numeric_limits<std::size_t>::max(),
                "Duplicate GPU survivor ID: " + std::to_string(id));
        offsets[id] = offset;
    }
    // Generation order, not geometric deduplication, identifies each occurrence.
    // Distinct alignments producing the same transform remain separate poses.
    for (std::size_t id = 1; id < offsets.size(); ++id)
        require(gpu.poses[offsets[id - 1]].alignment_index < gpu.poses[offsets[id]].alignment_index,
                "GPU survivor IDs do not preserve alignment order at pose " + std::to_string(id));
    return offsets;
}

void compare_top_ten(const std::vector<RankedPoseScore>& cpu,
                     const GpuDockingResult& gpu) {
    const std::size_t count = std::min<std::size_t>(10, cpu.size());
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto expected = cpu[rank].scored.pose_id;
        const auto actual = gpu.scores[rank].pose_id;
        if (actual != expected) {
            // Numerical near-ties are reported, not silently accepted or sorted
            // with a test-only epsilon comparator that changes production ranking.
            const auto found = std::find_if(cpu.begin(), cpu.end(),
                [actual](const RankedPoseScore& pose) { return pose.scored.pose_id == actual; });
            std::ostringstream message;
            message << std::setprecision(17) << "Hybrid Top-10 rank " << rank + 1
                    << ": expected pose_id " << expected << ", got " << actual
                    << "; CPU expected Hybrid=" << *cpu[rank].hybrid
                    << ", GPU actual Hybrid=" << *gpu.scores[rank].hybrid;
            if (found != cpu.end())
                message << ", CPU actual Hybrid=" << *found->hybrid
                        << ", CPU score gap=" << *cpu[rank].hybrid - *found->hybrid;
            message << ". A near-tie or equivalent transform still requires diagnosis; "
                       "exact score ties use original survivor ID.";
            throw std::runtime_error(message.str());
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::string phase = "input validation";
    try {
        require(argc == 2, "Usage: cpu_cuda_docking_test <complex_directory>");
        // Legacy CUDA timing switches are enabled by presence, even for "0".
        // This is a correctness test, so do not add their synchronization paths.
        require(unsetenv("DOCKING_STAGE_TIMINGS") == 0 &&
                unsetenv("DOCKING_KABSCH_TIMINGS") == 0,
                "Cannot disable legacy CUDA profiling");
        const auto kabsch = selected_method("DOCKING_KABSCH_METHOD", "svd", "svd", "quaternion");
        const auto queries = selected_method("DOCKING_QUERY_METHOD", "reference", "reference", "optimized");
        const auto config = configuration(argv[1]);
        std::cout << std::setprecision(17)
                  << "Protocol: complex=" << config.complex_id << ", Kabsch=" << kabsch
                  << ", GPU queries=" << queries << ", cutoff=ligand diameter, clearance=off"
                  << ", clash=" << config.operational_clash_threshold
                  << ", Hybrid lambda=0.3, Vinardo alpha=0.5, Fast prefilter=none\n";

        // These are independent production searches: the CPU receives no CUDA
        // alignments, candidates, transforms or survivor decisions.
        phase = "independent CPU pipeline";
        const auto cpu = run_pipeline(config);
        const auto& generated = cpu.pose_generation;
        require(cpu.pose_targets > 0, "CPU generated no protein targets");
        require(cpu.ligand_triplets > 0, "CPU generated no ligand triplets");
        require(generated.protein_triplet_records > 0, "CPU protein index is empty");
        require(generated.total_candidate_matches > 0, "CPU generated no candidate matches");
        require(generated.transforms_generated > 0, "CPU generated no alignments");
        require(generated.transforms_valid > 0, "CPU generated no valid transforms");
        require(!generated.surviving_poses.empty(), "CPU generated no clash survivors");
        require(std::isfinite(cpu.ligand_diameter) && cpu.ligand_diameter > 0.0,
                "CPU ligand diameter is not finite and positive");

        phase = "common protocol and independent preparation";
        auto ligand = load_ligand(config.ligand_path.string());
        PreparedLigand prepared;
        prepare_ligand(*ligand, prepared);
        same(prepared.diameter, cpu.ligand_diameter, "Independent ligand diameter");
        same(prepared.triplets.size(), cpu.ligand_triplets, "Independent ligand triplets");
        same(ligand->getNumAtoms(), cpu.ligand_atoms, "Independent ligand atom count");
        auto protein = prepare_protein(config); // nullopt: no clearance scan or pruning
        same(protein.targets.size(), cpu.pose_targets, "Independent protein targets");

        GpuDockingSettings settings;
        settings.cutoff = cpu.ligand_diameter;
        settings.ranking = {PoseScoreMethod::Hybrid, 0.5, 0.3};
        settings.rescoring_backend = GpuRescoringBackend::Cuda;
        settings.min_target_clearance = std::nullopt;
        settings.fast_prefilter = std::nullopt;
        settings.interaction_full_protein = false;
        phase = "independent CUDA target preparation";
        GpuDockingScreen screen(config, settings);
        same(screen.targets_before_pruning(), cpu.pose_targets, "GPU initial targets");
        same(screen.targets_after_pruning(), cpu.pose_targets, "GPU targets with pruning disabled");
        same(screen.index_records(), generated.protein_triplet_records, "Protein index records");
        phase = "independent CUDA search and rescoring";
        const auto gpu = screen.dock(*ligand, generated.surviving_poses.size());
        phase = "search counts and survivor identities";
        same(gpu.ligand_triplets, cpu.ligand_triplets, "Ligand triplets");
        same(gpu.diameter, cpu.ligand_diameter, "Ligand diameter");
        same(gpu.candidate_matches, generated.total_candidate_matches, "Candidate matches");
        same(gpu.alignments, generated.transforms_generated, "Generated alignments");
        same(gpu.valid_transforms, generated.transforms_valid, "Valid transforms");
        same(gpu.survivors, generated.surviving_poses.size(), "Clash survivors");
        same(gpu.candidate_pool, gpu.survivors, "Hybrid pool without Fast prefilter");
        const auto offsets = survivor_offsets(gpu);

        // The legacy CPU pipeline's Fast->WeightedInteraction ranking has different tie-breaks.
        // Reuse the common production rescoring/ranking API on its full survivor
        // pool; the original legacy ranking and its prefilter are not compared.
        phase = "CPU full-pool Hybrid rescoring";
        RescoringTarget receptor(config, false);
        const auto rescored = rescore_poses(generated.surviving_poses, *ligand, prepared,
            protein.grid, receptor.interaction_areas, receptor.vinardo, std::nullopt, false);
        same(rescored.poses.size(), generated.surviving_poses.size(), "CPU full rescoring pool");
        require(rescored.weighted_interaction_calculated && rescored.vinardo_calculated,
                "CPU did not calculate both WeightedInteraction and Vinardo");
        const auto ranked = rank_rescored_poses(rescored, settings.ranking, rescored.poses.size());
        same(ranked.size(), rescored.poses.size(), "CPU complete Hybrid ranking");
        std::vector<double> cpu_hybrid(rescored.poses.size(), std::numeric_limits<double>::quiet_NaN());
        for (const auto& row : ranked) {
            require(row.hybrid.has_value(), "CPU Hybrid score is missing");
            require(std::isnan(cpu_hybrid.at(row.scored.pose_id)), "Duplicate CPU ranked survivor ID");
            cpu_hybrid.at(row.scored.pose_id) = *row.hybrid;
        }

        phase = "per-survivor geometry, clash and scoring comparison";
        Errors errors;
        for (std::size_t id = 0; id < rescored.poses.size(); ++id) {
            const auto label = "pose " + std::to_string(id);
            const auto& expected = rescored.poses[id];
            same(expected.pose_id, id, label + " CPU survivor ID");
            const auto& actual = gpu.poses[offsets[id]];
            const auto& scores = gpu.scores[offsets[id]];
            require(actual.clash.valid && actual.fast_score.valid, label + ": invalid GPU score flags");
            require(scores.weighted_interaction && scores.vinardo && scores.vinardo_adjusted && scores.hybrid,
                    label + ": missing GPU rescoring fields");
            require(actual.alignment_index < gpu.alignments, label + ": alignment ID out of range");
            compare_transform(expected.pose.transform, actual.transform, *ligand, label, errors);
            same(actual.clash.occupied_atoms, expected.pose.occupied_atoms, label + " occupied atoms");
            same(actual.clash.total_atoms, ligand->getNumAtoms(), label + " clash total atoms");
            require(std::isfinite(actual.clash.fraction) && std::isfinite(expected.pose.clash_fraction),
                    label + ": nonfinite clash fraction");
            same(actual.clash.fraction, expected.pose.clash_fraction, label + " clash fraction");
            require(actual.clash.fraction <= config.operational_clash_threshold,
                    label + ": GPU survivor exceeds the operational clash threshold");
            same(actual.fast_score.total_areas, prepared.matching_areas.size(), label + " Fast total areas");
            same(actual.fast_score.compatible_areas, expected.pose.compatible_areas,
                 label + " Fast compatible areas");
            errors.close(actual.fast_score.weighted_score, expected.pose.fast_weighted_score,
                         1e-9, 0.0, errors.fast, label + " Fast score");
            compare_weighted_interaction(expected.pose.weighted_interaction_score, *scores.weighted_interaction, label + " WeightedInteraction");
            compare_vinardo(expected.vinardo, *scores.vinardo, label, errors);
            errors.close(*scores.vinardo_adjusted, vinardo_adjusted_score(expected.vinardo, 0.5),
                         1e-10, 1e-8, errors.vinardo, label + " Vinardo rep50");
            errors.close(*scores.hybrid, cpu_hybrid[id], 1e-11, 1e-9, errors.hybrid,
                         label + " Hybrid score");
        }
        phase = "Hybrid Top-10 order";
        compare_top_ten(ranked, gpu);
        std::cout << "PASS: candidates=" << gpu.candidate_matches << ", alignments=" << gpu.alignments
                  << ", valid=" << gpu.valid_transforms << ", survivors=" << gpu.survivors
                  << ", Hybrid top-" << std::min<std::size_t>(10, gpu.survivors)
                  << "; max errors: transform=" << errors.transform << ", atom=" << errors.point
                  << " A, Fast=" << errors.fast << ", Vinardo=" << errors.vinardo
                  << ", Hybrid=" << errors.hybrid << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu_cuda_docking_test FAILED during " << phase << ": " << error.what() << '\n';
        return 1;
    }
}
