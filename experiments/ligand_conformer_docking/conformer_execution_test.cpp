#include "conformer_docking.hpp"
#include "cuda_pose_rescoring.hpp"
#include "geometry.hpp"
#include "gpu_docking_output.hpp"
#include "molecule_io.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Results = std::vector<ConformerDockingResult>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class A, class B>
void same(A actual, B expected, const std::string& label) {
    if (actual != expected) {
        std::ostringstream message;
        message << label << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(message.str());
    }
}
void close(double actual, double expected, double absolute, double relative,
           const std::string& label) {
    const double allowed = absolute + relative * std::abs(expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > allowed) {
        std::ostringstream message;
        message << std::setprecision(17) << label << ": expected " << expected
                << ", got " << actual << ", tolerance=" << allowed;
        throw std::runtime_error(message.str());
    }
}
void optional_close(const std::optional<double>& actual, const std::optional<double>& expected,
                    const std::string& label) {
    same(actual.has_value(), expected.has_value(), label + " availability");
    if (actual) close(*actual, *expected, 1e-5, 0.0, label);
}
void rejects(const std::function<void()>& operation, const char* message) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}
void environment(const char* query) {
    for (const auto& [key, value] : {std::pair{"DOCKING_KABSCH_METHOD", "quaternion"},
         {"DOCKING_QUERY_METHOD", query}, {"DOCKING_QUERY_THREADS", "1"}, {"DOCKING_PROFILE_STAGES", "0"}})
        require(::setenv(key, value, 1) == 0, std::string("Cannot set ") + key);
    for (const auto* key : {"DOCKING_AREA_CROP_MARGIN_A", "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"})
        require(::unsetenv(key) == 0, std::string("Cannot unset ") + key);
}

void compare_interactions(const WeightedInteractionScore& actual,
                          const WeightedInteractionScore& expected, const std::string& label) {
    same(actual.total_interactions, expected.total_interactions, label + " contacts");
    same(actual.satisfied_ligand_areas, expected.satisfied_ligand_areas, label + " ligand coverage");
    same(actual.satisfied_protein_areas, expected.satisfied_protein_areas, label + " receptor coverage");
    same(actual.hydrophobic, expected.hydrophobic, label + " hydrophobic");
    same(actual.hydrogen_bonds, expected.hydrogen_bonds, label + " hydrogen bonds");
    same(actual.halogen_bonds, expected.halogen_bonds, label + " halogen bonds");
    same(actual.ionic, expected.ionic, label + " ionic");
    same(actual.cation_pi, expected.cation_pi, label + " cation-pi");
    same(actual.pi_stacking, expected.pi_stacking, label + " pi stacking");
    same(actual.metal, expected.metal, label + " metal");
    close(actual.weighted_score, expected.weighted_score, 0.0, 0.0, label + " reward");
}
void compare_vinardo(const VinardoLikeScore& actual, const VinardoLikeScore& expected,
                    const std::string& label) {
    const std::array<double, 5> a{actual.total, actual.gaussian, actual.repulsion,
                                actual.hydrophobic, actual.hydrogen_bond};
    const std::array<double, 5> b{expected.total, expected.gaussian, expected.repulsion,
                                expected.hydrophobic, expected.hydrogen_bond};
    for (std::size_t i = 0; i < a.size(); ++i)
        close(a[i], b[i], 1e-10, 1e-8, label + " Vinardo component " + std::to_string(i));
}
void compare_results(const Results& actual, const Results& expected,
                     const RunConfig& config, const std::string& mode) {
    same(actual.size(), expected.size(), mode + " conformer count");
    for (std::size_t conformer = 0; conformer < expected.size(); ++conformer) {
        const auto& a = actual[conformer];
        const auto& b = expected[conformer];
        const auto label = mode + " conformer " + std::to_string(conformer);
        same(a.triplets, b.triplets, label + " triplets");
        same(a.matches, b.matches, label + " matches");
        same(a.alignments, b.alignments, label + " alignments");
        same(a.valid_transforms, b.valid_transforms, label + " valid transforms");
        same(a.survivors, b.survivors, label + " survivors");
        same(a.poses.size(), a.survivors, label + " complete survivor pool");
        require(a.rescored_ids == b.rescored_ids, label + ": selected IDs differ");
        same(a.rescored_ids.size(), a.survivors, label + " no Fast prefilter");
        close(a.diameter, b.diameter, 0.0, 0.0, label + " diameter");
        optional_close(a.oracle_all_survivors, b.oracle_all_survivors, label + " all-pose Oracle");
        optional_close(a.oracle_after_prefilter, b.oracle_after_prefilter, label + " selected Oracle");
        for (std::size_t pose = 0; pose < b.poses.size(); ++pose) {
            const auto& x = a.poses[pose];
            const auto& y = b.poses[pose];
            const auto row = label + " pose " + std::to_string(pose);
            same(x.conformer_id, y.conformer_id, row + " conformer identity");
            same(x.pose_id, pose, row + " local identity");
            same(x.scored.pose_id, pose, row + " raw score identity");
            same(x.alignment_index, y.alignment_index, row + " alignment identity");
            if (pose) require(x.alignment_index > a.poses[pose - 1].alignment_index,
                              row + " lost alignment order or occurrence");
            for (std::size_t i = 0; i < 9; ++i)
                close(x.scored.pose.transform.rotation[i], y.scored.pose.transform.rotation[i],
                      1e-6, 0.0, row + " rotation");
            for (std::size_t i = 0; i < 3; ++i)
                close(x.scored.pose.transform.translation[i], y.scored.pose.transform.translation[i],
                      1e-6, 0.0, row + " translation");
            same(x.scored.pose.occupied_atoms, y.scored.pose.occupied_atoms, row + " occupied atoms");
            close(x.scored.pose.clash_fraction, y.scored.pose.clash_fraction, 0.0, 0.0, row + " clash");
            require(x.scored.pose.clash_fraction <= config.operational_clash_threshold,
                    row + " exceeds clash threshold");
            close(x.scored.pose.fast_weighted_score, y.scored.pose.fast_weighted_score,
                  1e-9, 0.0, row + " Fast");
            compare_interactions(x.scored.pose.weighted_interaction_score,
                                 y.scored.pose.weighted_interaction_score, row);
            compare_vinardo(x.scored.vinardo, y.scored.vinardo, row);
            close(x.rmsd, y.rmsd, 1e-5, 0.0, row + " direct heavy-atom RMSD");
        }
    }
}

void compare_centers(const Results& actual, const Results& expected,
                     const std::vector<RDKit::RWMol>& molecules, const RDKit::ROMol& reference,
                     const std::string& mode) {
    const auto reference_center = heavy_atom_centroid(reference);
    for (std::size_t conformer = 0; conformer < molecules.size(); ++conformer) {
        const auto center = heavy_atom_centroid(molecules[conformer]);
        for (std::size_t pose = 0; pose < actual[conformer].poses.size(); ++pose) {
            const auto& a = actual[conformer].poses[pose];
            const auto& b = expected[conformer].poses[pose];
            const double actual_distance = pose_center_distance(center, a.scored.pose.transform, reference_center);
            const double expected_distance = pose_center_distance(center, b.scored.pose.transform, reference_center);
            close(actual_distance, expected_distance, 1e-5, 0.0,
                  mode + " conformer " + std::to_string(conformer) + " pose " +
                  std::to_string(pose) + " optional center diagnostic");
        }
    }
}

std::vector<EnsemblePose> population(const Results& results, bool baseline) {
    std::vector<EnsemblePose> poses;
    for (std::size_t i = 0; i < results.size(); ++i)
        if ((i == 0) == baseline)
            poses.insert(poses.end(), results[i].poses.begin(), results[i].poses.end());
    return poses;
}
void compare_ranking(const std::vector<EnsemblePose>& actual,
                     const std::vector<EnsemblePose>& expected,
                     const std::vector<RankedEnsemblePose>& a,
                     const std::vector<RankedEnsemblePose>& b,
                     PoseScoreMethod method, bool oracle, bool full_order,
                     const std::string& label) {
    same(a.size(), b.size(), label + " ranked multiplicity");
    const double absolute = oracle ? 1e-5 : method == PoseScoreMethod::Fast ? 1e-9 :
        method == PoseScoreMethod::WeightedInteraction ? 0.0 : method == PoseScoreMethod::Vinardo ? 1e-10 : 1e-11;
    const double relative = oracle || method == PoseScoreMethod::Fast || method == PoseScoreMethod::WeightedInteraction
        ? 0.0 : method == PoseScoreMethod::Vinardo ? 1e-8 : 1e-9;
    std::vector<double> by_source(a.size());
    std::vector<double> expected_by_source(b.size());
    for (const auto& row : a) by_source.at(row.source_index) = row.score;
    for (const auto& row : b) {
        expected_by_source.at(row.source_index) = row.score;
        close(by_source.at(row.source_index), row.score, absolute, relative,
              label + " source score " + std::to_string(row.source_index));
    }
    const auto compared = full_order ? b.size() : std::min<std::size_t>(20, b.size());
    for (std::size_t i = 0; i < compared; ++i) {
        if (a[i].source_index != b[i].source_index) {
            std::ostringstream message;
            message << std::setprecision(17) << label << " rank " << i + 1
                    << ": expected source " << b[i].source_index << ", got " << a[i].source_index
                    << "; expected score=" << b[i].score << ", actual score=" << a[i].score
                    << ", reference score gap=" << b[i].score - expected_by_source.at(a[i].source_index)
                    << ". Near-ties require diagnosis; the test does not change production ordering.";
            throw std::runtime_error(message.str());
        }
        same(a[i].local_rank, b[i].local_rank, label + " local rank");
        same(a[i].global_rank, b[i].global_rank, label + " global rank");
    }
    const auto summary_a = summarize_ensemble_ranking(actual, a);
    const auto summary_b = summarize_ensemble_ranking(expected, b);
    optional_close(summary_a.oracle_rmsd, summary_b.oracle_rmsd, label + " Oracle");
    for (std::size_t i = 0; i < ensemble_summary_top_k.size(); ++i) {
        optional_close(summary_a.top_k_best_rmsd[i], summary_b.top_k_best_rmsd[i],
                       label + " Top-" + std::to_string(ensemble_summary_top_k[i]) + " RMSD");
        if (summary_a.top_k_best_rmsd[i])
            same(*summary_a.top_k_best_rmsd[i] < 2.0, *summary_b.top_k_best_rmsd[i] < 2.0,
                 label + " success below 2 A");
    }
}
void compare_populations(const Results& actual, const Results& expected, const std::string& mode,
                         bool full_order) {
    for (const bool baseline : {true, false}) {
        const auto a = population(actual, baseline);
        const auto b = population(expected, baseline);
        for (const auto method : {PoseScoreMethod::Fast, PoseScoreMethod::WeightedInteraction,
                                  PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid})
            compare_ranking(a, b, rank_ensemble_poses(a, method), rank_ensemble_poses(b, method),
                            method, false, full_order,
                            mode + (baseline ? " baseline " : " global ") + pose_score_method_name(method));
        if (baseline) continue;
        for (const auto& pose : a) require(pose.conformer_id != 0, "Baseline entered generated population");
        compare_ranking(a, b, rank_ensemble_oracle(a), rank_ensemble_oracle(b),
                        PoseScoreMethod::Fast, true, true, mode + " RMSD Oracle");
        auto changed_rmsd = a;
        for (std::size_t i = 0; i < changed_rmsd.size(); ++i) changed_rmsd[i].rmsd = i % 2 ? 1000.0 : 0.0;
        const auto ordinary = rank_ensemble_poses(a, PoseScoreMethod::Hybrid);
        const auto changed = rank_ensemble_poses(changed_rmsd, PoseScoreMethod::Hybrid);
        for (std::size_t i = 0; i < ordinary.size(); ++i) {
            same(changed[i].source_index, ordinary[i].source_index, mode + " RMSD-independent Hybrid identity");
            close(changed[i].score, ordinary[i].score, 0.0, 0.0, mode + " RMSD-independent Hybrid score");
        }
        auto contaminated = a;
        contaminated.push_back(actual.front().poses.front());
        rejects([&] { rank_ensemble_oracle(contaminated); }, "Oracle accepted crystallographic baseline");
    }
}

void edge_cases(const std::vector<ConformerDockingInput>& inputs, const RDKit::RWMol& reference,
                const PreparedGpuTarget& target, const RescoringTarget& scoring, const RunConfig& config,
                const CudaRescoringTarget& device_scoring) {
    ConformerExecutionOptions options;
    require(dock_conformers({}, reference, target, scoring, config, std::nullopt, options).empty(),
            "Empty input produced results");
    rejects([&] { dock_conformers({inputs.front(), inputs.front()}, reference, target, scoring,
                                  config, std::nullopt, options); }, "Duplicate conformer IDs were accepted");
    rejects([&] { dock_conformers({{1, nullptr}}, reference, target, scoring, config,
                                  std::nullopt, options); }, "Missing molecule owner was accepted");
    auto invalid = reference;
    invalid.getAtomWithIdx(0)->setAtomicNum(reference.getAtomWithIdx(0)->getAtomicNum() == 6 ? 7 : 6);
    rejects([&] { dock_conformers({{1, &invalid}}, reference, target, scoring, config,
                                  std::nullopt, options); }, "Invalid atom mapping was accepted");

    // Indexed sides below 0.001 A are always rejected, so this target has a
    // genuinely empty device index without modifying an existing target owner.
    const PreparedGpuTarget empty_target(config, 0.0005);
    same(empty_target.index_records, std::size_t{0}, "Empty-index fixture");
    options.rescoring_backend = GpuRescoringBackend::Cuda;
    options.cpu_gpu_overlap = true;
    for (const bool batch : {false, true}) {
        options.gpu_batch = batch;
        options.batch_size = 2;
        const auto empty = dock_conformers(inputs, reference, empty_target, scoring, config,
                                          std::nullopt, options, nullptr, &device_scoring);
        same(empty.size(), inputs.size(), "Empty-index conformer count");
        for (const auto& result : empty) {
            require(result.matches == 0 && result.alignments == 0 && result.valid_transforms == 0 &&
                    result.survivors == 0 && result.poses.empty() && result.rescored_ids.empty() &&
                    !result.oracle_all_survivors && !result.oracle_after_prefilter,
                    "Empty index produced poses, scores or an RMSD metric");
        }
        require(rank_ensemble_oracle(population(empty, false)).empty(), "Empty ensemble produced an Oracle");
    }
}
} // namespace

int main(int argc, char** argv) {
    std::string phase = "input";
    try {
        require(argc == 2, "Usage: conformer_execution_test <complex_directory>");
        environment("reference");
        auto config = parse_run_config(argc, argv);
        config.output_mode = OutputMode::Quiet;
        const auto reference = load_ligand(config.ligand_path.string());
        std::vector<RDKit::RWMol> molecules;
        molecules.reserve(4);
        for (std::size_t i = 0; i < 4; ++i) molecules.emplace_back(*reference);
        for (unsigned int atom = 0; atom < reference->getNumAtoms(); ++atom) {
            const auto& p = reference->getConformer().getAtomPos(atom);
            molecules[1].getConformer().setAtomPos(atom, {p.x + 2.0, p.y - 1.0, p.z + 0.5});
            molecules[2].getConformer().setAtomPos(atom, {-p.y - 3.0, p.x + 2.0, p.z - 1.0});
        }
        unsigned int id = 0;
        while (id < reference->getNumAtoms() && reference->getAtomWithIdx(id)->getAtomicNum() <= 1) ++id;
        require(id < reference->getNumAtoms(), "Fixture has no heavy atoms");
        auto p = molecules[3].getConformer().getAtomPos(id);
        p.x += 0.037; p.y -= 0.021;
        molecules[3].getConformer().setAtomPos(id, p);
        std::vector<ConformerDockingInput> inputs;
        for (std::size_t i = 0; i < molecules.size(); ++i) inputs.push_back({i, &molecules[i]});

        phase = "shared receptor";
        const PreparedGpuTarget target(config, 24.0);
        const RescoringTarget scoring(config);
        const CudaRescoringTarget device_scoring(scoring, {});
        require(target.index_records > 0, "Integration complex has an empty protein index");
        Results expected;
        phase = "legacy CPU-scored full survivor pools";
        for (std::size_t i = 0; i < molecules.size(); ++i) {
            expected.push_back(dock_conformer(i, molecules[i], *reference, target, scoring, config, std::nullopt));
            require(expected.back().matches > 0 && expected.back().valid_transforms > 0 &&
                    expected.back().survivors > 0, "Integration fixture does not exercise real pose generation");
        }
        std::optional<Results> cuda_reference;
        for (const char* query : {"reference", "optimized"}) {
            environment(query);
            for (const auto backend : {GpuRescoringBackend::Cpu, GpuRescoringBackend::Cuda}) {
                for (const std::size_t batch_size : {0u, 1u, 2u, 4u}) {
                    if (backend == GpuRescoringBackend::Cpu && batch_size) continue;
                    for (const bool overlap : {false, true}) {
                        ConformerExecutionOptions options;
                        options.gpu_batch = batch_size != 0;
                        options.batch_size = batch_size ? batch_size : 1;
                        options.cpu_gpu_overlap = overlap;
                        options.rescoring_backend = backend;
                        options.query_method = query;
                        phase = std::string(query) + (backend == GpuRescoringBackend::Cpu ? "/CPU" : "/CUDA") +
                            "/batch=" + std::to_string(batch_size) + (overlap ? "/overlap=on" : "/overlap=off");
                        ConformerExecutionTimings timings;
                        const auto actual = dock_conformers(inputs, *reference, target, scoring, config,
                            std::nullopt, options, &timings, &device_scoring);
                        compare_results(actual, expected, config, phase);
                        compare_centers(actual, expected, molecules, *reference, phase);
                        // Cross-backend checks retain strict Top-20 identities and
                        // compare every score; full order is checked per backend.
                        compare_populations(actual, expected, phase, backend == GpuRescoringBackend::Cpu);
                        if (backend == GpuRescoringBackend::Cuda) {
                            if (!cuda_reference) cuda_reference = actual;
                            else compare_populations(actual, *cuda_reference, phase + "/CUDA-reference", true);
                        }
                        require(timings.wall_s > 0.0 && timings.queue_capacity > 0 &&
                                timings.queue_high_water_records <= timings.queue_capacity,
                                phase + ": invalid prefetch timing/capacity metadata");
                        std::cout << "PASS " << phase << '\n';
                    }
                }
            }
        }
        phase = "empty input, invalid mapping and empty device index";
        environment("reference");
        edge_cases(inputs, *reference, target, scoring, config, device_scoring);
        std::cout << "Conformer execution agrees on full survivor pools, raw scores, global Hybrid and RMSD Oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conformer_execution_test failed during " << phase << ": " << error.what() << '\n';
        return 1;
    }
}
