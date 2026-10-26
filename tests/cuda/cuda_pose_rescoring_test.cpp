#include "cuda_weighted_interaction_scoring.hpp"
#include "cuda_pose_ranking.hpp"
#include "cuda_vinardo_scoring.hpp"
#include "geometry.hpp"
#include "interaction_scoring.hpp"
#include "pose_score_ranking.hpp"

#include <cuda_runtime_api.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T> void upload(const std::vector<T>& source, T*& destination) {
    if (source.empty()) return;
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&destination), source.size() * sizeof(T)),
                 "test allocation");
    cuda_require(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                            cudaMemcpyHostToDevice), "test upload");
}

template<class T> std::vector<T> download(const T* source, std::size_t count) {
    std::vector<T> result(count);
    if (count) cuda_require(cudaMemcpy(result.data(), source, count * sizeof(T),
                                       cudaMemcpyDeviceToHost), "test download");
    return result;
}

template<class T> void replace(const std::vector<T>& source, T* destination) {
    if (!source.empty()) cuda_require(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                                               cudaMemcpyHostToDevice), "test replacement");
}

struct Errors {
    double vinardo_absolute{}, hybrid_absolute{};

    void close(double gpu, double cpu, double relative, double absolute,
               double& maximum, const char* field) {
        require(std::isfinite(gpu) && std::isfinite(cpu), std::string("Nonfinite ") + field);
        const double error = std::abs(gpu - cpu);
        maximum = std::max(maximum, error);
        require(error <= absolute + relative * std::abs(cpu),
                std::string(field) + " CPU/GPU tolerance exceeded: " + std::to_string(error));
    }
};

void compare_weighted_interaction(const WeightedInteractionScore& gpu, const WeightedInteractionScore& cpu) {
    require(gpu.total_interactions == cpu.total_interactions &&
            gpu.satisfied_ligand_areas == cpu.satisfied_ligand_areas &&
            gpu.satisfied_protein_areas == cpu.satisfied_protein_areas &&
            gpu.hydrophobic == cpu.hydrophobic && gpu.hydrogen_bonds == cpu.hydrogen_bonds &&
            gpu.halogen_bonds == cpu.halogen_bonds && gpu.ionic == cpu.ionic &&
            gpu.cation_pi == cpu.cation_pi && gpu.pi_stacking == cpu.pi_stacking &&
            gpu.metal == cpu.metal, "WeightedInteraction interaction counts differ");
    require(gpu.weighted_score == cpu.weighted_score, "WeightedInteraction score differs");
}

void compare_vinardo(const VinardoLikeScore& gpu, const VinardoLikeScore& cpu, Errors& errors) {
    const std::array<double, 5> actual{gpu.total, gpu.gaussian, gpu.repulsion,
                                      gpu.hydrophobic, gpu.hydrogen_bond};
    const std::array<double, 5> expected{cpu.total, cpu.gaussian, cpu.repulsion,
                                        cpu.hydrophobic, cpu.hydrogen_bond};
    for (std::size_t i = 0; i < actual.size(); ++i)
        errors.close(actual[i], expected[i], 1e-8, 1e-10, errors.vinardo_absolute,
                     "Vinardo component");
}

interaction::interaction_area area(interaction::interaction_smarts_pattern pattern,
                                    interaction::point3D center,
                                    bool aromatic = false) {
    interaction::interaction_area result{};
    result.pattern = pattern;
    result.pattern_class = aromatic ? interaction::interaction_class_pattern::Aromatic :
                                    interaction::interaction_class_pattern::Hydrophobic;
    result.centroid = center;
    result.num_atoms = aromatic ? 3 : 1;
    result.x[0] = center.x; result.y[0] = center.y; result.z[0] = center.z;
    return result;
}

void set_atom(interaction::interaction_area& area, std::size_t index,
              interaction::point3D point) {
    area.x[index] = point.x; area.y[index] = point.y; area.z[index] = point.z;
    area.num_atoms = std::max(area.num_atoms, index + 1);
}

RDKit::RWMol molecule(const std::vector<unsigned int>& elements,
                      const std::vector<RDGeom::Point3D>& positions) {
    require(elements.size() == positions.size(), "Invalid molecule fixture");
    RDKit::RWMol result;
    auto conformer = std::make_unique<RDKit::Conformer>(static_cast<unsigned int>(elements.size()));
    for (unsigned int i = 0; i < elements.size(); ++i) {
        auto* atom = new RDKit::Atom(elements[i]);
        atom->setNoImplicit(true);
        result.addAtom(atom, true, true);
        conformer->setAtomPos(i, positions[i]);
    }
    result.addConformer(conformer.release(), true);
    // Vinardo's shared typing queries total H counts for N/O atoms.
    result.updatePropertyCache(false);
    return result;
}

RigidTransform host_transform(const GpuRigidTransform& source) {
    RigidTransform result;
    std::copy(std::begin(source.rotation), std::end(source.rotation), result.rotation.begin());
    std::copy(std::begin(source.translation), std::end(source.translation), result.translation.begin());
    return result;
}

struct Fixture {
    RDKit::RWMol ligand = molecule({6, 6, 7, 1},
        {{0.25, 0.25, 0.25}, {1.25, 0.25, 0.25}, {0.25, 2.25, 0.25}, {0.25, 3.25, 0.25}});
    RDKit::RWMol receptor = molecule({6, 8, 9, 30},
        {{3.25, 0.25, 0.25}, {0.25, 4.25, 0.25}, {6.25, 0.25, 0.25}, {100.0, 0.0, 0.0}});
    VinardoLikeScorer vinardo{receptor};
    std::vector<interaction::interaction_area> ligand_areas, protein_areas;
    std::vector<GpuRigidTransform> transforms;
    std::vector<GpuFastScore> fast;
    std::vector<GpuClashScore> clash;
    std::vector<std::uint32_t> alignments;
    GpuPosePipelineDeviceResult pool;
    GpuWeightedInteractionAreasDevice device_protein{}, device_ligand{};
    GpuVinardoAtomsDevice device_receptor_atoms{}, device_ligand_atoms{};

    Fixture() {
        using Pattern = interaction::interaction_smarts_pattern;
        const auto add = [&](interaction::interaction_area ligand_area,
                             interaction::interaction_area protein_area) {
            ligand_areas.push_back(ligand_area); protein_areas.push_back(protein_area);
        };
        add(area(Pattern::Hydrophobic, {0, 0, 0}), area(Pattern::Hydrophobic, {3, 0, 0}));
        // Multiple protein contacts reward the same ligand feature only once.
        protein_areas.push_back(area(Pattern::Hydrophobic, {4.5f, 0, 0}));
        protein_areas.push_back(area(Pattern::Hydrophobic,
            {std::nextafter(4.5f, std::numeric_limits<float>::infinity()), 0, 0}));
        auto donor = area(Pattern::Hydrogen_donor, {20, 0, 0});
        set_atom(donor, 1, {21, 0, 0});
        add(donor, area(Pattern::Hydrogen_acceptor, {23, 0, 0}));
        donor = area(Pattern::Halogen_donor, {40, 0, 0});
        set_atom(donor, 1, {41, 0, 0});
        auto acceptor = area(Pattern::Halogen_acceptor, {43, 0, 0});
        set_atom(acceptor, 1, {43, 1, 0});
        add(donor, acceptor);
        add(area(Pattern::Anion, {60, 0, 0}), area(Pattern::Cation, {63, 0, 0}));
        auto ring = area(Pattern::Aromatic_ring_6, {80, 0, 0}, true);
        set_atom(ring, 0, {81, 0, 0}); set_atom(ring, 1, {80, 1, 0});
        add(ring, area(Pattern::Cation, {80, 0, 3}));
        add(area(Pattern::Metal, {100, 0, 0}), area(Pattern::Chelated, {102.5f, 0, 0}));
        ring = area(Pattern::Aromatic_ring_6, {120, 0, 0}, true);
        set_atom(ring, 0, {121, 0, 0}); set_atom(ring, 1, {120, 1, 0});
        auto other_ring = ring;
        other_ring.centroid.z = 3; other_ring.z[0] = other_ring.z[1] = 3;
        add(ring, other_ring);
        // A zero plane normal must retain the dependency's NaN behavior.
        ring = area(Pattern::Aromatic_ring_5, {140, 0, 0}, true);
        set_atom(ring, 1, ring.centroid);
        add(ring, area(Pattern::Cation, {140, 0, 3}));

        const std::array<double, 6> shifts{0.0, 1.75, -4.0, 0.0, 10.0, 0.0};
        const std::array<double, 6> fast_scores{31.5, 20.0, 31.5, 31.5, 1.0, 11.0};
        for (std::size_t i = 0; i < shifts.size(); ++i) {
            GpuRigidTransform transform{};
            transform.rotation[0] = transform.rotation[4] = transform.rotation[8] = 1.0;
            transform.translation[0] = shifts[i]; transform.valid = 1;
            if (i == 5) {
                transform.rotation[0] = transform.rotation[4] = 0.0;
                transform.rotation[1] = -1.0; transform.rotation[3] = 1.0;
            }
            transforms.push_back(transform);
            GpuFastScore score{};
            score.valid = 1; score.weighted_score = fast_scores[i];
            score.total_areas = 10; score.compatible_areas = static_cast<std::uint32_t>(i + 1);
            fast.push_back(score);
            GpuClashScore clash_score{};
            clash_score.valid = 1; clash_score.fraction = 0.005 * i;
            clash_score.occupied_atoms = static_cast<std::uint32_t>(i);
            clash.push_back(clash_score);
            alignments.push_back(static_cast<std::uint32_t>(100 + 3 * i));
        }
    }

    void upload_data() {
        upload(transforms, pool.survivors.transforms);
        upload(clash, pool.survivors.clash_scores);
        upload(alignments, pool.survivors.alignment_indices);
        upload(fast, pool.fast_scores.scores);
        pool.survivors.count = pool.fast_scores.count = static_cast<std::uint32_t>(transforms.size());
        require(cuda_upload_weighted_interaction_areas(make_gpu_weighted_interaction_areas(protein_areas), device_protein),
                "WeightedInteraction receptor upload failed");
        require(cuda_upload_weighted_interaction_areas(make_gpu_weighted_interaction_areas(ligand_areas), device_ligand),
                "WeightedInteraction ligand upload failed");
        require(cuda_upload_vinardo_atoms(make_gpu_vinardo_receptor_atoms(vinardo), device_receptor_atoms),
                "Vinardo receptor upload failed");
        require(cuda_upload_vinardo_atoms(make_gpu_vinardo_ligand_atoms(ligand), device_ligand_atoms),
                "Vinardo ligand upload failed");
    }

    ~Fixture() {
        cuda_free_vinardo_atoms(device_ligand_atoms);
        cuda_free_vinardo_atoms(device_receptor_atoms);
        cuda_free_weighted_interaction_areas(device_ligand);
        cuda_free_weighted_interaction_areas(device_protein);
    }

    GpuRigidTransformArrayDevice device_transforms() const {
        return {pool.survivors.transforms, pool.survivors.count};
    }

    PoseRescoringResult cpu_scores(const std::vector<std::uint32_t>& ids,
                                  const WeightedInteractionWeights& weights = {}) const {
        PoseRescoringResult result;
        result.input_pose_count = transforms.size();
        result.weighted_interaction_calculated = result.vinardo_calculated = true;
        for (auto id : ids) {
            RescoredPose row;
            row.pose_id = id;
            row.pose.transform = host_transform(transforms[id]);
            row.pose.fast_weighted_score = fast[id].weighted_score;
            row.pose.occupied_atoms = clash[id].occupied_atoms;
            row.pose.clash_fraction = clash[id].fraction;
            row.pose.rmsd = id == 0 ? 100.0 : std::numeric_limits<double>::quiet_NaN();
            row.pose.weighted_interaction_score = score_weighted_interactions(protein_areas,
                transform_interaction_areas(ligand_areas, row.pose.transform), weights);
            row.vinardo = vinardo.score(ligand, row.pose.transform);
            result.poses.push_back(row);
        }
        return result;
    }
};

struct ScoredPool {
    GpuRescoringSelection selection{};
    GpuWeightedInteractionScoresDevice weighted_interaction{};
    GpuVinardoScoresDevice vinardo{};
    ~ScoredPool() {
        cuda_free_vinardo_scores(vinardo);
        cuda_free_weighted_interaction_scores(weighted_interaction);
        cuda_free_rescoring_selection(selection);
    }
};

void compare_pose(const GpuRankedRescoredPose& gpu, const RescoredPose& cpu,
                  const Fixture& fixture) {
    require(gpu.pose_id == cpu.pose_id, "Ranked pose IDs differ");
    require(gpu.pose.alignment_index == fixture.alignments[cpu.pose_id], "Alignment identity changed");
    for (std::size_t i = 0; i < 9; ++i)
        require(gpu.pose.transform.rotation[i] == cpu.pose.transform.rotation[i], "Rotation changed");
    for (std::size_t i = 0; i < 3; ++i)
        require(gpu.pose.transform.translation[i] == cpu.pose.transform.translation[i], "Translation changed");
    require(gpu.pose.fast_score.weighted_score == cpu.pose.fast_weighted_score &&
            gpu.pose.fast_score.compatible_areas == fixture.fast[cpu.pose_id].compatible_areas,
            "Rescoring changed precomputed Fast scores or diagnostics");
    require(gpu.pose.clash.occupied_atoms == cpu.pose.occupied_atoms &&
            gpu.pose.clash.fraction == cpu.pose.clash_fraction, "Clash data changed");
}

void test_pool(Fixture& fixture, std::uint32_t prefilter, Errors& errors) {
    ScoredPool gpu;
    require(cuda_select_rescoring_pool(fixture.pool.fast_scores, prefilter, gpu.selection),
            "GPU prefilter failed");
    const auto ids = download(gpu.selection.pose_ids, gpu.selection.count);
    std::vector<std::uint32_t> expected(fixture.transforms.size());
    std::iota(expected.begin(), expected.end(), 0u);
    if (prefilter && prefilter < expected.size()) {
        std::sort(expected.begin(), expected.end(), [&](auto a, auto b) {
            if (fixture.fast[a].weighted_score != fixture.fast[b].weighted_score)
                return fixture.fast[a].weighted_score > fixture.fast[b].weighted_score;
            return a < b;
        });
        expected.resize(prefilter); std::sort(expected.begin(), expected.end());
    }
    require(ids == expected, "Fast prefilter membership or survivor order differs");
    require(prefilter || ids.size() == fixture.transforms.size(), "All-pose scoring lost survivors");
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        fixture.device_transforms(), gpu.selection.pose_ids, gpu.selection.count,
        make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), gpu.weighted_interaction), "GPU WeightedInteraction scoring failed");
    require(cuda_score_vinardo_device(fixture.device_receptor_atoms, fixture.device_ligand_atoms,
        fixture.device_transforms(), gpu.selection.pose_ids, gpu.selection.count, gpu.vinardo),
        "GPU Vinardo scoring failed");
    const auto cpu = fixture.cpu_scores(ids);
    const auto weighted_interaction = download(gpu.weighted_interaction.scores, gpu.weighted_interaction.count);
    const auto vinardo = download(gpu.vinardo.scores, gpu.vinardo.count);
    require(weighted_interaction.size() == ids.size() && vinardo.size() == ids.size(), "Score pool was truncated");
    for (std::size_t i = 0; i < ids.size(); ++i) {
        compare_weighted_interaction(weighted_interaction_score_from_gpu(weighted_interaction[i]), cpu.poses[i].pose.weighted_interaction_score);
        compare_vinardo(vinardo_score_from_gpu(vinardo[i]), cpu.poses[i].vinardo, errors);
    }
    for (double alpha : {1.0, 0.5}) {
        for (PoseScoreMethod method : {PoseScoreMethod::WeightedInteraction, PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid}) {
            for (double lambda : {0.0, 0.3, 1.0}) {
                if (method != PoseScoreMethod::Hybrid && lambda != 0.3) continue;
                const PoseScoreRankingOptions options{method, alpha, lambda};
                for (std::uint32_t top_k : {1u, static_cast<std::uint32_t>(fixture.transforms.size())}) {
                    const auto cpu_ranked = rank_rescored_poses(cpu, options, top_k);
                    auto changed_rmsd = cpu;
                    for (auto& row : changed_rmsd.poses)
                        row.pose.rmsd = row.pose_id % 2 ? -1000.0 : std::numeric_limits<double>::infinity();
                    const auto rmsd_ranked = rank_rescored_poses(changed_rmsd, options, top_k);
                    for (std::size_t i = 0; i < cpu_ranked.size(); ++i)
                        require(cpu_ranked[i].scored.pose_id == rmsd_ranked[i].scored.pose_id,
                                "RMSD changed candidate selection or ranking");
                    std::vector<GpuRankedRescoredPose> ranked;
                    require(cuda_rank_rescored_poses(fixture.pool, gpu.selection, gpu.weighted_interaction,
                        gpu.vinardo, static_cast<int>(method), alpha, lambda, top_k, ranked),
                        "GPU ranking failed");
                    require(ranked.size() == cpu_ranked.size(), "Top-k size differs");
                    for (std::size_t i = 0; i < ranked.size(); ++i) {
                        compare_pose(ranked[i], cpu_ranked[i].scored, fixture);
                        if (method != PoseScoreMethod::Vinardo)
                            compare_weighted_interaction(weighted_interaction_score_from_gpu(ranked[i].weighted_interaction), cpu_ranked[i].scored.pose.weighted_interaction_score);
                        if (method != PoseScoreMethod::WeightedInteraction) {
                            compare_vinardo(vinardo_score_from_gpu(ranked[i].vinardo), cpu_ranked[i].scored.vinardo, errors);
                            errors.close(ranked[i].adjusted, *cpu_ranked[i].vinardo_adjusted,
                                1e-8, 1e-10, errors.vinardo_absolute, "Adjusted Vinardo");
                        }
                        if (method == PoseScoreMethod::Hybrid)
                            errors.close(ranked[i].hybrid, *cpu_ranked[i].hybrid, 1e-9, 1e-11,
                                         errors.hybrid_absolute, "Hybrid");
                    }
                }
            }
        }
    }
    const auto fast_after = download(fixture.pool.fast_scores.scores, fixture.fast.size());
    for (std::size_t i = 0; i < fast_after.size(); ++i)
        require(fast_after[i].weighted_score == fixture.fast[i].weighted_score &&
                fast_after[i].compatible_areas == fixture.fast[i].compatible_areas &&
                fast_after[i].valid == fixture.fast[i].valid, "Scorers modified resident Fast data");
}

void test_errors_and_constants(Fixture& fixture, Errors& errors) {
    ScoredPool gpu;
    require(cuda_select_rescoring_pool(fixture.pool.fast_scores, 0, gpu.selection), "Pool selection failed");
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        fixture.device_transforms(), gpu.selection.pose_ids, gpu.selection.count,
        make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), gpu.weighted_interaction), "WeightedInteraction setup failed");
    require(cuda_score_vinardo_device(fixture.device_receptor_atoms, fixture.device_ligand_atoms,
        fixture.device_transforms(), gpu.selection.pose_ids, gpu.selection.count, gpu.vinardo), "Vinardo setup failed");
    const auto original_weighted_interaction = download(gpu.weighted_interaction.scores, gpu.weighted_interaction.count);
    const auto original_vinardo = download(gpu.vinardo.scores, gpu.vinardo.count);
    std::vector<GpuRankedRescoredPose> ranked;
    const auto ids = download(gpu.selection.pose_ids, gpu.selection.count);
    const auto cpu_original = fixture.cpu_scores(ids);
    const auto cpu_rejects = [](const PoseRescoringResult& input, PoseScoreMethod method) {
        try { rank_rescored_poses(input, {method, 0.5, 0.3}, 1); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };

    auto bad_weighted_interaction = original_weighted_interaction;
    bad_weighted_interaction.back().weighted_score = std::numeric_limits<double>::quiet_NaN();
    replace(bad_weighted_interaction, gpu.weighted_interaction.scores);
    require(!cuda_rank_rescored_poses(fixture.pool, gpu.selection, gpu.weighted_interaction, gpu.vinardo,
        1, 1.0, 0.3, 1, ranked), "Nonfinite WeightedInteraction outside top-k was ignored");
    auto bad_cpu = cpu_original;
    bad_cpu.poses.back().pose.weighted_interaction_score.weighted_score = std::numeric_limits<double>::quiet_NaN();
    require(cpu_rejects(bad_cpu, PoseScoreMethod::WeightedInteraction), "CPU WeightedInteraction accepted the same nonfinite row");
    bad_weighted_interaction = original_weighted_interaction; bad_weighted_interaction.back().valid = 0;
    replace(bad_weighted_interaction, gpu.weighted_interaction.scores);
    require(!cuda_rank_rescored_poses(fixture.pool, gpu.selection, gpu.weighted_interaction, gpu.vinardo,
        3, 0.5, 0.3, 1, ranked), "Invalid WeightedInteraction flag outside top-k was ignored");
    replace(original_weighted_interaction, gpu.weighted_interaction.scores);
    auto bad_vinardo = original_vinardo;
    bad_vinardo.back().gaussian = std::numeric_limits<double>::infinity();
    replace(bad_vinardo, gpu.vinardo.scores);
    require(!cuda_rank_rescored_poses(fixture.pool, gpu.selection, gpu.weighted_interaction, gpu.vinardo,
        2, 0.5, 0.3, 1, ranked), "Nonfinite Vinardo component outside top-k was ignored");
    bad_cpu = cpu_original;
    bad_cpu.poses.back().vinardo.gaussian = std::numeric_limits<double>::infinity();
    require(cpu_rejects(bad_cpu, PoseScoreMethod::Vinardo), "CPU Vinardo accepted the same nonfinite row");
    replace(original_vinardo, gpu.vinardo.scores);
    auto bad_fast = fixture.fast;
    bad_fast.back().weighted_score = std::numeric_limits<double>::quiet_NaN();
    replace(bad_fast, fixture.pool.fast_scores.scores);
    GpuRescoringSelection rejected;
    require(!cuda_select_rescoring_pool(fixture.pool.fast_scores, 1, rejected),
            "Nonfinite Fast outside prefilter was ignored");
    bad_cpu = cpu_original;
    bad_cpu.poses.back().pose.fast_weighted_score = std::numeric_limits<double>::quiet_NaN();
    require(cpu_rejects(bad_cpu, PoseScoreMethod::Hybrid), "CPU ranking accepted nonfinite Fast");
    cuda_free_rescoring_selection(rejected);
    replace(fixture.fast, fixture.pool.fast_scores.scores);

    auto constant_weighted_interaction = original_weighted_interaction;
    auto constant_vinardo = original_vinardo;
    for (auto& score : constant_weighted_interaction) { score = {}; score.valid = 1; }
    for (auto& score : constant_vinardo) { score = {}; score.valid = 1; }
    replace(constant_weighted_interaction, gpu.weighted_interaction.scores); replace(constant_vinardo, gpu.vinardo.scores);
    auto cpu = fixture.cpu_scores(ids);
    for (auto& row : cpu.poses) { row.pose.weighted_interaction_score = {}; row.vinardo = {}; }
    const auto cpu_ranked = rank_rescored_poses(cpu, {PoseScoreMethod::Hybrid, 0.5, 0.3}, ids.size());
    require(cuda_rank_rescored_poses(fixture.pool, gpu.selection, gpu.weighted_interaction, gpu.vinardo,
        3, 0.5, 0.3, gpu.selection.count, ranked), "Constant-component Hybrid failed");
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        compare_pose(ranked[i], cpu_ranked[i].scored, fixture);
        require(ranked[i].pose_id == i && ranked[i].hybrid == 0.0,
                "Constant scores must use zero z-scores and pose-ID tie-breaking");
        errors.close(ranked[i].hybrid, *cpu_ranked[i].hybrid, 1e-9, 1e-11,
                     errors.hybrid_absolute, "Constant Hybrid");
    }
}

void test_weighted_interaction_weights(Fixture& fixture) {
    ScoredPool scores;
    WeightedInteractionWeights weights;
    weights.hydrophobic = -2.0;
    weights.hydrogen_bond = 0.0;
    weights.halogen_bond = 0.75;
    weights.ionic = -1.0;
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        fixture.device_transforms(), nullptr, fixture.pool.survivors.count,
        make_gpu_weighted_interaction_weights(weights), scores.weighted_interaction), "Custom-weight WeightedInteraction failed");
    std::vector<std::uint32_t> ids(fixture.transforms.size());
    std::iota(ids.begin(), ids.end(), 0u);
    const auto expected = fixture.cpu_scores(ids, weights);
    const auto actual = download(scores.weighted_interaction.scores, scores.weighted_interaction.count);
    for (std::size_t i = 0; i < actual.size(); ++i)
        compare_weighted_interaction(weighted_interaction_score_from_gpu(actual[i]), expected.poses[i].pose.weighted_interaction_score);

    weights.hydrophobic = std::numeric_limits<double>::infinity();
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        fixture.device_transforms(), nullptr, fixture.pool.survivors.count,
        make_gpu_weighted_interaction_weights(weights), scores.weighted_interaction), "Nonfinite-weight WeightedInteraction kernel failed");
    const auto invalid = download(scores.weighted_interaction.scores, scores.weighted_interaction.count);
    require(!invalid.front().valid && !std::isfinite(invalid.front().weighted_score),
            "Nonfinite WeightedInteraction scores must be marked invalid");

    auto invalid_transforms = fixture.transforms;
    invalid_transforms.back().valid = 0;
    replace(invalid_transforms, fixture.pool.survivors.transforms);
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        fixture.device_transforms(), nullptr, fixture.pool.survivors.count,
        make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), scores.weighted_interaction), "Invalid-transform WeightedInteraction call failed");
    require(cuda_score_vinardo_device(fixture.device_receptor_atoms, fixture.device_ligand_atoms,
        fixture.device_transforms(), nullptr, fixture.pool.survivors.count, scores.vinardo),
        "Invalid-transform Vinardo call failed");
    const auto invalid_weighted_interaction = download(scores.weighted_interaction.scores, scores.weighted_interaction.count);
    const auto invalid_vinardo = download(scores.vinardo.scores, scores.vinardo.count);
    require(!invalid_weighted_interaction.back().valid && !invalid_vinardo.back().valid,
            "Both scorers must mark an invalid transform as invalid");
    replace(fixture.transforms, fixture.pool.survivors.transforms);
}

void test_empty(Fixture& fixture, Errors& errors) {
    GpuPosePipelineDeviceResult empty;
    ScoredPool scores;
    require(cuda_select_rescoring_pool(empty.fast_scores, 0, scores.selection), "Empty selection failed");
    require(cuda_score_weighted_interaction_device(fixture.device_protein, fixture.device_ligand,
        {}, nullptr, 0, make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), scores.weighted_interaction), "Empty WeightedInteraction failed");
    require(cuda_score_vinardo_device(fixture.device_receptor_atoms, fixture.device_ligand_atoms,
        {}, nullptr, 0, scores.vinardo), "Empty Vinardo failed");
    for (int method : {1, 2, 3}) {
        std::vector<GpuRankedRescoredPose> ranked(1);
        require(cuda_rank_rescored_poses(empty, scores.selection, scores.weighted_interaction, scores.vinardo,
            method, 0.5, 0.3, 1, ranked) && ranked.empty(), "Empty pool ranking failed");
    }
    require(cuda_score_weighted_interaction_device({}, fixture.device_ligand, fixture.device_transforms(),
        nullptr, fixture.pool.survivors.count, make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), scores.weighted_interaction),
        "Empty WeightedInteraction receptor failed");
    for (const auto& score : download(scores.weighted_interaction.scores, scores.weighted_interaction.count))
        compare_weighted_interaction(weighted_interaction_score_from_gpu(score), {});
    require(cuda_score_weighted_interaction_device(fixture.device_protein, {}, fixture.device_transforms(),
        nullptr, fixture.pool.survivors.count, make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), scores.weighted_interaction),
        "Empty WeightedInteraction ligand failed");
    for (const auto& score : download(scores.weighted_interaction.scores, scores.weighted_interaction.count))
        compare_weighted_interaction(weighted_interaction_score_from_gpu(score), {});
    require(cuda_score_vinardo_device({}, fixture.device_ligand_atoms, fixture.device_transforms(),
        nullptr, fixture.pool.survivors.count, scores.vinardo), "Empty Vinardo receptor failed");
    for (const auto& score : download(scores.vinardo.scores, scores.vinardo.count))
        compare_vinardo(vinardo_score_from_gpu(score), {}, errors);
}

} // namespace

int main() {
    try {
        Fixture fixture;
        fixture.upload_data();
        require(fixture.vinardo.receptor_skipped_unsupported() == 1,
                "Unsupported receptor atoms must retain CPU skip semantics");
        const auto identity = fixture.cpu_scores({0});
        const auto& weighted_interaction = identity.poses[0].pose.weighted_interaction_score;
        require(weighted_interaction.hydrophobic == 2 && weighted_interaction.hydrogen_bonds > 0 && weighted_interaction.halogen_bonds > 0 &&
                weighted_interaction.ionic > 0 && weighted_interaction.cation_pi > 0 && weighted_interaction.metal > 0,
                "WeightedInteraction fixture does not exercise required predicates and the distance boundary");
        const auto& v = identity.poses[0].vinardo;
        require(v.gaussian != 0 && v.repulsion != 0 && v.hydrophobic != 0 && v.hydrogen_bond != 0,
                "Vinardo fixture does not exercise all four terms");
        Errors errors;
        test_pool(fixture, 0, errors);
        test_pool(fixture, 2, errors);
        test_pool(fixture, 99, errors);
        test_errors_and_constants(fixture, errors);
        test_weighted_interaction_weights(fixture);
        test_empty(fixture, errors);
        std::cout << std::setprecision(12)
                  << "CUDA rescoring equivalence passed; WeightedInteraction counts and weights exact; "
                  << "Vinardo tolerance=1e-10+1e-8*abs(cpu), Hybrid=1e-11+1e-9*abs(cpu)\n"
                  << "max_vinardo_absolute_error=" << errors.vinardo_absolute << '\n'
                  << "max_hybrid_absolute_error=" << errors.hybrid_absolute << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
