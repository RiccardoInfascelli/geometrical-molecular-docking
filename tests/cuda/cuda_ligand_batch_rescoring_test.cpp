#include "cuda_pose_ranking.hpp"
#include "cuda_scoring_range.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template<class T> struct DeviceArray {
    T* data{};
    explicit DeviceArray(const std::vector<T>& host) {
        if (host.empty()) return;
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&data), host.size() * sizeof(T)), "allocation");
        if (cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(data); data = nullptr;
            throw std::runtime_error("upload failed");
        }
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
template<class T> std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    if (count) cuda_require(cudaMemcpy(host.data(), device, count * sizeof(T),
                                      cudaMemcpyDeviceToHost), "download");
    return host;
}
struct Chemistry {
    GpuWeightedInteractionAreasDevice areas{};
    GpuVinardoAtomsDevice atoms{};
    Chemistry(const std::vector<GpuWeightedInteractionArea>& host_areas,
              const std::vector<GpuVinardoAtom>& host_atoms) {
        require(cuda_upload_weighted_interaction_areas(host_areas, areas), "area upload failed");
        if (!cuda_upload_vinardo_atoms(host_atoms, atoms)) {
            cuda_free_weighted_interaction_areas(areas);
            throw std::runtime_error("atom upload failed");
        }
    }
    ~Chemistry() {
        cuda_free_weighted_interaction_areas(areas);
        cuda_free_vinardo_atoms(atoms);
    }
};
struct Scores {
    GpuWeightedInteractionScoresDevice interaction{};
    GpuVinardoScoresDevice vinardo{};
    ~Scores() {
        cuda_free_weighted_interaction_scores(interaction);
        cuda_free_vinardo_scores(vinardo);
    }
};
GpuWeightedInteractionArea area(float x, GpuWeightedInteractionPattern pattern) {
    GpuWeightedInteractionArea result{};
    result.centroid = {x, 0, 0};
    result.atoms[0] = result.centroid;
    result.num_atoms = 1;
    result.pattern = pattern;
    return result;
}
GpuVinardoAtom atom(double x, bool hydrophobic, bool donor, bool acceptor) {
    return {x, 0, 0, 1.7, hydrophobic, donor, acceptor, 0};
}
GpuRigidTransform transform(double x) {
    GpuRigidTransform result{};
    result.rotation[0] = result.rotation[4] = result.rotation[8] = 1;
    result.translation[0] = x;
    result.valid = 1;
    return result;
}
void compare(const GpuWeightedInteractionScore& a, const GpuWeightedInteractionScore& b) {
    require(a.valid == b.valid && a.total_interactions == b.total_interactions &&
        a.satisfied_ligand_areas == b.satisfied_ligand_areas &&
        a.satisfied_protein_areas == b.satisfied_protein_areas &&
        a.hydrophobic == b.hydrophobic && a.hydrogen_bonds == b.hydrogen_bonds &&
        a.halogen_bonds == b.halogen_bonds && a.ionic == b.ionic &&
        a.cation_pi == b.cation_pi && a.pi_stacking == b.pi_stacking &&
        a.metal == b.metal && a.weighted_score == b.weighted_score,
        "batched WeightedInteraction differs from standalone scoring");
}
void compare(const GpuVinardoScore& a, const GpuVinardoScore& b) {
    require(a.valid == b.valid, "Vinardo validity differs");
    const double actual[]{a.total, a.gaussian, a.repulsion, a.hydrophobic, a.hydrogen_bond};
    const double expected[]{b.total, b.gaussian, b.repulsion, b.hydrophobic, b.hydrogen_bond};
    for (std::size_t i = 0; i < 5; ++i)
        require(std::abs(actual[i] - expected[i]) <= 1e-12 * (1.0 + std::abs(expected[i])),
                "batched Vinardo component differs from standalone scoring");
}
void standalone(const Chemistry& receptor, const Chemistry& ligand,
                const GpuRigidTransformArrayDevice& transforms,
                const std::uint32_t* selected, const GpuWeightedInteractionWeights& weights,
                Scores& scores) {
    require(cuda_score_weighted_interaction_device(receptor.areas, ligand.areas, transforms,
        selected, 2, weights, scores.interaction), "standalone interaction scoring failed");
    require(cuda_score_vinardo_device(receptor.atoms, ligand.atoms, transforms,
        selected, 2, scores.vinardo), "standalone Vinardo scoring failed");
}
void compare_rankings(const std::vector<GpuRigidTransform>& transforms,
                     const std::vector<std::uint32_t>& selection,
                     const Scores& standalone_scores, const Scores& batch_scores,
                     std::uint32_t score_begin) {
    GpuPosePipelineDeviceResult pool;
    const std::size_t count = transforms.size();
    std::vector<GpuFastScore> fast(count);
    std::vector<GpuClashScore> clashes(count);
    std::vector<std::uint32_t> alignment_ids(count);
    for (std::size_t i = 0; i < count; ++i) {
        fast[i].valid = 1; fast[i].weighted_score = 1;
        clashes[i].valid = 1; alignment_ids[i] = static_cast<std::uint32_t>(i);
    }
    const auto upload = [](const auto& host, auto*& device) {
        using T = typename std::decay_t<decltype(host)>::value_type;
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)), "pool allocation");
        cuda_require(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "pool upload");
    };
    upload(transforms, pool.survivors.transforms);
    upload(clashes, pool.survivors.clash_scores);
    upload(alignment_ids, pool.survivors.alignment_indices);
    upload(fast, pool.fast_scores.scores);
    pool.survivors.count = pool.fast_scores.count = static_cast<std::uint32_t>(count);
    DeviceArray<std::uint32_t> selected(selection);
    const GpuRescoringSelection view{selected.data, 2};
    const GpuWeightedInteractionScoresDevice batch_interaction{batch_scores.interaction.scores + score_begin, 2};
    const GpuVinardoScoresDevice batch_vinardo{batch_scores.vinardo.scores + score_begin, 2};
    for (int method : {1, 2, 3}) {
        for (double alpha : {1.0, 0.5}) {
            std::vector<GpuRankedRescoredPose> expected, actual;
            require(cuda_rank_rescored_poses(pool, view, standalone_scores.interaction,
                standalone_scores.vinardo, method, alpha, 0.3, 2, expected), "standalone ranking failed");
            require(cuda_rank_rescored_poses(pool, view, batch_interaction,
                batch_vinardo, method, alpha, 0.3, 2, actual), "batched score ranking failed");
            require(actual.size() == expected.size(), "ranking size differs");
            for (std::size_t i = 0; i < actual.size(); ++i) {
                require(actual[i].pose_id == expected[i].pose_id, "local top-k identity differs");
                require(actual[i].adjusted == expected[i].adjusted && actual[i].hybrid == expected[i].hybrid,
                        "local adjusted Vinardo or Hybrid normalization differs");
            }
        }
    }
}
void test_scoring() {
    using Pattern = GpuWeightedInteractionPattern;
    const std::vector<GpuWeightedInteractionArea> areas_a{
        area(0, Pattern::Hydrophobic), area(12, Pattern::Anion)};
    const std::vector<GpuWeightedInteractionArea> areas_b{
        area(30, Pattern::Metal), area(31, Pattern::Hydrophobic), area(70, Pattern::Hydrophobic)};
    const std::vector<GpuVinardoAtom> atoms_a{atom(0, true, false, false), atom(2, false, true, false)};
    const std::vector<GpuVinardoAtom> atoms_b{
        atom(30, true, false, false), atom(33, false, false, true),
        atom(70, true, false, false), atom(76, false, false, true)};
    Chemistry receptor({area(3, Pattern::Hydrophobic), area(15, Pattern::Cation),
                       area(32, Pattern::Chelated), area(34, Pattern::Hydrophobic)},
                      {atom(3, true, false, false), atom(5, false, false, true),
                       atom(32, true, true, false)});
    Chemistry a(areas_a, atoms_a), b(areas_b, atoms_b);
    auto all_areas = areas_a;
    all_areas.insert(all_areas.end(), areas_b.begin(), areas_b.end());
    auto all_atoms = atoms_a;
    all_atoms.insert(all_atoms.end(), atoms_b.begin(), atoms_b.end());
    Chemistry packed(all_areas, all_atoms);
    const std::vector<GpuRigidTransform> transforms_a{transform(0), transform(1.5), transform(-2)};
    const std::vector<GpuRigidTransform> transforms_b{transform(0), transform(2.5)};
    auto transforms = transforms_a;
    transforms.insert(transforms.end(), transforms_b.begin(), transforms_b.end());
    DeviceArray<GpuRigidTransform> all_transforms(transforms), ta(transforms_a), tb(transforms_b);
    const GpuRigidTransformArrayDevice device_transforms{all_transforms.data, 5};
    const std::vector<std::uint32_t> selected_ids{2, 0, 1, 0};
    DeviceArray<std::uint32_t> selected(selected_ids);
    DeviceArray<GpuLigandScoringRange> area_ranges({{0, 2, 0}, {0, 2, 0}, {2, 3, 3}, {2, 3, 3}});
    DeviceArray<GpuLigandScoringRange> atom_ranges({{0, 2, 0}, {0, 2, 0}, {2, 4, 3}, {2, 4, 3}});
    GpuWeightedInteractionWeights weights{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    Scores batch, single_a, single_b;
    require(cuda_score_weighted_interaction_batch_device(receptor.areas, packed.areas,
        device_transforms, selected.data, 4, area_ranges.data, weights, batch.interaction), "batch interaction failed");
    require(cuda_score_vinardo_batch_device(receptor.atoms, packed.atoms,
        device_transforms, selected.data, 4, atom_ranges.data, batch.vinardo), "batch Vinardo failed");
    standalone(receptor, a, {ta.data, 3}, selected.data, weights, single_a);
    standalone(receptor, b, {tb.data, 2}, selected.data + 2, weights, single_b);
    const auto interaction = download(batch.interaction.scores, 4);
    const auto vinardo = download(batch.vinardo.scores, 4);
    const auto expected_interaction_a = download(single_a.interaction.scores, 2);
    const auto expected_interaction_b = download(single_b.interaction.scores, 2);
    const auto expected_vinardo_a = download(single_a.vinardo.scores, 2);
    const auto expected_vinardo_b = download(single_b.vinardo.scores, 2);
    for (std::size_t i = 0; i < 4; ++i) {
        compare(interaction[i], i < 2 ? expected_interaction_a[i] : expected_interaction_b[i - 2]);
        compare(vinardo[i], i < 2 ? expected_vinardo_a[i] : expected_vinardo_b[i - 2]);
        require(interaction[i].valid && vinardo[i].valid, "fixture produced invalid scores");
    }
    require(interaction[0].weighted_score != interaction[2].weighted_score,
            "fixture does not distinguish ligand score distributions");
    compare_rankings(transforms_a, {2, 0}, single_a, batch, 0);
    compare_rankings(transforms_b, {1, 0}, single_b, batch, 2);

    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    DeviceArray<std::uint32_t> bad_ids({0, 0, 0, maximum});
    DeviceArray<GpuLigandScoringRange> bad_ranges({{maximum, 0, 0}, {0, maximum, 0},
                                                {0, 1, maximum}, {0, 1, 1}});
    require(cuda_score_weighted_interaction_batch_device(receptor.areas, packed.areas,
        device_transforms, bad_ids.data, 4, bad_ranges.data, weights, batch.interaction), "invalid interaction range launch failed");
    require(cuda_score_vinardo_batch_device(receptor.atoms, packed.atoms,
        device_transforms, bad_ids.data, 4, bad_ranges.data, batch.vinardo), "invalid atom range launch failed");
    for (const auto& row : download(batch.interaction.scores, 4)) require(!row.valid, "invalid area/pose range accepted");
    for (const auto& row : download(batch.vinardo.scores, 4)) require(!row.valid, "invalid atom/pose range accepted");

    DeviceArray<std::uint32_t> zero_id({0});
    DeviceArray<GpuLigandScoringRange> empty_range({{5, 0, 0}});
    DeviceArray<GpuLigandScoringRange> empty_atom_range({{6, 0, 0}});
    require(cuda_score_weighted_interaction_batch_device(receptor.areas, packed.areas,
        device_transforms, zero_id.data, 1, empty_range.data, weights, batch.interaction), "empty area range failed");
    require(cuda_score_vinardo_batch_device(receptor.atoms, packed.atoms,
        device_transforms, zero_id.data, 1, empty_atom_range.data, batch.vinardo), "empty atom range failed");
    require(download(batch.interaction.scores, 1)[0].valid &&
            download(batch.interaction.scores, 1)[0].weighted_score == 0, "empty areas changed score");
    require(download(batch.vinardo.scores, 1)[0].valid &&
            download(batch.vinardo.scores, 1)[0].total == 0, "empty atoms changed score");
    require(!cuda_score_weighted_interaction_batch_device(receptor.areas, packed.areas,
        device_transforms, selected.data, 1, nullptr, weights, batch.interaction), "missing area ranges accepted");
    require(!cuda_score_vinardo_batch_device(receptor.atoms, packed.atoms,
        device_transforms, nullptr, 1, atom_ranges.data, batch.vinardo), "missing local IDs accepted");
    require(cuda_score_weighted_interaction_batch_device({}, {}, {}, nullptr, 0, nullptr,
        weights, batch.interaction), "empty interaction batch failed");
    require(cuda_score_vinardo_batch_device({}, {}, {}, nullptr, 0, nullptr, batch.vinardo), "empty Vinardo batch failed");
    require(!batch.interaction.count && !batch.interaction.scores &&
            !batch.vinardo.count && !batch.vinardo.scores, "empty batch retained old scores");
}
} // namespace

int main() {
    try {
        test_scoring();
        std::cout << "CUDA batched ligand scoring matches standalone components and local rankings\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
