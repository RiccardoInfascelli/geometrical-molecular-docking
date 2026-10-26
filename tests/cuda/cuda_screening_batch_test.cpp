#include "cuda_screening_batch.hpp"
#include "cuda_pose_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

bool close(double a, double b, double tolerance = 1e-9)
{
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::abs(a - b) <= tolerance;
}

struct QuaternionMethod
{
    const char* name = "DOCKING_KABSCH_METHOD";
    bool existed = std::getenv(name) != nullptr;
    std::string old_value = existed ? std::getenv(name) : "";

    QuaternionMethod() { require(setenv(name, "quaternion", 1) == 0, "setenv failed"); }
    ~QuaternionMethod()
    {
        if (existed) setenv(name, old_value.c_str(), 1);
        else unsetenv(name);
    }
};

struct SharedDeviceData
{
    GpuFlatTripletIndexDevice index{};
    GpuPointArrayDevice targets{};
    GpuProteinGridDevice grid{};

    ~SharedDeviceData()
    {
        cuda_free_protein_grid(grid);
        cuda_free_points(targets);
        cuda_free_triplet_index(index);
    }
};

struct BaselineBuffers
{
    GpuPointArrayDevice points{}, atoms{};
    GpuFastMatchingAreaArrayDevice fast_areas{};
    GpuTripletNeighborCountArrayDevice neighbor_counts{};
    GpuTripletNeighborOffsetArrayDevice neighbor_offsets{};
    GpuTripletCandidateArrayDevice candidates{};
    GpuAlignmentCountArrayDevice alignment_counts{};
    GpuAlignmentOffsetArrayDevice alignment_offsets{};
    GpuAlignmentCandidateArrayDevice alignments{};

    ~BaselineBuffers()
    {
        cuda_free_alignment_candidates(alignments);
        cuda_free_alignment_offsets(alignment_offsets);
        cuda_free_alignment_counts(alignment_counts);
        cuda_free_triplet_candidates(candidates);
        cuda_free_triplet_neighbor_offsets(neighbor_offsets);
        cuda_free_triplet_neighbor_counts(neighbor_counts);
        cuda_free_fast_matching_areas(fast_areas);
        cuda_free_points(atoms);
        cuda_free_points(points);
    }
};

struct PoseIdentity
{
    std::uint32_t query{}, neighbor{}, record{}, order{};
};

// Recover the original expansion identity from the baseline alignment index.
// Duplicate protein records have the same coordinates but remain distinct IDs.
PoseIdentity identity_for_alignment(const GpuFlatTripletIndexHost& index,
                                    const GpuScreeningBatchLigand& ligand,
                                    std::uint32_t alignment_index)
{
    std::uint64_t next_alignment = 0;
    for (std::size_t q = 0; q < ligand.queries.size(); ++q) {
        std::uint32_t slot = 0;
        for (int d0 = -1; d0 <= 1; ++d0) {
            for (int d1 = -1; d1 <= 1; ++d1) {
                for (int d2 = -1; d2 <= 1; ++d2, ++slot) {
                    auto query = ligand.queries[q];
                    const int deltas[] = {d0, d1, d2};
                    bool valid = true;
                    for (int i = 0; i < 3; ++i) {
                        if (deltas[i] < 0 && query.distance_bins[i] == 0) {
                            valid = false;
                            break;
                        }
                        query.distance_bins[i] = static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(query.distance_bins[i]) + deltas[i]);
                    }
                    if (!valid) continue;
                    for (std::size_t k = 0; k < index.keys.size(); ++k) {
                        bool equal = true;
                        for (int i = 0; i < 3; ++i)
                            equal = equal && query.keys[i] == index.keys[k].keys[i] &&
                                query.distance_bins[i] == index.keys[k].distance_bins[i];
                        if (!equal) continue;
                        for (std::uint32_t r = index.offsets[k]; r < index.offsets[k + 1]; ++r) {
                            for (std::uint32_t o = 0;
                                 o < ligand.orders[q].equivalent_order_count; ++o) {
                                if (next_alignment++ == alignment_index)
                                    return {static_cast<std::uint32_t>(q), slot, r, o};
                            }
                        }
                    }
                }
            }
        }
    }
    throw std::runtime_error("Baseline top alignment has no expansion identity");
}

struct BaselineResult
{
    std::uint64_t candidate_matches{};
    GpuPosePipelineResult pipeline;
    bool has_top_pose = false;
    GpuPipelinePose top{};
    PoseIdentity top_identity{};
};

BaselineResult run_baseline(const SharedDeviceData& shared,
                            const GpuFlatTripletIndexHost& host_index,
                            const GpuScreeningBatchLigand& ligand,
                            double clash_threshold)
{
    BaselineResult result;
    BaselineBuffers device;
    require(cuda_upload_points(ligand.matching_points, device.points), "baseline points");
    require(cuda_upload_points(ligand.atoms, device.atoms), "baseline atoms");
    require(cuda_upload_fast_matching_areas(ligand.fast_areas, device.fast_areas),
            "baseline Fast areas");
    require(cuda_count_triplet_neighbor_records_device(shared.index, ligand.queries,
                                                       device.neighbor_counts),
            "baseline neighbor counts");
    require(cuda_exclusive_triplet_neighbor_prefix_sum_device(device.neighbor_counts,
                                                              device.neighbor_offsets),
            "baseline neighbor prefix sum");
    require(cuda_expand_triplet_neighbor_records_device(shared.index, ligand.queries,
                device.neighbor_counts, device.neighbor_offsets, device.candidates),
            "baseline candidate expansion");
    result.candidate_matches = device.candidates.count;
    require(cuda_count_alignment_candidates_device(device.candidates, ligand.orders,
                                                   device.alignment_counts),
            "baseline alignment counts");
    require(cuda_exclusive_prefix_sum_device(device.alignment_counts, device.alignment_offsets),
            "baseline alignment prefix sum");
    require(cuda_expand_alignment_candidates_device(device.candidates, ligand.orders,
                device.alignment_offsets, device.alignments), "baseline alignment expansion");
    require(cuda_run_pose_pipeline(device.points, shared.targets, device.atoms,
                device.fast_areas, shared.grid, device.alignments, clash_threshold,
                result.pipeline), "baseline pose pipeline");
    for (const auto& pose : result.pipeline.poses) {
        if (pose.fast_score.valid &&
            (!result.has_top_pose || pose.fast_score.weighted_score >
                                     result.top.fast_score.weighted_score)) {
            result.has_top_pose = true;
            result.top = pose;
        }
    }
    if (result.has_top_pose)
        result.top_identity = identity_for_alignment(host_index, ligand,
                                                    result.top.alignment_index);
    return result;
}

GpuTripletKey key(std::uint32_t first_bin = 5)
{
    GpuTripletKey value{};
    value.keys[0] = 0;
    value.keys[1] = 1;
    value.keys[2] = 2;
    value.distance_bins[0] = first_bin;
    value.distance_bins[1] = 5;
    value.distance_bins[2] = 7;
    return value;
}

GpuScreeningBatchLigand make_ligand(std::size_t query_count, bool equivalent_orders)
{
    GpuScreeningBatchLigand ligand;
    ligand.matching_points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    // The distant atom remains outside the grid, as in the ordinary clash kernel.
    // Its extent exceeds the short receptor triangles without changing the index.
    ligand.atoms = {{0.2, 0.2, 0.2}, {1.2, 0.2, 0.2}, {10, 10, 10}};
    constexpr std::uint16_t interaction_bit = 1;
    ligand.fast_areas = {
        {{0.2, 0.2, 0.2}, interaction_bit, {}, 1.5},
        {{1.2, 0.2, 0.2}, interaction_bit, {}, 2.5},
        {{10, 10, 10}, interaction_bit, {}, 3.5}
    };
    ligand.queries.assign(query_count, key());
    GpuLigandTripletOrders orders{};
    orders.equivalent_order_count = equivalent_orders ? 2 : 1;
    orders.equivalent_orders[0][0] = 0;
    orders.equivalent_orders[0][1] = 1;
    orders.equivalent_orders[0][2] = 2;
    orders.equivalent_orders[1][0] = 0;
    orders.equivalent_orders[1][1] = 2;
    orders.equivalent_orders[1][2] = 1;
    ligand.orders.assign(query_count, orders);
    return ligand;
}

void compare_result(const BaselineResult& baseline, const GpuScreeningBatchResult& batch,
                    const std::string& label)
{
    require(batch.candidate_matches == baseline.candidate_matches, label + ": candidates");
    require(batch.transforms_generated == baseline.pipeline.input_alignment_count,
            label + ": transforms");
    require(batch.valid_transforms == baseline.pipeline.valid_transform_count,
            label + ": valid transforms");
    require(batch.survivors == baseline.pipeline.clash_survivor_count, label + ": survivors");
    require(batch.has_top_pose == baseline.has_top_pose, label + ": top-1 availability");
    if (!baseline.has_top_pose) return;
    require(batch.top_query_index == baseline.top_identity.query &&
            batch.top_neighbor_slot == baseline.top_identity.neighbor &&
            batch.top_record_index == baseline.top_identity.record &&
            batch.top_order_index == baseline.top_identity.order, label + ": top-1 identity");
    const auto& top = baseline.top;
    require(close(batch.top_fast_score.weighted_score, top.fast_score.weighted_score),
            label + ": Fast score");
    require(batch.top_fast_score.valid == top.fast_score.valid &&
            batch.top_fast_score.total_areas == top.fast_score.total_areas &&
            batch.top_fast_score.areas_in_grid == top.fast_score.areas_in_grid &&
            batch.top_fast_score.areas_on_occupied_voxel == top.fast_score.areas_on_occupied_voxel &&
            batch.top_fast_score.compatible_areas == top.fast_score.compatible_areas,
            label + ": Fast diagnostics");
    require(batch.top_clash.valid == top.clash.valid &&
            batch.top_clash.total_atoms == top.clash.total_atoms &&
            batch.top_clash.occupied_atoms == top.clash.occupied_atoms &&
            close(batch.top_clash.fraction, top.clash.fraction), label + ": top-1 clash");
    require(batch.top_transform.valid == top.transform.valid, label + ": transform validity");
    for (int i = 0; i < 9; ++i)
        require(close(batch.top_transform.rotation[i], top.transform.rotation[i]),
                label + ": top-1 rotation");
    for (int i = 0; i < 3; ++i)
        require(close(batch.top_transform.translation[i], top.transform.translation[i]),
                label + ": top-1 translation");
}

} // namespace

int main()
{
    try {
        QuaternionMethod method;
        SharedDeviceData shared;
        GpuFlatTripletIndexHost host_index;
        constexpr std::uint32_t dense_records = 257;
        host_index.keys = {key(), key(6)};
        host_index.offsets = {0, dense_records + 1, dense_records + 2};
        // A single query reaches more records than a block has threads. The
        // rejected first record also makes the first winning ID nonzero.
        host_index.records.push_back({{3, 4, 5}});
        for (std::uint32_t i = 0; i < dense_records; ++i)
            host_index.records.push_back({{0, 1, 2}});
        host_index.records.push_back({{3, 4, 5}});
        const std::vector<GpuPoint3> targets = {
            {1, 0, 0}, {2, 0, 0}, {1, 1, 0},
            {4, 0, 0}, {5, 0, 0}, {4, 1, 0}
        };
        GpuProteinGridHost grid;
        grid.voxel_size = 1.0;
        grid.nx = 16;
        grid.ny = 8;
        grid.nz = 8;
        grid.voxels.resize(16 * 8 * 8);
        for (auto& voxel : grid.voxels) voxel.interactions = 1;
        grid.voxels[4].occupied = 1;
        grid.voxels[5].occupied = 1;
        grid.voxels[7].occupied = 1;
        require(cuda_upload_triplet_index(host_index, shared.index), "shared index upload");
        require(cuda_upload_points(targets, shared.targets), "shared targets upload");
        require(cuda_upload_protein_grid(grid, shared.grid), "shared grid upload");
        const auto original_index_pointer = shared.index.records;

        std::vector<GpuScreeningBatchLigand> ligands = {
            make_ligand(1, false), make_ligand(7, true), make_ligand(65, false),
            make_ligand(0, false), make_ligand(1, false), make_ligand(3, false),
            make_ligand(3, false), make_ligand(3, false), make_ligand(2, true)
        };
        for (int i = 0; i < 37; ++i)
            ligands[2].atoms.push_back({10.0 + i, 10, 10});
        // Missing chemical key and bins at zero exercise absent and negative neighbors.
        ligands[4].queries[0] = key(0);
        ligands[4].queries[0].keys[2] = 3;
        // Every atom hits an occupied voxel for either target translation.
        ligands[5].atoms = {{3.2, 0.2, 0.2}};
        // The earliest tied winner now belongs to a later query.
        ligands[6].queries[0].keys[2] = 3;
        // Fast keeps an initial NaN and treats equal infinities as ties, just
        // like the baseline's strict greater-than comparison in host order.
        ligands[7].fast_areas[0].weight = std::numeric_limits<double>::quiet_NaN();
        ligands[8].fast_areas[0].weight = std::numeric_limits<double>::infinity();

        GpuScreeningBatchOptions options;
        options.clash_threshold = 0.5;
        std::vector<BaselineResult> baselines;
        for (const auto& ligand : ligands)
            baselines.push_back(run_baseline(shared, host_index, ligand,
                                             options.clash_threshold));
        require(baselines[0].candidate_matches == dense_records + 2 &&
                baselines[0].pipeline.input_alignment_count == dense_records + 2 &&
                baselines[0].pipeline.clash_survivor_count == dense_records,
                "fixture must contain both accepted and rejected transforms");
        require(!baselines[3].has_top_pose && !baselines[4].has_top_pose,
                "empty and absent queries must produce no top pose");
        require(baselines[5].pipeline.clash_survivor_count == 0,
                "occupied atoms must reject every transform in the all-clash fixture");
        require(baselines[6].top_identity.query == 1 &&
                std::isnan(baselines[7].top.fast_score.weighted_score) &&
                baselines[8].top.fast_score.weighted_score ==
                    std::numeric_limits<double>::infinity(),
                "fixtures must exercise later-query ties, NaN and infinite Fast scores");

        std::size_t single_block_device_bytes = 0;
        for (const auto strategy : {GpuScreeningBatchStrategy::WarpPerLigand,
                                    GpuScreeningBatchStrategy::BlockPerLigand,
                                    GpuScreeningBatchStrategy::MultiBlockPerLigand}) {
            for (const std::uint32_t block_size : {128u, 256u}) {
                if (strategy == GpuScreeningBatchStrategy::WarpPerLigand && block_size == 256)
                    continue;
                options.strategy = strategy;
                options.block_size = strategy == GpuScreeningBatchStrategy::WarpPerLigand ?
                                     32u : block_size;
                for (const std::uint32_t blocks : {1u, 4u, 8u, 16u, 32u}) {
                    if (strategy != GpuScreeningBatchStrategy::MultiBlockPerLigand && blocks != 1)
                        continue;
                    options.blocks_per_ligand = blocks;
                    const std::string configuration = "strategy " +
                        std::to_string(static_cast<int>(strategy)) + ", threads " +
                        std::to_string(options.block_size) + ", blocks " + std::to_string(blocks);
                    std::vector<GpuScreeningBatchResult> results;
                    GpuScreeningBatchTimings timings;
                    std::string error;
                    const bool success = cuda_screen_ligand_batch(
                        shared.index, shared.targets, shared.grid, ligands, options,
                        results, timings, error);
                    require(success, configuration + ": " + error);
                    require(results.size() == ligands.size(), "batch must retain ligand order");
                    for (std::size_t i = 0; i < ligands.size(); ++i)
                        compare_result(baselines[i], results[i], configuration + ", ligand " +
                                       std::to_string(i));
                    for (std::size_t i = 0; i < 3; ++i)
                        require(results[i].top_query_index == 0 && results[i].top_neighbor_slot == 13 &&
                                results[i].top_record_index == 1 && results[i].top_order_index == 0,
                                "equal scores across threads must keep the earliest baseline alignment");
                    require(timings.batch_device_bytes <= options.max_device_bytes &&
                            std::isfinite(timings.kernel_seconds) && timings.kernel_seconds >= 0,
                            "batch memory and completed timing metrics");
                    require(shared.index.records == original_index_pointer &&
                            shared.index.record_count == dense_records + 2,
                            "heterogeneous ligands must reuse the same shared index");
                    if (strategy == GpuScreeningBatchStrategy::BlockPerLigand && block_size == 128)
                        single_block_device_bytes = timings.batch_device_bytes;
                }
            }
        }

        const auto expect_rejection = [&](const std::vector<GpuScreeningBatchLigand>& inputs,
                                          const GpuScreeningBatchOptions& invalid) {
            std::vector<GpuScreeningBatchResult> results(1);
            GpuScreeningBatchTimings timings;
            std::string error;
            require(!cuda_screen_ligand_batch(shared.index, shared.targets, shared.grid,
                        inputs, invalid, results, timings, error), "invalid batch accepted");
            require(results.empty() && !error.empty(), "failure must clear results and explain why");
        };
        options.strategy = GpuScreeningBatchStrategy::BlockPerLigand;
        options.block_size = 33;
        expect_rejection(ligands, options);
        options.block_size = 128;
        options.max_device_bytes = 1;
        expect_rejection(ligands, options);
        options.max_device_bytes = std::size_t{256} * 1024 * 1024;
        options.strategy = GpuScreeningBatchStrategy::MultiBlockPerLigand;
        options.blocks_per_ligand = 0;
        expect_rejection(ligands, options);
        options.blocks_per_ligand = std::numeric_limits<std::uint32_t>::max();
        expect_rejection(ligands, options);
        options.blocks_per_ligand = 32;
        require(single_block_device_bytes > 0, "single-block allocation size was not recorded");
        // A BlockPerLigand budget must still cover MultiBlockPerLigand summary
        // allocations; insufficient memory is rejected before device work.
        options.max_device_bytes = single_block_device_bytes;
        expect_rejection(ligands, options);
        options.max_device_bytes = std::size_t{256} * 1024 * 1024;
        options.blocks_per_ligand = 8;
        options.clash_threshold = std::numeric_limits<double>::quiet_NaN();
        expect_rejection(ligands, options);
        options.clash_threshold = 0.5;
        auto invalid_ligands = ligands;
        invalid_ligands[0].orders[0].equivalent_order_count = 7;
        expect_rejection(invalid_ligands, options);
        invalid_ligands = ligands;
        invalid_ligands[0].orders[0].equivalent_orders[0][2] = 1000;
        expect_rejection(invalid_ligands, options);
        std::vector<GpuScreeningBatchResult> empty_results(1);
        GpuScreeningBatchTimings empty_timings;
        std::string error;
        require(cuda_screen_ligand_batch(shared.index, shared.targets, shared.grid, {},
                    options, empty_results, empty_timings, error) && empty_results.empty(),
                "empty batch must succeed with no records");
        std::cout << "CUDA batch screening: warp/block/multi-block counts, top-1, shared index and bounds passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA batch screening test failed: " << error.what() << '\n';
        return 1;
    }
}
