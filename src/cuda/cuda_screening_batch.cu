#include "cuda_screening_batch.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct LigandDescriptor
{
    std::size_t queries_offset{};
    std::size_t orders_offset{};
    std::size_t matching_offset{};
    std::size_t atoms_offset{};
    std::size_t areas_offset{};
    std::uint32_t query_count{};
    std::uint32_t atom_count{};
    std::uint32_t area_count{};
};

struct PoseSummary
{
    GpuRigidTransform transform{};
    GpuClashScore clash{};
    GpuFastScore fast{};
    std::uint32_t query{};
    std::uint32_t neighbor{};
    std::uint32_t record{};
    std::uint32_t order{};
    std::uint32_t present{};
};

struct ThreadSummary
{
    std::uint64_t matches{};
    std::uint64_t transforms{};
    std::uint64_t valid{};
    std::uint64_t survivors{};
    std::uint32_t error{};
    // Preserve the baseline's first-score behavior even for NaN: an initial
    // NaN wins forever, whereas a later NaN never replaces a numeric score.
    PoseSummary first{};
    PoseSummary best_numeric{};
};

__host__ __device__ bool earlier(const PoseSummary& a, const PoseSummary& b)
{
    if (a.query != b.query) return a.query < b.query;
    if (a.neighbor != b.neighbor) return a.neighbor < b.neighbor;
    if (a.record != b.record) return a.record < b.record;
    return a.order < b.order;
}

__device__ void increment(std::uint64_t& count, std::uint32_t& error)
{
    if (count == UINT64_MAX) error |= 2u;
    else ++count;
}

// All execution strategies use the same candidate evaluation and score rules.
__device__ __forceinline__ void evaluate_record(
    ThreadSummary& summary,
    std::uint32_t query_index,
    std::uint32_t neighbor_slot,
    std::uint32_t record_index,
    const GpuLigandTripletOrders& ligand_orders,
    const GpuPoint3* matching,
    const GpuPoint3* atoms,
    const GpuFastMatchingArea* areas,
    const LigandDescriptor& ligand,
    const GpuFlatTripletIndexDevice& protein_index,
    const GpuPointArrayDevice& protein_targets,
    const GpuProteinGridDevice& protein_grid,
    double clash_threshold)
{
    increment(summary.matches, summary.error);
    const auto record = protein_index.records[record_index];
    bool indices_valid = true;
    for (int i = 0; i < 3; ++i)
        indices_valid = indices_valid && record.indices[i] < protein_targets.count;
    if (!indices_valid) {
        summary.error |= 1u;
        return;
    }
    for (std::uint32_t o = 0; o < ligand_orders.equivalent_order_count; ++o) {
        increment(summary.transforms, summary.error);
        GpuAlignmentCandidate alignment{};
        for (int i = 0; i < 3; ++i) {
            alignment.ligand_indices[i] = ligand_orders.equivalent_orders[o][i];
            alignment.protein_indices[i] = record.indices[i];
        }
        const auto transform = docking_device::quaternion_fit(
            matching, protein_targets.points, alignment);
        if (!transform.valid) continue;
        increment(summary.valid, summary.error);
        const auto clash = docking_device::grid_clash(
            atoms, ligand.atom_count, protein_grid, transform);
        if (!clash.valid || !(clash.fraction <= clash_threshold)) continue;
        increment(summary.survivors, summary.error);
        const auto fast = docking_device::fast_score(
            areas, ligand.area_count, protein_grid, transform);
        if (!fast.valid) continue;
        PoseSummary pose{};
        pose.transform = transform;
        pose.clash = clash;
        pose.fast = fast;
        pose.query = query_index;
        pose.neighbor = neighbor_slot;
        pose.record = record_index;
        pose.order = o;
        pose.present = 1;
        if (!summary.first.present) summary.first = pose;
        if (!isnan(fast.weighted_score) &&
            (!summary.best_numeric.present ||
             fast.weighted_score > summary.best_numeric.fast.weighted_score)) {
            summary.best_numeric = pose;
        }
    }
}

__global__ void screen_batch_kernel(
    const unsigned char* packed,
    GpuFlatTripletIndexDevice protein_index,
    GpuPointArrayDevice protein_targets,
    GpuProteinGridDevice protein_grid,
    double clash_threshold,
    ThreadSummary* summaries)
{
    const auto* descriptors = reinterpret_cast<const LigandDescriptor*>(packed);
    const LigandDescriptor ligand = descriptors[blockIdx.x];
    const auto* queries = reinterpret_cast<const GpuTripletKey*>(packed + ligand.queries_offset);
    const auto* orders = reinterpret_cast<const GpuLigandTripletOrders*>(packed + ligand.orders_offset);
    const auto* matching = reinterpret_cast<const GpuPoint3*>(packed + ligand.matching_offset);
    const auto* atoms = reinterpret_cast<const GpuPoint3*>(packed + ligand.atoms_offset);
    const auto* areas = reinterpret_cast<const GpuFastMatchingArea*>(packed + ligand.areas_offset);
    ThreadSummary summary{};

    // Queries are disjoint between threads; within each query the traversal is
    // identical to neighbor expansion followed by equivalent-order expansion.
    for (std::uint64_t q = threadIdx.x; q < ligand.query_count; q += blockDim.x) {
        const GpuTripletKey base = queries[q];
        const GpuLigandTripletOrders ligand_orders = orders[q];
        std::uint32_t slot = 0;
        for (int d0 = -1; d0 <= 1; ++d0) {
            for (int d1 = -1; d1 <= 1; ++d1) {
                for (int d2 = -1; d2 <= 1; ++d2, ++slot) {
                    GpuTripletKey query = base;
                    const int deltas[3] = {d0, d1, d2};
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
                    const std::uint32_t key = docking_device::find_triplet_key(
                        protein_index.keys, protein_index.key_count, query);
                    if (key == GPU_TRIPLET_KEY_NOT_FOUND) continue;
                    const std::uint32_t begin = protein_index.offsets[key];
                    const std::uint32_t end = protein_index.offsets[key + 1];
                    if (begin > end || end > protein_index.record_count) {
                        summary.error |= 1u;
                        continue;
                    }
                    for (std::uint64_t r = begin; r < end; ++r) {
                        evaluate_record(summary, static_cast<std::uint32_t>(q), slot,
                            static_cast<std::uint32_t>(r), ligand_orders, matching, atoms,
                            areas, ligand, protein_index, protein_targets, protein_grid,
                            clash_threshold);
                    }
                }
            }
        }
    }
    summaries[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = summary;
}

__global__ void screen_multi_block_kernel(
    const unsigned char* packed,
    GpuFlatTripletIndexDevice protein_index,
    GpuPointArrayDevice protein_targets,
    GpuProteinGridDevice protein_grid,
    double clash_threshold,
    std::uint32_t blocks_per_ligand,
    ThreadSummary* summaries)
{
    const auto* descriptors = reinterpret_cast<const LigandDescriptor*>(packed);
    const LigandDescriptor ligand = descriptors[blockIdx.x / blocks_per_ligand];
    const auto* queries = reinterpret_cast<const GpuTripletKey*>(packed + ligand.queries_offset);
    const auto* orders = reinterpret_cast<const GpuLigandTripletOrders*>(packed + ligand.orders_offset);
    const auto* matching = reinterpret_cast<const GpuPoint3*>(packed + ligand.matching_offset);
    const auto* atoms = reinterpret_cast<const GpuPoint3*>(packed + ligand.atoms_offset);
    const auto* areas = reinterpret_cast<const GpuFastMatchingArea*>(packed + ligand.areas_offset);
    const std::uint32_t shard = blockIdx.x % blocks_per_ligand;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x % 32;
    ThreadSummary summary{};

    // Each query is visited by one warp in every block of its ligand. Only
    // lane zero looks up a bucket; records, including very large buckets, are
    // disjoint across lanes and blocks. Equivalent orders keep their order.
    for (std::uint64_t q = warp; q < ligand.query_count; q += blockDim.x / 32) {
        const GpuTripletKey base = queries[q];
        const GpuLigandTripletOrders ligand_orders = orders[q];
        std::uint32_t slot = 0;
        for (int d0 = -1; d0 <= 1; ++d0) {
            for (int d1 = -1; d1 <= 1; ++d1) {
                for (int d2 = -1; d2 <= 1; ++d2, ++slot) {
                    std::uint32_t begin = 0, end = 0;
                    if (lane == 0) {
                        GpuTripletKey query = base;
                        const int deltas[3] = {d0, d1, d2};
                        bool valid = true;
                        for (int i = 0; i < 3; ++i) {
                            if (deltas[i] < 0 && query.distance_bins[i] == 0) {
                                valid = false;
                                break;
                            }
                            query.distance_bins[i] = static_cast<std::uint32_t>(
                                static_cast<std::int64_t>(query.distance_bins[i]) + deltas[i]);
                        }
                        if (valid) {
                            const std::uint32_t key = docking_device::find_triplet_key(
                                protein_index.keys, protein_index.key_count, query);
                            if (key != GPU_TRIPLET_KEY_NOT_FOUND) {
                                begin = protein_index.offsets[key];
                                end = protein_index.offsets[key + 1];
                                if (begin > end || end > protein_index.record_count) {
                                    summary.error |= 1u;
                                    begin = end = 0;
                                }
                            }
                        }
                    }
                    // All 32 lanes enter together, including empty queries.
                    begin = __shfl_sync(0xffffffffu, begin, 0);
                    end = __shfl_sync(0xffffffffu, end, 0);
                    const std::uint64_t stride = static_cast<std::uint64_t>(blocks_per_ligand) * 32;
                    for (std::uint64_t r = static_cast<std::uint64_t>(begin) +
                             static_cast<std::uint64_t>(shard) * 32 + lane;
                         r < end; r += stride) {
                        evaluate_record(summary, static_cast<std::uint32_t>(q), slot,
                            static_cast<std::uint32_t>(r), ligand_orders, matching, atoms,
                            areas, ligand, protein_index, protein_targets, protein_grid,
                            clash_threshold);
                    }
                }
            }
        }
    }
    summaries[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = summary;
}

__host__ __device__ void merge_summary(ThreadSummary& total, const ThreadSummary& value)
{
    total.error |= value.error;
    std::uint64_t* totals[] = {&total.matches, &total.transforms, &total.valid, &total.survivors};
    const std::uint64_t values[] = {value.matches, value.transforms, value.valid, value.survivors};
    for (int i = 0; i < 4; ++i) {
        if (values[i] > UINT64_MAX - *totals[i]) total.error |= 2u;
        else *totals[i] += values[i];
    }
    if (value.first.present && (!total.first.present || earlier(value.first, total.first)))
        total.first = value.first;
    const auto& candidate = value.best_numeric;
    if (candidate.present && (!total.best_numeric.present ||
        candidate.fast.weighted_score > total.best_numeric.fast.weighted_score ||
        (candidate.fast.weighted_score == total.best_numeric.fast.weighted_score &&
         earlier(candidate, total.best_numeric))))
        total.best_numeric = candidate;
}

// This second kernel has no grid-wide barrier or atomic top-pose updates.
// Each thread reduces one producing block; only one summary/block is copied.
__global__ void reduce_block_summaries_kernel(
    const ThreadSummary* partials,
    std::uint32_t producing_blocks,
    std::uint32_t producing_threads,
    ThreadSummary* summaries)
{
    const std::uint64_t block = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block >= producing_blocks) return;
    ThreadSummary summary{};
    for (std::uint32_t t = 0; t < producing_threads; ++t)
        merge_summary(summary, partials[block * producing_threads + t]);
    summaries[block] = summary;
}

template<class T>
std::size_t append_array(
    std::vector<unsigned char>& packed,
    const std::vector<T>& values,
    std::size_t budget)
{
    const std::size_t padding = (alignof(T) - packed.size() % alignof(T)) % alignof(T);
    if (packed.size() > budget || padding > budget - packed.size())
        throw std::runtime_error("Batch input exceeds the device-buffer budget");
    const std::size_t offset = packed.size() + padding;
    if (values.size() > (budget - offset) / sizeof(T))
        throw std::runtime_error("Batch input exceeds the device-buffer budget");
    const std::size_t bytes = values.size() * sizeof(T);
    packed.resize(offset + bytes);
    if (bytes) std::memcpy(packed.data() + offset, values.data(), bytes);
    return offset;
}

bool add_count(std::uint64_t& total, std::uint64_t value)
{
    if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += value;
    return true;
}
} // namespace

bool cuda_screen_ligand_batch(
    const GpuFlatTripletIndexDevice& protein_index,
    const GpuPointArrayDevice& protein_targets,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuScreeningBatchLigand>& ligands,
    const GpuScreeningBatchOptions& options,
    std::vector<GpuScreeningBatchResult>& results,
    GpuScreeningBatchTimings& timings,
    std::string& error)
{
    results.clear();
    timings = {};
    error.clear();
    unsigned char* device_packed = nullptr;
    ThreadSummary* device_summaries = nullptr;
    ThreadSummary* device_reduced_summaries = nullptr;
    cudaEvent_t start_event = nullptr, stop_event = nullptr;

    const auto check = [](cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    };
    const auto sample_memory = [&]() {
        std::size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        if (timings.total_device_bytes == 0 || free_bytes < timings.min_free_device_bytes)
            timings.min_free_device_bytes = free_bytes;
        timings.total_device_bytes = total_bytes;
    };
    const auto cleanup = [&]() {
        const auto start = Clock::now();
        cudaError_t status = cudaSuccess;
        const auto release = [&](cudaError_t next) { if (status == cudaSuccess) status = next; };
        if (stop_event) { release(cudaEventDestroy(stop_event)); stop_event = nullptr; }
        if (start_event) { release(cudaEventDestroy(start_event)); start_event = nullptr; }
        if (device_reduced_summaries) {
            release(cudaFree(device_reduced_summaries));
            device_reduced_summaries = nullptr;
        }
        if (device_summaries) { release(cudaFree(device_summaries)); device_summaries = nullptr; }
        if (device_packed) { release(cudaFree(device_packed)); device_packed = nullptr; }
        timings.cleanup_seconds += seconds_since(start);
        return status;
    };

    try {
        if (options.strategy != GpuScreeningBatchStrategy::WarpPerLigand &&
            options.strategy != GpuScreeningBatchStrategy::BlockPerLigand &&
            options.strategy != GpuScreeningBatchStrategy::MultiBlockPerLigand)
            throw std::runtime_error("Unknown batch strategy");
        const bool multi_block = options.strategy == GpuScreeningBatchStrategy::MultiBlockPerLigand;
        if (multi_block && options.blocks_per_ligand == 0)
            throw std::runtime_error("Multi-block strategy requires at least one block per ligand");
        const std::uint32_t blocks_per_ligand = multi_block ? options.blocks_per_ligand : 1;
        const std::uint32_t threads = options.strategy == GpuScreeningBatchStrategy::WarpPerLigand
            ? 32 : options.block_size;
        if (threads < 32 || threads > 1024 || threads % 32 != 0)
            throw std::runtime_error("Block size must be a multiple of 32 between 32 and 1024");
        if (!std::isfinite(options.clash_threshold) || options.clash_threshold < 0.0 ||
            options.clash_threshold > 1.0)
            throw std::runtime_error("Clash threshold must be finite and between zero and one");
        if (ligands.empty()) return true;
        if (ligands.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / blocks_per_ligand)
            throw std::runtime_error("Too many ligand blocks for a one-dimensional CUDA grid");
        const std::size_t producing_blocks = ligands.size() * blocks_per_ligand;
        if ((protein_index.key_count && (!protein_index.keys || !protein_index.offsets)) ||
            (protein_index.record_count && !protein_index.records) ||
            (protein_index.key_count &&
             static_cast<std::uint64_t>(protein_index.offset_count) !=
                 static_cast<std::uint64_t>(protein_index.key_count) + 1) ||
            (protein_targets.count && !protein_targets.points))
            throw std::runtime_error("Invalid borrowed protein index or target buffer");
        if (!protein_grid.voxels || !protein_grid.voxel_count || protein_grid.nx <= 0 ||
            protein_grid.ny <= 0 || protein_grid.nz <= 0 ||
            !std::isfinite(protein_grid.voxel_size) || protein_grid.voxel_size <= 0.0)
            throw std::runtime_error("Invalid borrowed protein grid");
        const std::uint64_t xy = static_cast<std::uint64_t>(protein_grid.nx) * protein_grid.ny;
        if (xy > protein_grid.voxel_count ||
            xy * static_cast<std::uint64_t>(protein_grid.nz) != protein_grid.voxel_count)
            throw std::runtime_error("Protein grid dimensions do not match the voxel buffer");

        const auto pack_start = Clock::now();
        if (producing_blocks > std::numeric_limits<std::size_t>::max() / threads)
            throw std::runtime_error("Batch summary count exceeds size_t capacity");
        const std::size_t summary_count = producing_blocks * threads;
        if (summary_count > options.max_device_bytes / sizeof(ThreadSummary))
            throw std::runtime_error("Batch summaries exceed the device-buffer budget");
        const std::size_t summary_bytes = summary_count * sizeof(ThreadSummary);
        const std::size_t reduced_count = multi_block ? producing_blocks : 0;
        if (reduced_count > (options.max_device_bytes - summary_bytes) / sizeof(ThreadSummary))
            throw std::runtime_error("Reduced batch summaries exceed the device-buffer budget");
        const std::size_t reduced_bytes = reduced_count * sizeof(ThreadSummary);
        const std::size_t input_budget = options.max_device_bytes - summary_bytes - reduced_bytes;
        if (ligands.size() > input_budget / sizeof(LigandDescriptor))
            throw std::runtime_error("Batch descriptors exceed the device-buffer budget");
        std::vector<LigandDescriptor> descriptors(ligands.size());
        std::vector<unsigned char> packed(ligands.size() * sizeof(LigandDescriptor));
        for (std::size_t i = 0; i < ligands.size(); ++i) {
            const auto& ligand = ligands[i];
            const auto u32max = std::numeric_limits<std::uint32_t>::max();
            if (ligand.queries.size() != ligand.orders.size() ||
                ligand.queries.size() > u32max || ligand.matching_points.size() > u32max ||
                ligand.atoms.size() > u32max || ligand.fast_areas.size() > u32max)
                throw std::runtime_error("Invalid ligand array sizes");
            for (const auto& order : ligand.orders) {
                if (order.equivalent_order_count > 6)
                    throw std::runtime_error("A ligand query has more than six equivalent orders");
                for (std::uint32_t o = 0; o < order.equivalent_order_count; ++o)
                    for (int k = 0; k < 3; ++k)
                        if (order.equivalent_orders[o][k] >= ligand.matching_points.size())
                            throw std::runtime_error("A ligand order addresses a missing matching point");
            }
            auto& descriptor = descriptors[i];
            descriptor.query_count = static_cast<std::uint32_t>(ligand.queries.size());
            descriptor.atom_count = static_cast<std::uint32_t>(ligand.atoms.size());
            descriptor.area_count = static_cast<std::uint32_t>(ligand.fast_areas.size());
            descriptor.queries_offset = append_array(packed, ligand.queries, input_budget);
            descriptor.orders_offset = append_array(packed, ligand.orders, input_budget);
            descriptor.matching_offset = append_array(packed, ligand.matching_points, input_budget);
            descriptor.atoms_offset = append_array(packed, ligand.atoms, input_budget);
            descriptor.areas_offset = append_array(packed, ligand.fast_areas, input_budget);
        }
        std::memcpy(packed.data(), descriptors.data(), descriptors.size() * sizeof(LigandDescriptor));
        const std::size_t downloaded_count = multi_block ? reduced_count : summary_count;
        const std::size_t downloaded_bytes = downloaded_count * sizeof(ThreadSummary);
        std::vector<ThreadSummary> summaries(downloaded_count);
        timings.pack_seconds = seconds_since(pack_start);
        timings.batch_device_bytes = packed.size() + summary_bytes + reduced_bytes;
        timings.h2d_bytes = packed.size();
        timings.d2h_bytes = downloaded_bytes;

        const auto allocation_start = Clock::now();
        sample_memory();
        check(cudaMalloc(reinterpret_cast<void**>(&device_packed), packed.size()), "Allocate batch inputs");
        check(cudaMalloc(reinterpret_cast<void**>(&device_summaries), summary_bytes), "Allocate batch summaries");
        if (multi_block)
            check(cudaMalloc(reinterpret_cast<void**>(&device_reduced_summaries), reduced_bytes),
                "Allocate reduced batch summaries");
        check(cudaEventCreate(&start_event), "Create batch start event");
        check(cudaEventCreate(&stop_event), "Create batch stop event");
        sample_memory();
        timings.allocation_seconds = seconds_since(allocation_start);

        const auto upload_start = Clock::now();
        check(cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice), "Upload batch");
        check(cudaDeviceSynchronize(), "Complete batch upload");
        timings.h2d_seconds = seconds_since(upload_start);

        check(cudaEventRecord(start_event), "Record batch start");
        if (multi_block) {
            screen_multi_block_kernel<<<static_cast<unsigned int>(producing_blocks), threads>>>(
                device_packed, protein_index, protein_targets, protein_grid,
                options.clash_threshold, blocks_per_ligand, device_summaries);
            check(cudaGetLastError(), "Launch multi-block batch kernel");
            constexpr std::uint32_t reduction_threads = 128;
            const std::size_t reduction_blocks = (producing_blocks + reduction_threads - 1) / reduction_threads;
            reduce_block_summaries_kernel<<<static_cast<unsigned int>(reduction_blocks), reduction_threads>>>(
                device_summaries, static_cast<std::uint32_t>(producing_blocks), threads,
                device_reduced_summaries);
            check(cudaGetLastError(), "Launch batch summary reduction kernel");
        }
        else {
            screen_batch_kernel<<<static_cast<unsigned int>(ligands.size()), threads>>>(
                device_packed, protein_index, protein_targets, protein_grid,
                options.clash_threshold, device_summaries);
            check(cudaGetLastError(), "Launch batch kernel");
        }
        check(cudaEventRecord(stop_event), "Record batch stop");
        check(cudaEventSynchronize(stop_event), "Complete batch kernel");
        float kernel_ms = 0.0f;
        check(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "Measure batch kernel");
        timings.kernel_seconds = static_cast<double>(kernel_ms) / 1000.0;
        sample_memory();

        const auto download_start = Clock::now();
        check(cudaMemcpy(summaries.data(), multi_block ? device_reduced_summaries : device_summaries,
                downloaded_bytes, cudaMemcpyDeviceToHost),
            "Download batch summaries");
        timings.d2h_seconds = seconds_since(download_start);

        const auto reduction_start = Clock::now();
        results.resize(ligands.size());
        for (std::size_t i = 0; i < ligands.size(); ++i) {
            auto& result = results[i];
            PoseSummary first{}, best{};
            const std::uint32_t summaries_per_ligand = multi_block ? blocks_per_ligand : threads;
            for (std::uint32_t t = 0; t < summaries_per_ligand; ++t) {
                const auto& summary = summaries[i * summaries_per_ligand + t];
                if (summary.error & 1u)
                    throw std::runtime_error("Shared index contains invalid record offsets or target indices");
                if ((summary.error & 2u) ||
                    !add_count(result.candidate_matches, summary.matches) ||
                    !add_count(result.transforms_generated, summary.transforms) ||
                    !add_count(result.valid_transforms, summary.valid) ||
                    !add_count(result.survivors, summary.survivors))
                    throw std::runtime_error("Batch count exceeds uint64 capacity; no result was truncated");
                if (summary.first.present && (!first.present || earlier(summary.first, first)))
                    first = summary.first;
                const auto& candidate = summary.best_numeric;
                if (candidate.present && (!best.present ||
                    candidate.fast.weighted_score > best.fast.weighted_score ||
                    (candidate.fast.weighted_score == best.fast.weighted_score && earlier(candidate, best))))
                    best = candidate;
            }
            if (first.present) {
                // Match the baseline's strict best-score update: an initial NaN
                // remains selected; finite ties retain the earliest pose.
                const auto& winner = std::isnan(first.fast.weighted_score) ? first : best;
                result.has_top_pose = true;
                result.top_transform = winner.transform;
                result.top_clash = winner.clash;
                result.top_fast_score = winner.fast;
                result.top_query_index = winner.query;
                result.top_neighbor_slot = winner.neighbor;
                result.top_record_index = winner.record;
                result.top_order_index = winner.order;
            }
        }
        timings.reduction_seconds = seconds_since(reduction_start);
        check(cleanup(), "Release batch resources");
        return true;
    }
    catch (const std::exception& exception) {
        error = exception.what();
        const auto cleanup_status = cleanup();
        if (cleanup_status != cudaSuccess)
            error += std::string("; cleanup: ") + cudaGetErrorString(cleanup_status);
        results.clear();
        return false;
    }
}
