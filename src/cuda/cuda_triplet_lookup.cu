#include "cuda_backend.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <cub/cub.cuh>


namespace
{

using docking_device::find_triplet_key;


__global__ void triplet_exact_lookup_kernel(
    const GpuTripletKey* index_keys,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* results
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (query_index >= query_count)
        return;


    results[query_index] =
        find_triplet_key(
            index_keys,
            index_key_count,
            queries[query_index]
        );
}

__global__ void triplet_neighbor_lookup_kernel(
    const GpuTripletKey* index_keys,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* results
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];


    std::uint32_t slot = 0;


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate =
                    base;


                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };


                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }


                    candidate.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                const std::uint32_t output_index =
                    query_index * 27 + slot;


                if (valid) {
                    results[output_index] =
                        find_triplet_key(
                            index_keys,
                            index_key_count,
                            candidate
                        );
                }
                else {
                    results[output_index] =
                        GPU_TRIPLET_KEY_NOT_FOUND;
                }


                ++slot;
            }
        }
    }
}

// Count and expansion must use the same bin traversal. The prefix scan gives
// each query a disjoint output range, preserving query order despite parallelism.
__global__ void triplet_neighbor_count_kernel(
    const GpuTripletKey* index_keys,
    const std::uint32_t* index_offsets,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* counts
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];

    std::uint32_t count = 0;


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate =
                    base;

                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };

                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }

                    candidate.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                if (!valid)
                    continue;


                const std::uint32_t key_index =
                    find_triplet_key(
                        index_keys,
                        index_key_count,
                        candidate
                    );


                if (
                    key_index ==
                    GPU_TRIPLET_KEY_NOT_FOUND
                ) {
                    continue;
                }


                const std::uint32_t begin =
                    index_offsets[key_index];

                const std::uint32_t end =
                    index_offsets[key_index + 1];


                count += end - begin;
            }
        }
    }


    counts[query_index] = count;
}

__global__ void validate_alignment_scan_overflow_kernel(
    const std::uint32_t* counts,
    const std::uint32_t* offsets,
    std::uint32_t count,
    std::uint32_t* overflow
)
{
    const std::uint32_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
        return;

    if (counts[i] > 0xffffffffu - offsets[i])
        atomicExch(overflow, 1u);
}


__global__ void write_final_prefix_offset_kernel(
    const std::uint32_t* counts,
    std::uint32_t count,
    std::uint32_t* offsets
)
{
    if (
        blockIdx.x == 0 &&
        threadIdx.x == 0
    ) {
        offsets[count] =
            offsets[count - 1] +
            counts[count - 1];
    }
}


__global__ void validate_device_triplet_candidate_offsets_kernel(
    const std::uint32_t* counts,
    const std::uint32_t* offsets,
    std::uint32_t count,
    std::uint32_t* invalid
)
{
    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;

    if (i >= count)
        return;

    const std::uint32_t begin = offsets[i];
    const std::uint32_t end = offsets[i + 1];

    if (
        (i == 0 && begin != 0) ||
        end < begin ||
        end - begin != counts[i]
    ) {
        atomicExch(invalid, 1u);
    }
}


// Within each query, emit d0/d1/d2 neighbor order, then stored bucket order.
// No real-distance cutoff is reapplied after this binned lookup.
__global__ void triplet_neighbor_expand_kernel(
    const GpuTripletKey* index_keys,
    const std::uint32_t* index_offsets,
    const GpuProteinTripletRecord* index_records,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    const std::uint32_t* candidate_offsets,
    GpuTripletCandidate* candidates
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];


    std::uint32_t output_index =
        candidate_offsets[query_index];


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate_key =
                    base;


                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };


                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate_key.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }


                    candidate_key.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate_key.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                if (!valid)
                    continue;


                const std::uint32_t key_index =
                    find_triplet_key(
                        index_keys,
                        index_key_count,
                        candidate_key
                    );


                if (
                    key_index ==
                    GPU_TRIPLET_KEY_NOT_FOUND
                ) {
                    continue;
                }


                const std::uint32_t begin =
                    index_offsets[key_index];

                const std::uint32_t end =
                    index_offsets[key_index + 1];


                for (
                    std::uint32_t record_index = begin;
                    record_index < end;
                    ++record_index
                ) {
                    const auto record =
                        index_records[record_index];


                    GpuTripletCandidate output{};

                    output.ligand_triplet_index =
                        query_index;

                    output.protein_indices[0] =
                        record.indices[0];

                    output.protein_indices[1] =
                        record.indices[1];

                    output.protein_indices[2] =
                        record.indices[2];


                    candidates[output_index] =
                        output;


                    ++output_index;
                }
            }
        }
    }
}

__global__ void alignment_candidate_count_kernel(
    const GpuTripletCandidate* candidates,
    std::uint32_t candidate_count,
    const GpuLigandTripletOrders* ligand_orders,
    std::uint32_t ligand_order_count,
    std::uint32_t* alignment_counts,
    std::uint32_t* invalid_flag
)
{
    const std::uint32_t candidate_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (candidate_index >= candidate_count)
        return;

    const std::uint32_t ligand_triplet_index =
        candidates[candidate_index].ligand_triplet_index;

    if (ligand_triplet_index >= ligand_order_count) {
        alignment_counts[candidate_index] = 0;

        if (invalid_flag != nullptr)
            atomicExch(invalid_flag, 1u);

        return;
    }

    const std::uint32_t equivalent_order_count =
        ligand_orders[ligand_triplet_index]
            .equivalent_order_count;

    if (equivalent_order_count > 6) {
        alignment_counts[candidate_index] = 0;

        if (invalid_flag != nullptr)
            atomicExch(invalid_flag, 1u);

        return;
    }

    alignment_counts[candidate_index] =
        equivalent_order_count;
}


__global__ void validate_alignment_expansion_inputs_kernel(
    const GpuTripletCandidate* candidates,
    std::uint32_t candidate_count,
    const GpuLigandTripletOrders* ligand_orders,
    std::uint32_t ligand_order_count,
    const std::uint32_t* offsets,
    std::uint32_t* invalid
)
{
    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;

    if (i >= candidate_count)
        return;

    const std::uint32_t ligand_index =
        candidates[i].ligand_triplet_index;

    if (ligand_index >= ligand_order_count) {
        atomicExch(invalid, 1u);
        return;
    }

    const std::uint32_t expected =
        ligand_orders[ligand_index]
            .equivalent_order_count;

    const std::uint32_t begin = offsets[i];
    const std::uint32_t end = offsets[i + 1];

    if (
        expected > 6 ||
        (i == 0 && begin != 0) ||
        end < begin ||
        end - begin != expected
    ) {
        atomicExch(invalid, 1u);
    }
}


// Expand each record in its stored equivalent-order sequence, without sorting.
__global__ void alignment_candidate_expand_kernel(
    const GpuTripletCandidate* candidates,
    std::uint32_t candidate_count,
    const GpuLigandTripletOrders* ligand_orders,
    const std::uint32_t* alignment_offsets,
    GpuAlignmentCandidate* alignments
)
{
    const std::uint32_t candidate_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (candidate_index >= candidate_count)
        return;


    const GpuTripletCandidate candidate =
        candidates[candidate_index];


    const GpuLigandTripletOrders orders =
        ligand_orders[
            candidate.ligand_triplet_index
        ];


    const std::uint32_t output_begin =
        alignment_offsets[candidate_index];


    for (
        std::uint32_t order_index = 0;
        order_index <
            orders.equivalent_order_count;
        ++order_index
    ) {
        GpuAlignmentCandidate output{};


        for (int i = 0; i < 3; ++i) {

            output.ligand_indices[i] =
                orders.equivalent_orders[
                    order_index
                ][i];

            output.protein_indices[i] =
                candidate.protein_indices[i];
        }


        alignments[
            output_begin + order_index
        ] = output;
    }
}

} // namespace


bool cuda_lookup_triplet_keys_exact(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& key_indices
)
{
    key_indices.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.key_count > 0 &&
        device_index.keys == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries = nullptr;

    std::uint32_t* device_results =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t result_bytes =
        queries.size() *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    constexpr std::uint32_t block_size = 256;

    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );

    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_exact_lookup_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.key_count,
            device_queries,
            query_count,
            device_results
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    key_indices.resize(
        queries.size()
    );


    if (
        cudaMemcpy(
            key_indices.data(),
            device_results,
            result_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        key_indices.clear();

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_results);
    cudaFree(device_queries);


    return true;
}

bool cuda_lookup_triplet_key_neighbors(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& neighbor_key_indices
)
{
    neighbor_key_indices.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.key_count > 0 &&
        device_index.keys == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_results =
        nullptr;


    const std::size_t result_count =
        queries.size() * 27;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t result_bytes =
        result_count *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_lookup_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.key_count,
            device_queries,
            query_count,
            device_results
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    neighbor_key_indices.resize(
        result_count
    );


    if (
        cudaMemcpy(
            neighbor_key_indices.data(),
            device_results,
            result_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        neighbor_key_indices.clear();

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_results);
    cudaFree(device_queries);


    return true;
}

bool cuda_count_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    GpuTripletNeighborCountArrayDevice& counts
)
{
    if (
        counts.counts != nullptr ||
        counts.count != 0
    ) {
        return false;
    }

    if (queries.empty())
        return true;

    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }

    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr
    ) {
        return false;
    }

    GpuTripletKey* device_queries = nullptr;
    std::uint32_t* device_counts = nullptr;

    const auto cleanup = [&]() {
        cudaFree(device_counts);
        cudaFree(device_queries);
    };

    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t count_bytes =
        queries.size() *
        sizeof(std::uint32_t);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            query_bytes
        ) != cudaSuccess ||
        cudaMalloc(
            reinterpret_cast<void**>(&device_counts),
            count_bytes
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );

    constexpr std::uint32_t block_size = 256;

    const std::uint32_t block_count =
        (query_count - 1) / block_size + 1;

    // Reuse the existing neighbor counting kernel.
    triplet_neighbor_count_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.key_count,
            device_queries,
            query_count,
            device_counts
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    // Queries are no longer needed, but the counts
    // must remain device-resident.
    cudaFree(device_queries);

    counts.counts = device_counts;
    counts.count = query_count;

    return true;
}


bool cuda_download_triplet_neighbor_counts(
    const GpuTripletNeighborCountArrayDevice& device_counts,
    std::vector<std::uint32_t>& host_counts
)
{
    host_counts.clear();

    if (device_counts.count == 0)
        return device_counts.counts == nullptr;

    if (device_counts.counts == nullptr)
        return false;

    const std::size_t count_bytes =
        static_cast<std::size_t>(
            device_counts.count
        ) *
        sizeof(std::uint32_t);

    host_counts.resize(device_counts.count);

    if (
        cudaMemcpy(
            host_counts.data(),
            device_counts.counts,
            count_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_counts.clear();
        return false;
    }

    return true;
}


void cuda_free_triplet_neighbor_counts(
    GpuTripletNeighborCountArrayDevice& counts
)
{
    cudaFree(counts.counts);
    counts = {};
}


bool cuda_count_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& record_counts
)
{
    record_counts.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_counts =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t count_bytes =
        queries.size() *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_counts
            ),
            count_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    constexpr std::uint32_t block_size =
        256;

    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_count_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.key_count,
            device_queries,
            query_count,
            device_counts
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    record_counts.resize(
        queries.size()
    );


    if (
        cudaMemcpy(
            record_counts.data(),
            device_counts,
            count_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        record_counts.clear();

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_counts);
    cudaFree(device_queries);


    return true;
}

bool cuda_exclusive_prefix_sum(
    const std::vector<std::uint32_t>& counts,
    std::vector<std::uint32_t>& offsets
)
{
    offsets.clear();


    if (counts.empty()) {
        offsets.push_back(0);
        return true;
    }


    if (
        counts.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    std::uint64_t total = 0;

    for (const auto count : counts) {

        total += count;

        if (
            total >
            std::numeric_limits<std::uint32_t>::max()
        ) {
            return false;
        }
    }


    const std::uint32_t count =
        static_cast<std::uint32_t>(
            counts.size()
        );


    std::uint32_t* device_counts =
        nullptr;

    std::uint32_t* device_offsets =
        nullptr;

    void* temporary_storage =
        nullptr;


    const std::size_t count_bytes =
        counts.size() *
        sizeof(std::uint32_t);

    const std::size_t offset_bytes =
        (counts.size() + 1) *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_counts
            ),
            count_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_offsets
            ),
            offset_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_counts);
        return false;
    }


    if (
        cudaMemcpy(
            device_counts,
            counts.data(),
            count_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    std::size_t temporary_bytes = 0;


    if (
        cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            temporary_bytes,
            device_counts,
            device_offsets,
            count
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (
        cudaMalloc(
            &temporary_storage,
            temporary_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (
        cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            temporary_bytes,
            device_counts,
            device_offsets,
            count
        ) != cudaSuccess
    ) {
        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    write_final_prefix_offset_kernel
        <<<1, 1>>>(
            device_counts,
            count,
            device_offsets
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    offsets.resize(
        counts.size() + 1
    );


    if (
        cudaMemcpy(
            offsets.data(),
            device_offsets,
            offset_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        offsets.clear();

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    cudaFree(temporary_storage);
    cudaFree(device_offsets);
    cudaFree(device_counts);


    return true;
}

bool cuda_expand_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const GpuTripletNeighborCountArrayDevice& device_counts,
    const GpuTripletNeighborOffsetArrayDevice& device_offsets,
    GpuTripletCandidateArrayDevice& candidates
)
{
    if (
        candidates.candidates != nullptr ||
        candidates.count != 0 ||
        queries.size() >=
            std::numeric_limits<std::uint32_t>::max() ||
        queries.size() != device_counts.count ||
        (device_counts.count == 0 &&
            device_counts.counts != nullptr) ||
        (device_counts.count > 0 &&
            device_counts.counts == nullptr) ||
        device_offsets.offsets == nullptr ||
        device_offsets.count != device_counts.count + 1
    ) {
        return false;
    }

    // Read only the final offset to determine output size.
    // The complete offsets array stays on the GPU.
    std::uint32_t total_candidates = 0;

    if (
        cudaMemcpy(
            &total_candidates,
            device_offsets.offsets + device_counts.count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        return false;
    }

    if (queries.empty())
        return total_candidates == 0;

    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr ||
        (
            device_index.record_count > 0 &&
            device_index.records == nullptr
        )
    ) {
        return false;
    }

    GpuTripletKey* device_queries = nullptr;
    std::uint32_t* device_invalid = nullptr;
    GpuTripletCandidate* device_candidates = nullptr;

    const auto cleanup = [&]() {
        cudaFree(device_candidates);
        cudaFree(device_invalid);
        cudaFree(device_queries);
    };

    const std::size_t query_bytes =
        queries.size() * sizeof(GpuTripletKey);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            query_bytes
        ) != cudaSuccess ||
        cudaMalloc(
            reinterpret_cast<void**>(&device_invalid),
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess ||
        cudaMemset(
            device_invalid,
            0,
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    const std::uint32_t query_count =
        static_cast<std::uint32_t>(queries.size());

    constexpr std::uint32_t block_size = 256;

    const std::uint32_t block_count =
        (query_count - 1) / block_size + 1;

    // Verify GPU offsets before any candidate write.
    validate_device_triplet_candidate_offsets_kernel
        <<<block_count, block_size>>>(
            device_counts.counts,
            device_offsets.offsets,
            query_count,
            device_invalid
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    std::uint32_t invalid = 0;

    if (
        cudaMemcpy(
            &invalid,
            device_invalid,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||
        invalid != 0
    ) {
        cleanup();
        return false;
    }

    // Valid input may contain no candidates.
    if (total_candidates == 0) {
        cleanup();
        return true;
    }

    const std::size_t candidate_bytes =
        static_cast<std::size_t>(total_candidates) *
        sizeof(GpuTripletCandidate);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_candidates),
            candidate_bytes
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    triplet_neighbor_expand_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.records,
            device_index.key_count,
            device_queries,
            query_count,
            device_offsets.offsets,
            device_candidates
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    candidates.candidates = device_candidates;
    candidates.count = total_candidates;

    // Ownership transferred to the caller.
    device_candidates = nullptr;

    cleanup();
    return true;
}


bool cuda_expand_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    GpuTripletCandidateArrayDevice& candidates
)
{
    if (
        candidates.candidates != nullptr ||
        candidates.count != 0
    ) {
        return false;
    }


    if (
        candidate_offsets.size() !=
        queries.size() + 1
    ) {
        return false;
    }


    if (queries.empty()) {

        return (
            candidate_offsets.size() == 1 &&
            candidate_offsets[0] == 0
        );
    }


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    for (
        std::size_t i = 1;
        i < candidate_offsets.size();
        ++i
    ) {
        if (
            candidate_offsets[i] <
            candidate_offsets[i - 1]
        ) {
            return false;
        }
    }


    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr ||
        (
            device_index.record_count > 0 &&
            device_index.records == nullptr
        )
    ) {
        return false;
    }


    const std::uint32_t total_candidates =
        candidate_offsets.back();


    if (total_candidates == 0)
        return true;


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_candidate_offsets =
        nullptr;

    GpuTripletCandidate* device_candidates =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t offset_bytes =
        candidate_offsets.size() *
        sizeof(std::uint32_t);

    const std::size_t candidate_bytes =
        static_cast<std::size_t>(
            total_candidates
        ) *
        sizeof(GpuTripletCandidate);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidate_offsets
            ),
            offset_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidates
            ),
            candidate_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (
        cudaMemcpy(
            device_candidate_offsets,
            candidate_offsets.data(),
            offset_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_expand_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.records,
            device_index.key_count,
            device_queries,
            query_count,
            device_candidate_offsets,
            device_candidates
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    candidates.candidates = device_candidates;
    candidates.count = total_candidates;

    cudaFree(device_candidate_offsets);
    cudaFree(device_queries);

    return true;
}


bool cuda_download_triplet_candidates(
    const GpuTripletCandidateArrayDevice& device_candidates,
    std::vector<GpuTripletCandidate>& host_candidates
)
{
    host_candidates.clear();

    if (device_candidates.count == 0)
        return device_candidates.candidates == nullptr;

    if (device_candidates.candidates == nullptr)
        return false;

    host_candidates.resize(device_candidates.count);

    const std::size_t bytes =
        static_cast<std::size_t>(device_candidates.count) *
        sizeof(GpuTripletCandidate);

    if (
        cudaMemcpy(
            host_candidates.data(),
            device_candidates.candidates,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_candidates.clear();
        return false;
    }

    return true;
}


void cuda_free_triplet_candidates(
    GpuTripletCandidateArrayDevice& candidates
)
{
    cudaFree(candidates.candidates);
    candidates = {};
}


bool cuda_expand_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    std::vector<GpuTripletCandidate>& candidates
)
{
    candidates.clear();

    GpuTripletCandidateArrayDevice device_candidates{};

    if (
        !cuda_expand_triplet_neighbor_records_device(
            device_index,
            queries,
            candidate_offsets,
            device_candidates
        )
    ) {
        return false;
    }

    const bool success =
        cuda_download_triplet_candidates(
            device_candidates,
            candidates
        );

    cuda_free_triplet_candidates(device_candidates);

    return success;
}

bool cuda_count_alignment_candidates_device(
    const GpuTripletCandidateArrayDevice& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    GpuAlignmentCountArrayDevice& counts
)
{
    if (counts.count != 0 || counts.counts != nullptr)
        return false;

    if (candidates.count == 0)
        return candidates.candidates == nullptr;

    if (
        candidates.candidates == nullptr ||
        ligand_orders.empty() ||
        ligand_orders.size() >
            std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }

    for (const auto& orders : ligand_orders) {
        if (orders.equivalent_order_count > 6)
            return false;
    }

    GpuLigandTripletOrders* device_orders = nullptr;
    std::uint32_t* device_counts = nullptr;
    std::uint32_t* device_invalid = nullptr;

    const std::size_t order_bytes =
        ligand_orders.size() *
        sizeof(GpuLigandTripletOrders);

    const std::size_t count_bytes =
        static_cast<std::size_t>(candidates.count) *
        sizeof(std::uint32_t);

    const auto cleanup = [&]() {
        if (device_invalid != nullptr)
            cudaFree(device_invalid);

        if (device_orders != nullptr)
            cudaFree(device_orders);
    };

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_orders),
            order_bytes
        ) != cudaSuccess
        ||
        cudaMalloc(
            reinterpret_cast<void**>(&device_counts),
            count_bytes
        ) != cudaSuccess
        ||
        cudaMalloc(
            reinterpret_cast<void**>(&device_invalid),
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        if (device_counts != nullptr)
            cudaFree(device_counts);
        return false;
    }

    if (
        cudaMemcpy(
            device_orders,
            ligand_orders.data(),
            order_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
        ||
        cudaMemset(
            device_invalid,
            0,
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        cudaFree(device_counts);
        return false;
    }

    constexpr std::uint32_t block_size = 256;

    const std::uint32_t block_count =
        (candidates.count + block_size - 1) /
        block_size;

    alignment_candidate_count_kernel
        <<<block_count, block_size>>>(
            candidates.candidates,
            candidates.count,
            device_orders,
            static_cast<std::uint32_t>(
                ligand_orders.size()
            ),
            device_counts,
            device_invalid
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        cudaFree(device_counts);
        return false;
    }

    std::uint32_t invalid = 0;

    const bool valid =
        cudaMemcpy(
            &invalid,
            device_invalid,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) == cudaSuccess
        &&
        invalid == 0;

    cleanup();

    if (!valid) {
        cudaFree(device_counts);
        return false;
    }

    counts.counts = device_counts;
    counts.count = candidates.count;

    return true;
}


bool cuda_download_alignment_counts(
    const GpuAlignmentCountArrayDevice& device_counts,
    std::vector<std::uint32_t>& host_counts
)
{
    host_counts.clear();

    if (device_counts.count == 0)
        return device_counts.counts == nullptr;

    if (device_counts.counts == nullptr)
        return false;

    host_counts.resize(device_counts.count);

    if (
        cudaMemcpy(
            host_counts.data(),
            device_counts.counts,
            static_cast<std::size_t>(device_counts.count) *
                sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_counts.clear();
        return false;
    }

    return true;
}


void cuda_free_alignment_counts(
    GpuAlignmentCountArrayDevice& counts
)
{
    if (counts.counts != nullptr)
        cudaFree(counts.counts);

    counts = {};
}


bool cuda_exclusive_prefix_sum_device(
    const GpuAlignmentCountArrayDevice& counts,
    GpuAlignmentOffsetArrayDevice& offsets
)
{
    if (offsets.offsets != nullptr || offsets.count != 0)
        return false;

    if (
        (counts.count == 0 && counts.counts != nullptr) ||
        (counts.count > 0 && counts.counts == nullptr) ||
        counts.count ==
            std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }

    std::uint32_t* device_offsets = nullptr;
    std::uint32_t* device_overflow = nullptr;
    void* temporary_storage = nullptr;

    const auto cleanup = [&]() {
        cudaFree(temporary_storage);
        cudaFree(device_overflow);
        cudaFree(device_offsets);
    };

    const std::size_t offset_bytes =
        (static_cast<std::size_t>(counts.count) + 1) *
        sizeof(std::uint32_t);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_offsets),
            offset_bytes
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    // Empty input produces exactly one offset: zero.
    if (counts.count == 0) {
        if (
            cudaMemset(
                device_offsets,
                0,
                sizeof(std::uint32_t)
            ) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess
        ) {
            cleanup();
            return false;
        }

        offsets.offsets = device_offsets;
        offsets.count = 1;
        return true;
    }

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_overflow),
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cudaMemset(
            device_overflow,
            0,
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    std::size_t temporary_bytes = 0;

    if (
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            temporary_bytes,
            counts.counts,
            device_offsets,
            counts.count
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cudaMalloc(
            &temporary_storage,
            temporary_bytes == 0 ? 1 : temporary_bytes
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            temporary_bytes,
            counts.counts,
            device_offsets,
            counts.count
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    constexpr std::uint32_t block_size = 256;

    const std::uint32_t block_count =
        (counts.count - 1) / block_size + 1;

    validate_alignment_scan_overflow_kernel
        <<<block_count, block_size>>>(
            counts.counts,
            device_offsets,
            counts.count,
            device_overflow
        );

    if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return false;
    }

    write_final_prefix_offset_kernel
        <<<1, 1>>>(
            counts.counts,
            counts.count,
            device_offsets
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    std::uint32_t overflow = 0;

    if (
        cudaMemcpy(
            &overflow,
            device_overflow,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||
        overflow != 0
    ) {
        cleanup();
        return false;
    }

    cudaFree(temporary_storage);
    cudaFree(device_overflow);

    offsets.offsets = device_offsets;
    offsets.count = counts.count + 1;

    return true;
}


bool cuda_download_alignment_offsets(
    const GpuAlignmentOffsetArrayDevice& device_offsets,
    std::vector<std::uint32_t>& host_offsets
)
{
    host_offsets.clear();

    if (device_offsets.count == 0)
        return device_offsets.offsets == nullptr;

    if (device_offsets.offsets == nullptr)
        return false;

    host_offsets.resize(device_offsets.count);

    const std::size_t bytes =
        static_cast<std::size_t>(device_offsets.count) *
        sizeof(std::uint32_t);

    if (
        cudaMemcpy(
            host_offsets.data(),
            device_offsets.offsets,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_offsets.clear();
        return false;
    }

    return true;
}


void cuda_free_alignment_offsets(
    GpuAlignmentOffsetArrayDevice& offsets
)
{
    cudaFree(offsets.offsets);
    offsets = {};
}


bool cuda_count_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    std::vector<std::uint32_t>& alignment_counts
)
{
    alignment_counts.clear();


    if (candidates.empty())
        return true;


    if (
        candidates.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        ligand_orders.size() >
            std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    for (const auto& orders :
         ligand_orders) {

        if (
            orders.equivalent_order_count > 6
        ) {
            return false;
        }
    }


    for (const auto& candidate :
         candidates) {

        if (
            candidate.ligand_triplet_index >=
            ligand_orders.size()
        ) {
            return false;
        }
    }


    GpuTripletCandidate* device_candidates =
        nullptr;

    GpuLigandTripletOrders* device_orders =
        nullptr;

    std::uint32_t* device_counts =
        nullptr;


    const std::size_t candidate_bytes =
        candidates.size() *
        sizeof(GpuTripletCandidate);

    const std::size_t order_bytes =
        ligand_orders.size() *
        sizeof(GpuLigandTripletOrders);

    const std::size_t count_bytes =
        candidates.size() *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidates
            ),
            candidate_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_orders
            ),
            order_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_candidates);
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_counts
            ),
            count_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_orders);
        cudaFree(device_candidates);
        return false;
    }


    if (
        cudaMemcpy(
            device_candidates,
            candidates.data(),
            candidate_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess ||
        cudaMemcpy(
            device_orders,
            ligand_orders.data(),
            order_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_counts);
        cudaFree(device_orders);
        cudaFree(device_candidates);

        return false;
    }


    const std::uint32_t candidate_count =
        static_cast<std::uint32_t>(
            candidates.size()
        );


    constexpr std::uint32_t block_size =
        256;

    const std::uint32_t block_count =
        (
            candidate_count +
            block_size - 1
        ) / block_size;


    alignment_candidate_count_kernel
        <<<block_count, block_size>>>(
            device_candidates,
            candidate_count,
            device_orders,
            static_cast<std::uint32_t>(
                ligand_orders.size()
            ),
            device_counts,
            nullptr
        );


    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cudaFree(device_counts);
        cudaFree(device_orders);
        cudaFree(device_candidates);

        return false;
    }


    alignment_counts.resize(
        candidates.size()
    );


    if (
        cudaMemcpy(
            alignment_counts.data(),
            device_counts,
            count_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        alignment_counts.clear();

        cudaFree(device_counts);
        cudaFree(device_orders);
        cudaFree(device_candidates);

        return false;
    }


    cudaFree(device_counts);
    cudaFree(device_orders);
    cudaFree(device_candidates);


    return true;
}


bool cuda_exclusive_triplet_neighbor_prefix_sum_device(
    const GpuTripletNeighborCountArrayDevice& counts,
    GpuTripletNeighborOffsetArrayDevice& offsets
)
{
    if (
        offsets.offsets != nullptr ||
        offsets.count != 0
    ) {
        return false;
    }

    // Non-owning view of the existing GPU counts.
    // The scan must not free the input buffer.
    const GpuAlignmentCountArrayDevice scan_counts{
        counts.counts,
        counts.count
    };

    GpuAlignmentOffsetArrayDevice scan_offsets{};

    // Reuse the validated CUB implementation,
    // including empty input and overflow checks.
    if (
        !cuda_exclusive_prefix_sum_device(
            scan_counts,
            scan_offsets
        )
    ) {
        return false;
    }

    // Transfer ownership of the resulting GPU
    // buffer to the neighbor-offset structure.
    offsets.offsets = scan_offsets.offsets;
    offsets.count = scan_offsets.count;

    scan_offsets = {};

    return true;
}


bool cuda_download_triplet_neighbor_offsets(
    const GpuTripletNeighborOffsetArrayDevice& device_offsets,
    std::vector<std::uint32_t>& host_offsets
)
{
    const GpuAlignmentOffsetArrayDevice scan_offsets{
        device_offsets.offsets,
        device_offsets.count
    };

    return cuda_download_alignment_offsets(
        scan_offsets,
        host_offsets
    );
}


void cuda_free_triplet_neighbor_offsets(
    GpuTripletNeighborOffsetArrayDevice& offsets
)
{
    GpuAlignmentOffsetArrayDevice scan_offsets{
        offsets.offsets,
        offsets.count
    };

    cuda_free_alignment_offsets(scan_offsets);

    offsets = {};
}


bool cuda_expand_alignment_candidates_device(
    const GpuTripletCandidateArrayDevice& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const GpuAlignmentOffsetArrayDevice& alignment_offsets,
    GpuAlignmentCandidateArrayDevice& alignments
)
{
    cuda_free_alignment_candidates(alignments);

    if (
        candidates.count ==
            std::numeric_limits<std::uint32_t>::max() ||
        (candidates.count == 0 &&
            candidates.candidates != nullptr) ||
        (candidates.count > 0 &&
            candidates.candidates == nullptr) ||
        alignment_offsets.offsets == nullptr ||
        alignment_offsets.count != candidates.count + 1 ||
        ligand_orders.size() >
            std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }

    // Only the final offset is transferred to the host.
    // Candidates and the offsets array remain on the GPU.
    std::uint32_t total_alignments = 0;

    if (
        cudaMemcpy(
            &total_alignments,
            alignment_offsets.offsets + candidates.count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        return false;
    }

    if (candidates.count == 0)
        return total_alignments == 0;

    if (ligand_orders.empty())
        return false;

    GpuLigandTripletOrders* device_orders = nullptr;
    std::uint32_t* device_invalid = nullptr;
    GpuAlignmentCandidate* device_alignments = nullptr;

    const auto cleanup = [&]() {
        cudaFree(device_alignments);
        cudaFree(device_invalid);
        cudaFree(device_orders);
    };

    const std::size_t order_bytes =
        ligand_orders.size() *
        sizeof(GpuLigandTripletOrders);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_orders),
            order_bytes
        ) != cudaSuccess ||
        cudaMalloc(
            reinterpret_cast<void**>(&device_invalid),
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    if (
        cudaMemcpy(
            device_orders,
            ligand_orders.data(),
            order_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess ||
        cudaMemset(
            device_invalid,
            0,
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    constexpr std::uint32_t block_size = 256;

    const std::uint32_t block_count =
        (candidates.count - 1) / block_size + 1;

    // Validate inputs before allowing the expansion kernel
    // to write into the output buffer.
    validate_alignment_expansion_inputs_kernel
        <<<block_count, block_size>>>(
            candidates.candidates,
            candidates.count,
            device_orders,
            static_cast<std::uint32_t>(
                ligand_orders.size()
            ),
            alignment_offsets.offsets,
            device_invalid
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    std::uint32_t invalid = 0;

    if (
        cudaMemcpy(
            &invalid,
            device_invalid,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||
        invalid != 0
    ) {
        cleanup();
        return false;
    }

    // A valid expansion may contain zero alignments.
    if (total_alignments == 0) {
        cleanup();
        return true;
    }

    const std::size_t alignment_bytes =
        static_cast<std::size_t>(total_alignments) *
        sizeof(GpuAlignmentCandidate);

    if (
        cudaMalloc(
            reinterpret_cast<void**>(&device_alignments),
            alignment_bytes
        ) != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    alignment_candidate_expand_kernel
        <<<block_count, block_size>>>(
            candidates.candidates,
            candidates.count,
            device_orders,
            alignment_offsets.offsets,
            device_alignments
        );

    if (
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess
    ) {
        cleanup();
        return false;
    }

    alignments.alignments = device_alignments;
    alignments.count = total_alignments;

    // Ownership of the output buffer moves to caller.
    device_alignments = nullptr;

    cleanup();
    return true;
}


bool cuda_expand_alignment_candidates_device(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const std::vector<std::uint32_t>& alignment_offsets,
    GpuAlignmentCandidateArrayDevice& alignments
)
{
    cuda_free_alignment_candidates(
        alignments
    );


    if (
        alignment_offsets.size() !=
        candidates.size() + 1
    ) {
        return false;
    }


    if (candidates.empty()) {

        return (
            alignment_offsets.size() == 1 &&
            alignment_offsets[0] == 0
        );
    }


    if (
        candidates.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    for (
        std::size_t i = 0;
        i < candidates.size();
        ++i
    ) {
        const auto ligand_index =
            candidates[i]
                .ligand_triplet_index;


        if (
            ligand_index >=
            ligand_orders.size()
        ) {
            return false;
        }


        const std::uint32_t expected =
            ligand_orders[ligand_index]
                .equivalent_order_count;


        if (expected > 6)
            return false;


        if (
            alignment_offsets[i + 1] -
            alignment_offsets[i] !=
            expected
        ) {
            return false;
        }
    }


    const std::uint32_t total_alignments =
        alignment_offsets.back();


    if (total_alignments == 0)
        return true;


    GpuTripletCandidate*
        device_candidates =
            nullptr;

    GpuLigandTripletOrders*
        device_orders =
            nullptr;

    std::uint32_t*
        device_offsets =
            nullptr;

    GpuAlignmentCandidate*
        device_alignments =
            nullptr;


    auto cleanup =
        [&]()
        {
            cudaFree(
                device_alignments
            );

            cudaFree(
                device_offsets
            );

            cudaFree(
                device_orders
            );

            cudaFree(
                device_candidates
            );
        };


    const std::size_t candidate_bytes =
        candidates.size() *
        sizeof(GpuTripletCandidate);


    const std::size_t order_bytes =
        ligand_orders.size() *
        sizeof(GpuLigandTripletOrders);


    const std::size_t offset_bytes =
        alignment_offsets.size() *
        sizeof(std::uint32_t);


    const std::size_t alignment_bytes =
        static_cast<std::size_t>(
            total_alignments
        ) *
        sizeof(GpuAlignmentCandidate);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidates
            ),
            candidate_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_orders
            ),
            order_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_offsets
            ),
            offset_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_alignments
            ),
            alignment_bytes
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (
        cudaMemcpy(
            device_candidates,
            candidates.data(),
            candidate_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess ||

        cudaMemcpy(
            device_orders,
            ligand_orders.data(),
            order_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess ||

        cudaMemcpy(
            device_offsets,
            alignment_offsets.data(),
            offset_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    const std::uint32_t candidate_count =
        static_cast<std::uint32_t>(
            candidates.size()
        );


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            candidate_count +
            block_size - 1
        ) /
        block_size;


    alignment_candidate_expand_kernel
        <<<block_count, block_size>>>(
            device_candidates,
            candidate_count,
            device_orders,
            device_offsets,
            device_alignments
        );


    if (
        cudaGetLastError() !=
            cudaSuccess ||
        cudaDeviceSynchronize() !=
            cudaSuccess
    ) {
        cleanup();

        return false;
    }


    alignments.alignments =
        device_alignments;

    alignments.count =
        total_alignments;


    /*
     * Ownership transferred to caller.
     */
    device_alignments =
        nullptr;


    cleanup();


    return true;
}


bool cuda_download_alignment_candidates(
    const GpuAlignmentCandidateArrayDevice& device_alignments,
    std::vector<GpuAlignmentCandidate>& host_alignments
)
{
    host_alignments.clear();


    if (device_alignments.count == 0)
        return true;


    if (
        device_alignments.alignments ==
        nullptr
    ) {
        return false;
    }


    host_alignments.resize(
        device_alignments.count
    );


    const std::size_t bytes =
        static_cast<std::size_t>(
            device_alignments.count
        ) *
        sizeof(GpuAlignmentCandidate);


    if (
        cudaMemcpy(
            host_alignments.data(),
            device_alignments.alignments,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_alignments.clear();

        return false;
    }


    return true;
}


void cuda_free_alignment_candidates(
    GpuAlignmentCandidateArrayDevice& alignments
)
{
    cudaFree(
        alignments.alignments
    );


    alignments.alignments =
        nullptr;

    alignments.count =
        0;
}


bool cuda_expand_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const std::vector<std::uint32_t>& alignment_offsets,
    std::vector<GpuAlignmentCandidate>& alignments
)
{
    alignments.clear();


    GpuAlignmentCandidateArrayDevice
        device_alignments{};


    if (
        !cuda_expand_alignment_candidates_device(
            candidates,
            ligand_orders,
            alignment_offsets,
            device_alignments
        )
    ) {
        return false;
    }


    const bool download_ok =
        cuda_download_alignment_candidates(
            device_alignments,
            alignments
        );


    cuda_free_alignment_candidates(
        device_alignments
    );


    return download_ok;
}