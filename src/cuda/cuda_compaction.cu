#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>


namespace
{

__global__ void mark_clash_survivors_kernel(
    const GpuRigidTransform* transforms,
    const GpuClashScore* clash_scores,
    std::uint32_t count,
    double clash_threshold,
    std::uint32_t* survivor_flags,
    std::uint32_t* valid_transform_count
)
{
    const std::uint32_t index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (index >= count)
        return;


    const auto& transform =
        transforms[index];

    const auto& clash =
        clash_scores[index];


    const bool transform_valid =
        transform.valid != 0;


    if (transform_valid) {

        atomicAdd(
            valid_transform_count,
            1u
        );
    }


    const bool keep =
        transform_valid &&
        clash.valid != 0 &&
        clash.fraction <=
            clash_threshold;


    survivor_flags[index] =
        keep ? 1u : 0u;
}


// Exclusive offsets of the keep flags make this compaction stable: thread
// execution order cannot change pose order or the original alignment indices.
__global__ void scatter_clash_survivors_kernel(
    const GpuRigidTransform* input_transforms,
    const GpuClashScore* input_clash_scores,
    const std::uint32_t* survivor_flags,
    const std::uint32_t* survivor_offsets,
    std::uint32_t count,
    GpuRigidTransform* output_transforms,
    GpuClashScore* output_clash_scores,
    std::uint32_t* output_alignment_indices
)
{
    const std::uint32_t index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (index >= count)
        return;


    if (survivor_flags[index] == 0)
        return;


    const std::uint32_t output_index =
        survivor_offsets[index];


    output_transforms[
        output_index
    ] =
        input_transforms[index];


    output_clash_scores[
        output_index
    ] =
        input_clash_scores[index];


    output_alignment_indices[
        output_index
    ] =
        index;
}

} // namespace


bool cuda_compact_clash_survivors(
    const GpuRigidTransformArrayDevice& transforms,
    const GpuClashScoreArrayDevice& clash_scores,
    double clash_threshold,
    GpuClashSurvivorArrayDevice& survivors,
    std::uint32_t& valid_transform_count
)
{
    cuda_free_clash_survivors(
        survivors
    );


    valid_transform_count = 0;


    if (
        !std::isfinite(
            clash_threshold
        ) ||
        clash_threshold < 0.0 ||
        clash_threshold > 1.0
    ) {
        return false;
    }


    if (
        transforms.count !=
        clash_scores.count
    ) {
        return false;
    }


    if (transforms.count == 0)
        return true;


    if (
        transforms.transforms == nullptr ||
        clash_scores.scores == nullptr
    ) {
        return false;
    }


    if (
        transforms.count >
        static_cast<std::uint32_t>(
            INT_MAX
        )
    ) {
        return false;
    }


    std::uint32_t* device_flags =
        nullptr;

    std::uint32_t* device_offsets =
        nullptr;

    std::uint32_t* device_valid_count =
        nullptr;

    void* device_scan_storage =
        nullptr;


    GpuRigidTransform*
        device_survivor_transforms =
            nullptr;

    GpuClashScore*
        device_survivor_clash_scores =
            nullptr;

    std::uint32_t*
        device_survivor_indices =
            nullptr;


    auto cleanup =
        [&]()
        {
            cudaFree(
                device_survivor_indices
            );

            cudaFree(
                device_survivor_clash_scores
            );

            cudaFree(
                device_survivor_transforms
            );

            cudaFree(
                device_scan_storage
            );

            cudaFree(
                device_valid_count
            );

            cudaFree(
                device_offsets
            );

            cudaFree(
                device_flags
            );
        };


    const std::size_t flag_bytes =
        static_cast<std::size_t>(
            transforms.count
        ) *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_flags
            ),
            flag_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_offsets
            ),
            flag_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_valid_count
            ),
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (
        cudaMemset(
            device_valid_count,
            0,
            sizeof(std::uint32_t)
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            transforms.count +
            block_size - 1
        ) /
        block_size;


    mark_clash_survivors_kernel
        <<<block_count, block_size>>>(
            transforms.transforms,
            clash_scores.scores,
            transforms.count,
            clash_threshold,
            device_flags,
            device_valid_count
        );


    if (
        cudaGetLastError() !=
        cudaSuccess
    ) {
        cleanup();

        return false;
    }


    std::size_t scan_storage_bytes =
        0;


    if (
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_storage_bytes,
            device_flags,
            device_offsets,
            static_cast<int>(
                transforms.count
            )
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (scan_storage_bytes != 0) {

        if (
            cudaMalloc(
                &device_scan_storage,
                scan_storage_bytes
            ) != cudaSuccess
        ) {
            cleanup();

            return false;
        }
    }


    if (
        cub::DeviceScan::ExclusiveSum(
            device_scan_storage,
            scan_storage_bytes,
            device_flags,
            device_offsets,
            static_cast<int>(
                transforms.count
            )
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    std::uint32_t last_flag = 0;
    std::uint32_t last_offset = 0;


    if (
        cudaMemcpy(
            &valid_transform_count,
            device_valid_count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||

        cudaMemcpy(
            &last_flag,
            device_flags +
                transforms.count - 1,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||

        cudaMemcpy(
            &last_offset,
            device_offsets +
                transforms.count - 1,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        valid_transform_count = 0;

        cleanup();

        return false;
    }


    const std::uint32_t survivor_count =
        last_offset +
        last_flag;


    if (survivor_count == 0) {

        cleanup();

        return true;
    }


    const std::size_t transform_bytes =
        static_cast<std::size_t>(
            survivor_count
        ) *
        sizeof(GpuRigidTransform);


    const std::size_t clash_bytes =
        static_cast<std::size_t>(
            survivor_count
        ) *
        sizeof(GpuClashScore);


    const std::size_t index_bytes =
        static_cast<std::size_t>(
            survivor_count
        ) *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_survivor_transforms
            ),
            transform_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_survivor_clash_scores
            ),
            clash_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_survivor_indices
            ),
            index_bytes
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    scatter_clash_survivors_kernel
        <<<block_count, block_size>>>(
            transforms.transforms,
            clash_scores.scores,
            device_flags,
            device_offsets,
            transforms.count,
            device_survivor_transforms,
            device_survivor_clash_scores,
            device_survivor_indices
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


    survivors.transforms =
        device_survivor_transforms;

    survivors.clash_scores =
        device_survivor_clash_scores;

    survivors.alignment_indices =
        device_survivor_indices;

    survivors.count =
        survivor_count;


    /*
     * Ownership transferred to survivors.
     */
    device_survivor_transforms =
        nullptr;

    device_survivor_clash_scores =
        nullptr;

    device_survivor_indices =
        nullptr;


    cleanup();


    return true;
}


bool cuda_download_clash_survivors(
    const GpuClashSurvivorArrayDevice& device_survivors,
    std::vector<GpuRigidTransform>& host_transforms,
    std::vector<GpuClashScore>& host_clash_scores,
    std::vector<std::uint32_t>& host_alignment_indices
)
{
    host_transforms.clear();
    host_clash_scores.clear();
    host_alignment_indices.clear();


    if (device_survivors.count == 0)
        return true;


    if (
        device_survivors.transforms == nullptr ||
        device_survivors.clash_scores == nullptr ||
        device_survivors.alignment_indices == nullptr
    ) {
        return false;
    }


    host_transforms.resize(
        device_survivors.count
    );

    host_clash_scores.resize(
        device_survivors.count
    );

    host_alignment_indices.resize(
        device_survivors.count
    );


    const std::size_t transform_bytes =
        static_cast<std::size_t>(
            device_survivors.count
        ) *
        sizeof(GpuRigidTransform);


    const std::size_t clash_bytes =
        static_cast<std::size_t>(
            device_survivors.count
        ) *
        sizeof(GpuClashScore);


    const std::size_t index_bytes =
        static_cast<std::size_t>(
            device_survivors.count
        ) *
        sizeof(std::uint32_t);


    if (
        cudaMemcpy(
            host_transforms.data(),
            device_survivors.transforms,
            transform_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||

        cudaMemcpy(
            host_clash_scores.data(),
            device_survivors.clash_scores,
            clash_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess ||

        cudaMemcpy(
            host_alignment_indices.data(),
            device_survivors.alignment_indices,
            index_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_transforms.clear();
        host_clash_scores.clear();
        host_alignment_indices.clear();

        return false;
    }


    return true;
}


void cuda_free_clash_survivors(
    GpuClashSurvivorArrayDevice& survivors
)
{
    cudaFree(
        survivors.alignment_indices
    );

    cudaFree(
        survivors.clash_scores
    );

    cudaFree(
        survivors.transforms
    );


    survivors.alignment_indices =
        nullptr;

    survivors.clash_scores =
        nullptr;

    survivors.transforms =
        nullptr;

    survivors.count =
        0;
}