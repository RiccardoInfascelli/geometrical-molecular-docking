#include "cuda_backend.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>


namespace
{

__global__ void fast_score_kernel(
    const GpuFastMatchingArea* ligand_areas,
    std::uint32_t ligand_area_count,
    GpuProteinGridDevice protein_grid,
    const GpuRigidTransform* transforms,
    std::uint32_t transform_count,
    GpuFastScore* scores
)
{
    const std::uint32_t pose_index =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;


    if (
        pose_index >=
        transform_count
    ) {
        return;
    }


    const auto transform =
        transforms[
            pose_index
        ];


    scores[pose_index] = docking_device::fast_score(
        ligand_areas, ligand_area_count, protein_grid, transform);
}

} // namespace


void cuda_free_fast_matching_areas(
    GpuFastMatchingAreaArrayDevice& device_areas
)
{
    if (
        device_areas.areas !=
        nullptr
    ) {
        cudaFree(
            device_areas.areas
        );
    }


    device_areas =
        GpuFastMatchingAreaArrayDevice{};
}


bool cuda_upload_fast_matching_areas(
    const std::vector<GpuFastMatchingArea>& host_areas,
    GpuFastMatchingAreaArrayDevice& device_areas
)
{
    if (
        host_areas.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    if (host_areas.empty()) {

        cuda_free_fast_matching_areas(
            device_areas
        );

        return true;
    }


    GpuFastMatchingAreaArrayDevice
        replacement;


    const std::size_t bytes =
        host_areas.size() *
        sizeof(GpuFastMatchingArea);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &replacement.areas
            ),
            bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMemcpy(
            replacement.areas,
            host_areas.data(),
            bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_fast_matching_areas(
            replacement
        );

        return false;
    }


    replacement.count =
        static_cast<std::uint32_t>(
            host_areas.size()
        );


    cuda_free_fast_matching_areas(
        device_areas
    );


    device_areas =
        replacement;


    return true;
}


bool cuda_score_fast_interactions_device(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuFastScoreArrayDevice& scores
)
{
    cuda_free_fast_scores(
        scores
    );


    if (transforms.count == 0)
        return true;


    if (
        transforms.transforms == nullptr
    ) {
        return false;
    }


    if (
        ligand_areas.count != 0 &&
        ligand_areas.areas == nullptr
    ) {
        return false;
    }


    if (
        protein_grid.voxels == nullptr ||
        protein_grid.voxel_count == 0
    ) {
        return false;
    }


    GpuFastScore*
        device_scores =
            nullptr;


    const std::size_t score_bytes =
        static_cast<std::size_t>(
            transforms.count
        ) *
        sizeof(GpuFastScore);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_scores
            ),
            score_bytes
        ) != cudaSuccess
    ) {
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


    fast_score_kernel
        <<<block_count, block_size>>>(
            ligand_areas.areas,
            ligand_areas.count,
            protein_grid,
            transforms.transforms,
            transforms.count,
            device_scores
        );


    if (
        cudaGetLastError() !=
            cudaSuccess ||
        cudaDeviceSynchronize() !=
            cudaSuccess
    ) {
        cudaFree(
            device_scores
        );

        return false;
    }


    scores.scores =
        device_scores;

    scores.count =
        transforms.count;


    return true;
}


bool cuda_download_fast_scores(
    const GpuFastScoreArrayDevice& device_scores,
    std::vector<GpuFastScore>& host_scores
)
{
    host_scores.clear();


    if (device_scores.count == 0)
        return true;


    if (
        device_scores.scores == nullptr
    ) {
        return false;
    }


    host_scores.resize(
        device_scores.count
    );


    const std::size_t score_bytes =
        static_cast<std::size_t>(
            device_scores.count
        ) *
        sizeof(GpuFastScore);


    if (
        cudaMemcpy(
            host_scores.data(),
            device_scores.scores,
            score_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_scores.clear();

        return false;
    }


    return true;
}


void cuda_free_fast_scores(
    GpuFastScoreArrayDevice& scores
)
{
    cudaFree(
        scores.scores
    );


    scores.scores =
        nullptr;

    scores.count =
        0;
}


bool cuda_score_fast_interactions(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuFastScore>& scores
)
{
    scores.clear();


    if (
        transforms.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    if (transforms.empty())
        return true;


    GpuRigidTransformArrayDevice
        device_transforms{};


    const std::size_t transform_bytes =
        transforms.size() *
        sizeof(GpuRigidTransform);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_transforms.transforms
            ),
            transform_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    device_transforms.count =
        static_cast<std::uint32_t>(
            transforms.size()
        );


    if (
        cudaMemcpy(
            device_transforms.transforms,
            transforms.data(),
            transform_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    GpuFastScoreArrayDevice
        device_scores{};


    if (
        !cuda_score_fast_interactions_device(
            ligand_areas,
            protein_grid,
            device_transforms,
            device_scores
        )
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    const bool download_ok =
        cuda_download_fast_scores(
            device_scores,
            scores
        );


    cuda_free_fast_scores(
        device_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    return download_ok;
}