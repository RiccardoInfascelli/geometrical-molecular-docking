#include "cuda_backend.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>


namespace
{

__global__ void grid_clash_kernel(
    const GpuPoint3* ligand_atoms,
    std::uint32_t ligand_atom_count,
    GpuProteinGridDevice protein_grid,
    const GpuRigidTransform* transforms,
    std::uint32_t transform_count,
    GpuClashScore* scores
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


    scores[pose_index] = docking_device::grid_clash(
        ligand_atoms, ligand_atom_count, protein_grid, transform);
}

} // namespace


bool cuda_score_grid_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores
)
{
    cuda_free_clash_scores(
        scores
    );


    if (transforms.count == 0)
        return true;


    if (transforms.transforms == nullptr)
        return false;


    if (
        ligand_atoms.count != 0 &&
        ligand_atoms.points == nullptr
    ) {
        return false;
    }


    if (
        protein_grid.voxels == nullptr ||
        protein_grid.voxel_count == 0
    ) {
        return false;
    }


    GpuClashScore* device_scores =
        nullptr;


    const std::size_t score_bytes =
        static_cast<std::size_t>(
            transforms.count
        ) *
        sizeof(GpuClashScore);


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


    grid_clash_kernel
        <<<block_count, block_size>>>(
            ligand_atoms.points,
            ligand_atoms.count,
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


    /*
     * Keep clash scores resident on the GPU.
     * Ownership is transferred to the caller.
     */
    scores.scores =
        device_scores;

    scores.count =
        transforms.count;


    return true;
}


bool cuda_download_clash_scores(
    const GpuClashScoreArrayDevice& device_scores,
    std::vector<GpuClashScore>& host_scores
)
{
    host_scores.clear();


    if (device_scores.count == 0)
        return true;


    if (device_scores.scores == nullptr)
        return false;


    host_scores.resize(
        device_scores.count
    );


    const std::size_t bytes =
        static_cast<std::size_t>(
            device_scores.count
        ) *
        sizeof(GpuClashScore);


    if (
        cudaMemcpy(
            host_scores.data(),
            device_scores.scores,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_scores.clear();

        return false;
    }


    return true;
}


void cuda_free_clash_scores(
    GpuClashScoreArrayDevice& device_scores
)
{
    if (device_scores.scores != nullptr) {

        cudaFree(
            device_scores.scores
        );
    }


    device_scores.scores =
        nullptr;

    device_scores.count =
        0;
}


bool cuda_score_grid_clash(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuClashScore>& scores
)
{
    scores.clear();


    if (transforms.empty())
        return true;


    if (
        transforms.size() >
        static_cast<std::size_t>(
            std::numeric_limits<
                std::uint32_t
            >::max()
        )
    ) {
        return false;
    }


    GpuRigidTransformArrayDevice
        device_transforms;


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


    GpuClashScoreArrayDevice
        device_scores;


    if (
        !cuda_score_grid_clash_device(
            ligand_atoms,
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
        cuda_download_clash_scores(
            device_scores,
            scores
        );


    cuda_free_clash_scores(
        device_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    return download_ok;
}