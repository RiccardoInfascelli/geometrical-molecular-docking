#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>


void cuda_free_protein_grid(
    GpuProteinGridDevice& device_grid
)
{
    if (device_grid.voxels != nullptr) {

        cudaFree(
            device_grid.voxels
        );
    }


    device_grid =
        GpuProteinGridDevice{};
}


bool cuda_upload_protein_grid(
    const GpuProteinGridHost& host_grid,
    GpuProteinGridDevice& device_grid
)
{
    if (
        host_grid.nx <= 0 ||
        host_grid.ny <= 0 ||
        host_grid.nz <= 0 ||
        host_grid.voxel_size <= 0.0
    ) {
        return false;
    }


    const std::size_t expected_voxels =
        static_cast<std::size_t>(
            host_grid.nx
        ) *
        static_cast<std::size_t>(
            host_grid.ny
        ) *
        static_cast<std::size_t>(
            host_grid.nz
        );


    if (
        host_grid.voxels.size() !=
        expected_voxels
    ) {
        return false;
    }


    if (
        host_grid.voxels.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    GpuProteinGridDevice replacement;


    replacement.xmin =
        host_grid.xmin;

    replacement.ymin =
        host_grid.ymin;

    replacement.zmin =
        host_grid.zmin;

    replacement.voxel_size =
        host_grid.voxel_size;

    replacement.nx =
        host_grid.nx;

    replacement.ny =
        host_grid.ny;

    replacement.nz =
        host_grid.nz;

    replacement.voxel_count =
        static_cast<std::uint32_t>(
            host_grid.voxels.size()
        );


    const std::size_t bytes =
        host_grid.voxels.size() *
        sizeof(GpuGridVoxel);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &replacement.voxels
            ),
            bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMemcpy(
            replacement.voxels,
            host_grid.voxels.data(),
            bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_protein_grid(
            replacement
        );

        return false;
    }


    cuda_free_protein_grid(
        device_grid
    );


    device_grid =
        replacement;


    return true;
}