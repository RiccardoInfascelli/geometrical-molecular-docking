#include "cuda_grid.hpp"

#include <limits>


GpuProteinGridHost make_gpu_protein_grid(
    const ProteinGrid& grid
)
{
    GpuProteinGridHost result;


    result.xmin =
        grid.xmin;

    result.ymin =
        grid.ymin;

    result.zmin =
        grid.zmin;

    result.voxel_size =
        grid.voxel_size;


    result.nx =
        static_cast<std::int32_t>(
            grid.nx
        );

    result.ny =
        static_cast<std::int32_t>(
            grid.ny
        );

    result.nz =
        static_cast<std::int32_t>(
            grid.nz
        );


    result.voxels.reserve(
        grid.voxels.size()
    );


    for (const auto& voxel :
         grid.voxels) {

        GpuGridVoxel gpu_voxel;


        gpu_voxel.interactions =
            static_cast<std::uint16_t>(
                voxel.interactions
            );


        gpu_voxel.occupied =
            voxel.occupied
            ? std::uint8_t{1}
            : std::uint8_t{0};


        result.voxels.push_back(
            gpu_voxel
        );
    }


    return result;
}


bool validate_gpu_protein_grid(
    const ProteinGrid& cpu_grid,
    const GpuProteinGridHost& gpu_grid
)
{
    if (
        gpu_grid.xmin != cpu_grid.xmin ||
        gpu_grid.ymin != cpu_grid.ymin ||
        gpu_grid.zmin != cpu_grid.zmin ||
        gpu_grid.voxel_size !=
            cpu_grid.voxel_size
    ) {
        return false;
    }


    if (
        gpu_grid.nx != cpu_grid.nx ||
        gpu_grid.ny != cpu_grid.ny ||
        gpu_grid.nz != cpu_grid.nz
    ) {
        return false;
    }


    if (
        gpu_grid.voxels.size() !=
        cpu_grid.voxels.size()
    ) {
        return false;
    }


    for (
        std::size_t i = 0;
        i < cpu_grid.voxels.size();
        ++i
    ) {
        const auto& cpu =
            cpu_grid.voxels[i];

        const auto& gpu =
            gpu_grid.voxels[i];


        if (
            gpu.interactions !=
            static_cast<std::uint16_t>(
                cpu.interactions
            )
        ) {
            return false;
        }


        const std::uint8_t expected_occupied =
            cpu.occupied
            ? std::uint8_t{1}
            : std::uint8_t{0};


        if (
            gpu.occupied !=
            expected_occupied
        ) {
            return false;
        }
    }


    return true;
}