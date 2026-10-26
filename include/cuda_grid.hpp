#ifndef CUDA_GRID_HPP
#define CUDA_GRID_HPP

#include "protein_grid.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuGridVoxel
{
    std::uint16_t interactions{};
    std::uint8_t occupied{};
    std::uint8_t padding{};
};


struct GpuProteinGridHost
{
    double xmin{};
    double ymin{};
    double zmin{};

    double voxel_size{};

    std::int32_t nx{};
    std::int32_t ny{};
    std::int32_t nz{};

    std::vector<GpuGridVoxel> voxels;
};

struct GpuProteinGridDevice
{
    double xmin{};
    double ymin{};
    double zmin{};

    double voxel_size{};

    std::int32_t nx{};
    std::int32_t ny{};
    std::int32_t nz{};

    GpuGridVoxel* voxels{nullptr};

    std::uint32_t voxel_count{};
};


struct GpuClashScore
{
    std::uint32_t occupied_atoms{};
    std::uint32_t total_atoms{};

    double fraction{};

    std::uint32_t valid{};
    std::uint32_t padding{};
};


static_assert(
    std::is_trivially_copyable_v<
        GpuProteinGridDevice
    >
);

static_assert(
    std::is_trivially_copyable_v<
        GpuClashScore
    >
);


static_assert(
    std::is_trivially_copyable_v<
        GpuGridVoxel
    >
);

static_assert(
    sizeof(GpuGridVoxel) == 4
);


// Pack occupancy and interaction masks without changing voxel order or extent.
// This is a host conversion; dimensions must fit the packed integer types.
GpuProteinGridHost make_gpu_protein_grid(
    const ProteinGrid& grid
);


bool validate_gpu_protein_grid(
    const ProteinGrid& cpu_grid,
    const GpuProteinGridHost& gpu_grid
);


#endif
