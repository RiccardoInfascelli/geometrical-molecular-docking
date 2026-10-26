#include "matching_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>


int MatchingGrid::index(
    int x,
    int y,
    int z) const
{
    return x + nx * (y + ny * z);
}

bool MatchingGrid::in_bounds(
    int x,
    int y,
    int z) const
{
    return
        x >= 0 && x < nx &&
        y >= 0 && y < ny &&
        z >= 0 && z < nz;
}

int MatchingGrid::to_grid(
    double coordinate,
    double minimum) const
{
    return static_cast<int>(
        std::floor(
            (coordinate - minimum) /
            cell_size
        )
    );
}


MatchingGrid build_matching_grid(
    const std::vector<MatchingArea>& areas,
    double cell_size)
{
    if (areas.empty())
        return {};

    if (cell_size <= 0.0)
        throw std::invalid_argument(
            "Matching grid cell size must be positive."
        );

    double xmin = areas[0].centroid.x;
    double ymin = areas[0].centroid.y;
    double zmin = areas[0].centroid.z;

    double xmax = xmin;
    double ymax = ymin;
    double zmax = zmin;

    for (const auto& area : areas) {

        const double x =
            static_cast<double>(area.centroid.x);

        const double y =
            static_cast<double>(area.centroid.y);

        const double z =
            static_cast<double>(area.centroid.z);

        xmin = std::min(xmin, x);
        ymin = std::min(ymin, y);
        zmin = std::min(zmin, z);

        xmax = std::max(xmax, x);
        ymax = std::max(ymax, y);
        zmax = std::max(zmax, z);
    }

    MatchingGrid grid;

    grid.cell_size = cell_size;

    grid.xmin = xmin;
    grid.ymin = ymin;
    grid.zmin = zmin;

    grid.nx =
        static_cast<int>(
            std::floor((xmax - xmin) / cell_size)
        ) + 1;

    grid.ny =
        static_cast<int>(
            std::floor((ymax - ymin) / cell_size)
        ) + 1;

    grid.nz =
        static_cast<int>(
            std::floor((zmax - zmin) / cell_size)
        ) + 1;

    const std::size_t num_cells =
        static_cast<std::size_t>(grid.nx) *
        static_cast<std::size_t>(grid.ny) *
        static_cast<std::size_t>(grid.nz);

    std::vector<std::size_t> counts(
        num_cells,
        0
    );

    // Count each cell before allocating one contiguous index array.
    for (const auto& area : areas) {

        const int x =
            grid.to_grid(
                area.centroid.x,
                grid.xmin
            );

        const int y =
            grid.to_grid(
                area.centroid.y,
                grid.ymin
            );

        const int z =
            grid.to_grid(
                area.centroid.z,
                grid.zmin
            );

        const std::size_t cell =
            static_cast<std::size_t>(
                grid.index(x, y, z)
            );

        ++counts[cell];
    }

    // Exclusive offsets delimit the entries belonging to each cell.
    grid.cell_offsets.resize(
        num_cells + 1,
        0
    );

    for (std::size_t i = 0;
         i < num_cells;
         ++i) {

        grid.cell_offsets[i + 1] =
            grid.cell_offsets[i] +
            counts[i];
    }

    grid.area_indices.resize(
        areas.size()
    );

    std::vector<std::size_t> cursor(
        grid.cell_offsets.begin(),
        grid.cell_offsets.end() - 1
    );

    // Scattering in input order preserves area order within each cell.
    for (std::size_t i = 0;
         i < areas.size();
         ++i) {

        const auto& area = areas[i];

        const int x =
            grid.to_grid(
                area.centroid.x,
                grid.xmin
            );

        const int y =
            grid.to_grid(
                area.centroid.y,
                grid.ymin
            );

        const int z =
            grid.to_grid(
                area.centroid.z,
                grid.zmin
            );

        const std::size_t cell =
            static_cast<std::size_t>(
                grid.index(x, y, z)
            );

        grid.area_indices[
            cursor[cell]++
        ] = i;
    }

    return grid;
}
