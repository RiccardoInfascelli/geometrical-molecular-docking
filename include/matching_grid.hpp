#ifndef MATCHING_GRID_HPP
#define MATCHING_GRID_HPP

#include "matching_areas.hpp"

#include <cstddef>
#include <vector>

// Spatial bins of MatchingArea centroids, used by the spatial triplet counter.
// This is independent of the chemical/distance-key index used for docking lookup.
struct MatchingGrid
{
    double cell_size{};

    double xmin{};
    double ymin{};
    double zmin{};

    int nx{};
    int ny{};
    int nz{};

    // Cell c occupies area_indices[cell_offsets[c] ... cell_offsets[c + 1]).
    std::vector<std::size_t> cell_offsets;
    std::vector<std::size_t> area_indices;

    int index(int x, int y, int z) const;

    bool in_bounds(int x, int y, int z) const;

    int to_grid(
        double coordinate,
        double minimum
    ) const;
};

// Bins centroids in input order within each cell. Empty input returns an empty grid;
// nonempty input requires positive cell_size.
MatchingGrid build_matching_grid(
    const std::vector<MatchingArea>& areas,
    double cell_size
);

#endif
