#include "protein_grid.hpp"
#include <interaction/elements.hpp>

#include <algorithm>
#include <stdexcept>
#include <cmath>

GridBounds compute_pocket_bounds(
    const RDKit::ROMol& pocket)
{
    if (pocket.getNumAtoms() == 0) {
        throw std::runtime_error(
            "Cannot compute bounds of an empty pocket"
        );
    }

    const auto& conformer = pocket.getConformer();

    const auto first_position =
        conformer.getAtomPos(0);

    GridBounds bounds{
        first_position.x,
        first_position.x,

        first_position.y,
        first_position.y,

        first_position.z,
        first_position.z
    };

    for (unsigned int i = 1;
         i < pocket.getNumAtoms();
         ++i) {

        const auto position =
            conformer.getAtomPos(i);

        bounds.xmin = std::min(
            bounds.xmin,
            static_cast<double>(position.x)
        );

        bounds.xmax = std::max(
            bounds.xmax,
            static_cast<double>(position.x)
        );

        bounds.ymin = std::min(
            bounds.ymin,
            static_cast<double>(position.y)
        );

        bounds.ymax = std::max(
            bounds.ymax,
            static_cast<double>(position.y)
        );

        bounds.zmin = std::min(
            bounds.zmin,
            static_cast<double>(position.z)
        );

        bounds.zmax = std::max(
            bounds.zmax,
            static_cast<double>(position.z)
        );
    }

    return bounds;
}

std::size_t ProteinGrid::index(
    int x,
    int y,
    int z) const
{
    return static_cast<std::size_t>(
        x + nx * (y + ny * z)
    );
}


ProteinGrid build_empty_protein_grid(
    const RDKit::ROMol& pocket,
    double voxel_size,
    double margin)
{
    const auto bounds =
        compute_pocket_bounds(pocket);

    ProteinGrid grid;

    grid.voxel_size = voxel_size;

    grid.xmin = bounds.xmin - margin;
    grid.ymin = bounds.ymin - margin;
    grid.zmin = bounds.zmin - margin;

    const double xmax =
        bounds.xmax + margin;

    const double ymax =
        bounds.ymax + margin;

    const double zmax =
        bounds.zmax + margin;

    grid.nx = static_cast<int>(
        std::ceil(
            (xmax - grid.xmin) / voxel_size
        )
    );

    grid.ny = static_cast<int>(
        std::ceil(
            (ymax - grid.ymin) / voxel_size
        )
    );

    grid.nz = static_cast<int>(
        std::ceil(
            (zmax - grid.zmin) / voxel_size
        )
    );

    const std::size_t total_voxels =
        static_cast<std::size_t>(grid.nx) *
        static_cast<std::size_t>(grid.ny) *
        static_cast<std::size_t>(grid.nz);

    grid.voxels.resize(total_voxels);

    return grid;
}

bool ProteinGrid::in_bounds(
    int x,
    int y,
    int z) const
{
    return
        x >= 0 && x < nx &&
        y >= 0 && y < ny &&
        z >= 0 && z < nz;
}


int ProteinGrid::to_grid(
    double coordinate,
    double minimum) const
{
    return static_cast<int>(
        std::floor(
            (coordinate - minimum) / voxel_size
        )
    );
}

namespace
{

bool sphere_intersects_voxel(
    double cx,
    double cy,
    double cz,
    double radius,
    const ProteinGrid& grid,
    int x,
    int y,
    int z)
{
    const double x0 =
        grid.xmin + x * grid.voxel_size;
    const double y0 =
        grid.ymin + y * grid.voxel_size;
    const double z0 =
        grid.zmin + z * grid.voxel_size;

    const double x1 = x0 + grid.voxel_size;
    const double y1 = y0 + grid.voxel_size;
    const double z1 = z0 + grid.voxel_size;

    // Test the whole voxel box, not just its center; touching the sphere counts.
    const double closest_x =
        std::clamp(cx, x0, x1);

    const double closest_y =
        std::clamp(cy, y0, y1);

    const double closest_z =
        std::clamp(cz, z0, z1);

    const double dx = cx - closest_x;
    const double dy = cy - closest_y;
    const double dz = cz - closest_z;

    return
        dx * dx +
        dy * dy +
        dz * dz
        <= radius * radius;
}

}

void mark_protein_occupancy(
    const RDKit::ROMol& protein,
    ProteinGrid& grid)
{
    const auto& conformer =
        protein.getConformer();

    for (unsigned int i = 0;
         i < protein.getNumAtoms();
         ++i) {

        const auto position =
            conformer.getAtomPos(i);

        const auto symbol =
            protein.getAtomWithIdx(i)->getSymbol();

        const auto element =
            interaction::parse_chemical_element(
                symbol
            );

        double radius = 1.7;

        if (element.has_value()) {
            radius =
                interaction::get(
                    element.value()
                ).vdw_radius;
        }

        const int xmin = std::max(
            0,
            grid.to_grid(
                position.x - radius,
                grid.xmin
            )
        );

        const int xmax = std::min(
            grid.nx - 1,
            grid.to_grid(
                position.x + radius,
                grid.xmin
            )
        );

        const int ymin = std::max(
            0,
            grid.to_grid(
                position.y - radius,
                grid.ymin
            )
        );

        const int ymax = std::min(
            grid.ny - 1,
            grid.to_grid(
                position.y + radius,
                grid.ymin
            )
        );

        const int zmin = std::max(
            0,
            grid.to_grid(
                position.z - radius,
                grid.zmin
            )
        );

        const int zmax = std::min(
            grid.nz - 1,
            grid.to_grid(
                position.z + radius,
                grid.zmin
            )
        );

        for (int z = zmin; z <= zmax; ++z) {
            for (int y = ymin; y <= ymax; ++y) {
                for (int x = xmin; x <= xmax; ++x) {

                    if (sphere_intersects_voxel(
                            position.x,
                            position.y,
                            position.z,
                            radius,
                            grid,
                            x,
                            y,
                            z)) {

                        grid.voxels[
                            grid.index(x, y, z)
                        ].occupied = true;
                    }
                }
            }
        }
    }
}

void mark_interaction_regions(
    const std::vector<MatchingArea>& areas,
    ProteinGrid& grid)
{
    for (const auto& area : areas) {

        const double radius =
            area.interaction_radius;

        if (radius <= 0.0)
            continue;

        const double cx = area.centroid.x;
        const double cy = area.centroid.y;
        const double cz = area.centroid.z;

        const int xmin = std::max(
            0,
            grid.to_grid(
                cx - radius,
                grid.xmin
            )
        );

        const int xmax = std::min(
            grid.nx - 1,
            grid.to_grid(
                cx + radius,
                grid.xmin
            )
        );

        const int ymin = std::max(
            0,
            grid.to_grid(
                cy - radius,
                grid.ymin
            )
        );

        const int ymax = std::min(
            grid.ny - 1,
            grid.to_grid(
                cy + radius,
                grid.ymin
            )
        );

        const int zmin = std::max(
            0,
            grid.to_grid(
                cz - radius,
                grid.zmin
            )
        );

        const int zmax = std::min(
            grid.nz - 1,
            grid.to_grid(
                cz + radius,
                grid.zmin
            )
        );

        if (xmin > xmax ||
            ymin > ymax ||
            zmin > zmax) {
            continue;
        }

        for (int z = zmin; z <= zmax; ++z) {
            for (int y = ymin; y <= ymax; ++y) {
                for (int x = xmin; x <= xmax; ++x) {

                    auto& voxel =
                        grid.voxels[
                            grid.index(x, y, z)
                        ];

                    // Free means unoccupied here, not connected to the outside.
                    if (voxel.occupied)
                        continue;

                    if (!sphere_intersects_voxel(
                            cx,
                            cy,
                            cz,
                            radius,
                            grid,
                            x,
                            y,
                            z)) {
                        continue;
                    }

                    add_interaction(
                        voxel.interactions,
                        area.key
                    );
                }
            }
        }
    }
}


GridPoseScore score_ligand_on_grid(
    const std::vector<MatchingArea>& ligand_areas,
    const ProteinGrid& grid)
{
    GridPoseScore score;

    score.total_areas =
        ligand_areas.size();

    for (const auto& area : ligand_areas) {

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

        if (!grid.in_bounds(x, y, z))
            continue;

        ++score.areas_in_grid;

        const auto& voxel =
            grid.voxels[
                grid.index(x, y, z)
            ];

        if (voxel.occupied) {

            ++score.areas_on_occupied_voxel;
            continue;
        }

        if (has_interaction(
                voxel.interactions,
                area.key)) {

            ++score.compatible_areas;
        }
    }

    return score;
}



std::size_t count_occupied_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        if (voxel.occupied)
            ++count;
    }

    return count;
}


std::size_t count_interaction_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        if (voxel.interactions != 0)
            ++count;
    }

    return count;
}


std::size_t count_multi_interaction_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        InteractionMask mask =
            voxel.interactions;

        std::size_t bits = 0;

        while (mask != 0) {

            bits +=
                static_cast<std::size_t>(
                    mask & InteractionMask{1}
                );

            mask >>= 1;
        }

        if (bits > 1)
            ++count;
    }

    return count;
}
