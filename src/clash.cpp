#include "clash.hpp"

#include <interaction/elements.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

double atom_vdw_radius(const RDKit::Atom& atom)
{
    const auto element =
        interaction::parse_chemical_element(
            atom.getSymbol()
        );

    if (element.has_value()) {
        return interaction::get(
            element.value()
        ).vdw_radius;
    }

    return 1.7;
}

}

int ClashGrid::index(
    int x,
    int y,
    int z) const
{
    return x + nx * (y + ny * z);
}

bool ClashGrid::in_bounds(
    int x,
    int y,
    int z) const
{
    return
        x >= 0 && x < nx &&
        y >= 0 && y < ny &&
        z >= 0 && z < nz;
}

int ClashGrid::to_grid(
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

ClashGrid build_clash_grid(
    const RDKit::ROMol& protein,
    double cell_size)
{
    const auto& conformer =
        protein.getConformer();

    double xmin =
        std::numeric_limits<double>::infinity();

    double ymin =
        std::numeric_limits<double>::infinity();

    double zmin =
        std::numeric_limits<double>::infinity();

    double xmax =
        -std::numeric_limits<double>::infinity();

    double ymax =
        -std::numeric_limits<double>::infinity();

    double zmax =
        -std::numeric_limits<double>::infinity();

    double max_protein_radius = 0.0;

    for (unsigned int i = 0;
         i < protein.getNumAtoms();
         ++i) {

        const auto position =
            conformer.getAtomPos(i);

        const double radius =
            atom_vdw_radius(
                *protein.getAtomWithIdx(i)
            );

        xmin = std::min(xmin, position.x);
        ymin = std::min(ymin, position.y);
        zmin = std::min(zmin, position.z);

        xmax = std::max(xmax, position.x);
        ymax = std::max(ymax, position.y);
        zmax = std::max(zmax, position.z);

        max_protein_radius =
            std::max(
                max_protein_radius,
                radius
            );
    }

    ClashGrid grid;

    grid.cell_size = cell_size;

    grid.xmin = xmin;
    grid.ymin = ymin;
    grid.zmin = zmin;

    grid.nx =
        static_cast<int>(
            std::floor(
                (xmax - xmin) /
                cell_size
            )
        ) + 1;

    grid.ny =
        static_cast<int>(
            std::floor(
                (ymax - ymin) /
                cell_size
            )
        ) + 1;

    grid.nz =
        static_cast<int>(
            std::floor(
                (zmax - zmin) /
                cell_size
            )
        ) + 1;

    grid.max_protein_radius =
        max_protein_radius;

    grid.cells.resize(
        static_cast<std::size_t>(
            grid.nx *
            grid.ny *
            grid.nz
        )
    );

    for (unsigned int i = 0;
         i < protein.getNumAtoms();
         ++i) {

        const auto position =
            conformer.getAtomPos(i);

        ClashAtom atom{
            position.x,
            position.y,
            position.z,
            atom_vdw_radius(
                *protein.getAtomWithIdx(i)
            )
        };

        const int x =
            grid.to_grid(
                position.x,
                grid.xmin
            );

        const int y =
            grid.to_grid(
                position.y,
                grid.ymin
            );

        const int z =
            grid.to_grid(
                position.z,
                grid.zmin
            );

        if (!grid.in_bounds(x, y, z))
            continue;

        grid.cells[
            grid.index(x, y, z)
        ].push_back(atom);
    }

    return grid;
}

double clash_score(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance)
{
    const auto& conformer =
        ligand.getConformer();

    double total_overlap = 0.0;

    for (unsigned int i = 0;
         i < ligand.getNumAtoms();
         ++i) {

        const auto ligand_position =
            conformer.getAtomPos(i);

        const double ligand_radius =
            atom_vdw_radius(
                *ligand.getAtomWithIdx(i)
            );

        const double maximum_interaction_distance =
            ligand_radius
            + protein_grid.max_protein_radius
            - tolerance;

        const int cell_radius =
            static_cast<int>(
                std::ceil(
                    maximum_interaction_distance /
                    protein_grid.cell_size
                )
            );

        const int gx =
            protein_grid.to_grid(
                ligand_position.x,
                protein_grid.xmin
            );

        const int gy =
            protein_grid.to_grid(
                ligand_position.y,
                protein_grid.ymin
            );

        const int gz =
            protein_grid.to_grid(
                ligand_position.z,
                protein_grid.zmin
            );

        for (int dx = -cell_radius;
             dx <= cell_radius;
             ++dx) {

            for (int dy = -cell_radius;
                 dy <= cell_radius;
                 ++dy) {

                for (int dz = -cell_radius;
                     dz <= cell_radius;
                     ++dz) {

                    const int x = gx + dx;
                    const int y = gy + dy;
                    const int z = gz + dz;

                    if (!protein_grid.in_bounds(
                            x,
                            y,
                            z))
                        continue;

                    const auto& cell =
                        protein_grid.cells[
                            protein_grid.index(
                                x,
                                y,
                                z
                            )
                        ];

                    for (const auto& protein_atom : cell) {

                        const double minimum_distance =
                            ligand_radius
                            + protein_atom.vdw_radius
                            - tolerance;

                        const double vx =
                            ligand_position.x
                            - protein_atom.x;

                        const double vy =
                            ligand_position.y
                            - protein_atom.y;

                        const double vz =
                            ligand_position.z
                            - protein_atom.z;

                        const double distance_squared =
                            vx * vx
                            + vy * vy
                            + vz * vz;

                        if (distance_squared >=
                            minimum_distance *
                            minimum_distance)
                            continue;

                        const double distance =
                            std::sqrt(
                                distance_squared
                            );

                        total_overlap +=
                            minimum_distance
                            - distance;
                    }
                }
            }
        }
    }

    return total_overlap;
}

double clash_score_per_ligand_atom(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance)
{
    if (ligand.getNumAtoms() == 0)
        return 0.0;

    return clash_score(
        ligand,
        protein_grid,
        tolerance
    ) /
    static_cast<double>(
        ligand.getNumAtoms()
    );
}

FastClashScore score_fast_grid_clash_cached(
    const std::vector<interaction::point3D>& ligand_points,
    const RigidTransform& transform,
    const ProteinGrid& grid)
{
    FastClashScore score;
    score.total_atoms = ligand_points.size();

    for (const auto& point : ligand_points) {
        const auto transformed = transform_point(point, transform);

        const int gx = grid.to_grid(transformed.x, grid.xmin);
        const int gy = grid.to_grid(transformed.y, grid.ymin);
        const int gz = grid.to_grid(transformed.z, grid.zmin);

        if (!grid.in_bounds(gx, gy, gz))
            continue;

        const auto& voxel = grid.voxels[grid.index(gx, gy, gz)];
        if (voxel.occupied)
            ++score.occupied_atoms;
    }

    if (score.total_atoms != 0) {
        score.fraction =
            static_cast<double>(score.occupied_atoms) /
            static_cast<double>(score.total_atoms);
    }
    return score;
}

FastClashScore score_fast_grid_clash(
    const RDKit::ROMol& ligand,
    const RigidTransform& transform,
    const ProteinGrid& grid)
{
    FastClashScore score;

    score.total_atoms = ligand.getNumAtoms();

    const auto& conformer = ligand.getConformer();

    for (unsigned int atom_index = 0;
         atom_index < ligand.getNumAtoms();
         ++atom_index) {

        const auto position = conformer.getAtomPos(atom_index);

        // Preserve the CPU/CUDA path: narrow input coordinates, transform in double,
        // then narrow again before voxel lookup. Boundary decisions depend on this.
        const interaction::point3D point{
            static_cast<interaction::coordinate_type>(position.x),
            static_cast<interaction::coordinate_type>(position.y),
            static_cast<interaction::coordinate_type>(position.z)
        };

        const auto transformed = transform_point(point, transform);

        const int gx = grid.to_grid(transformed.x, grid.xmin);
        const int gy = grid.to_grid(transformed.y, grid.ymin);
        const int gz = grid.to_grid(transformed.z, grid.zmin);

        if (!grid.in_bounds(gx, gy, gz))
            continue;

        const auto& voxel = grid.voxels[grid.index(gx, gy, gz)];

        if (voxel.occupied)
            ++score.occupied_atoms;
    }

    if (score.total_atoms != 0) {
        score.fraction =
            static_cast<double>(score.occupied_atoms) /
            static_cast<double>(score.total_atoms);
    }

    return score;
}
