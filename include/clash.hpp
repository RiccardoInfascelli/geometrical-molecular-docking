#ifndef CLASH_HPP
#define CLASH_HPP

#include "geometry.hpp"
#include "protein_grid.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <vector>

struct ClashAtom
{
    double x{};
    double y{};
    double z{};
    double vdw_radius{};
};

struct ClashGrid
{
    double cell_size{};

    double xmin{};
    double ymin{};
    double zmin{};

    int nx{};
    int ny{};
    int nz{};

    double max_protein_radius{};

    std::vector<std::vector<ClashAtom>> cells;

    int index(int x, int y, int z) const;
    bool in_bounds(int x, int y, int z) const;
    int to_grid(double coordinate, double minimum) const;
};

// Spatial bins of protein atom centers for VDW overlap queries. Requires a
// nonempty protein with coordinates and a positive cell_size.
ClashGrid build_clash_grid(
    const RDKit::ROMol& protein,
    double cell_size
);

// Sums pairwise penetration depths in angstroms at the ligand's current coordinates.
// Pair contact distance is r_ligand + r_protein - tolerance; lower is better.
// Assumes tolerance leaves the contact distances nonnegative.
double clash_score(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance
);

// Divides overlap by all ligand atoms; an empty ligand scores zero.
double clash_score_per_ligand_atom(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance
);

// Voxel occupancy of transformed atom centers, distinct from VDW overlap above.
// Outside-grid atoms stay in total_atoms but do not count as occupied.
struct FastClashScore
{
    std::size_t total_atoms{};
    std::size_t occupied_atoms{};
    double fraction{};
};

// Samples every transformed atom center, including hydrogens present in the input.
// Requires a conformer even for an empty ligand, whose fraction is zero.
FastClashScore score_fast_grid_clash(
    const RDKit::ROMol& ligand,
    const RigidTransform& transform,
    const ProteinGrid& grid
);

// Same occupancy test using atom coordinates narrowed to coordinate_type once.
// Rebuild the snapshot when the conformer, atom order or atom count changes.
// Outside-grid atoms remain in the denominator; an empty snapshot scores zero.
FastClashScore score_fast_grid_clash_cached(
    const std::vector<interaction::point3D>& ligand_points,
    const RigidTransform& transform,
    const ProteinGrid& grid
);

#endif
