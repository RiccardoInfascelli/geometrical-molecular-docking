#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <interaction/interaction_area.hpp>

#include "matching_areas.hpp"

#include <array>
#include <vector>

namespace RDKit { class ROMol; }

// Maximum distance between chemical-area centroids, not between ligand atoms.
// Returns zero when fewer than two areas are available.
double compute_ligand_diameter(
    const std::vector<interaction::interaction_area>& ligand_areas
);

struct RigidTransform
{
    // Row-major 3x3 rotation matrix.
    std::array<double, 9> rotation{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    std::array<double, 3> translation{
        0.0, 0.0, 0.0
    };
};

// Geometric centroid in angstroms, using atoms with atomic number > 1.
// Requires finite heavy-atom coordinates and at least one heavy atom.
std::array<double, 3> heavy_atom_centroid(const RDKit::ROMol& molecule);

// Distance in the receptor frame after R*c+t. Linearity makes this the
// centroid of the transformed atoms; no fitting, mapping or symmetry search.
double pose_center_distance(const std::array<double, 3>& ligand_center,
    const RigidTransform& transform, const std::array<double, 3>& reference_center);


// Applies R * point + t in double, then casts back to coordinate_type.
interaction::point3D transform_point(
    const interaction::point3D& point,
    const RigidTransform& transform
);


// Transforms centroids in a copy; keys, radii and source indices are unchanged.
std::vector<MatchingArea> transform_matching_areas(
    const std::vector<MatchingArea>& areas,
    const RigidTransform& transform
);

// Transforms both centroids and defining atoms, preserving interaction geometry.
std::vector<interaction::interaction_area>
transform_interaction_areas(
    const std::vector<interaction::interaction_area>& areas,
    const RigidTransform& transform
);

#endif
