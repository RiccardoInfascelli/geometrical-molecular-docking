#ifndef POSE_TARGETS_HPP
#define POSE_TARGETS_HPP

#include "matching_areas.hpp"
#include "protein_grid.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <optional>
#include <vector>

struct PoseTarget
{
    std::size_t source_area_index{};
    interaction::point3D position{};
    InteractionKey key{};
};

// Select at most one free voxel center per MatchingArea, in area order.
// The area interaction radius and the pocket-region radius are distinct limits;
// source_area_index is retained even when multiple logical areas share it.
// Prefer clearance to the protein VDW surface; omit areas with no valid voxel.
std::vector<PoseTarget> generate_pose_targets(
    const std::vector<MatchingArea>& protein_areas,
    const ProteinGrid& grid,
    const RDKit::ROMol& protein,
    const RDKit::ROMol& pocket,
    double pocket_region_radius = 6.5
);

// Stable post-selection filter using min(distance to atom center - VDW radius).
// A missing threshold leaves targets untouched and does not access the protein;
// zero is an active threshold. Returns the number removed, preserving source IDs.
// Apply before indexing: surviving source IDs remain, but vector positions change.
std::size_t prune_pose_targets(
    std::vector<PoseTarget>& targets,
    const RDKit::ROMol& protein,
    std::optional<double> minimum_clearance = std::nullopt
);


// Alternative pocket filter: every defining atom must match a pocket atom within
// 0.001 A. This is distinct from the centroid-distance filter used in preparation.
std::vector<MatchingArea>
filter_matching_areas_in_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const RDKit::ROMol& pocket
);

#endif
