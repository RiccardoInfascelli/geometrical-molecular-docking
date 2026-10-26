#ifndef INTERACTION_AREAS_HPP
#define INTERACTION_AREAS_HPP

#include <GraphMol/RWMol.h>
#include <interaction/interaction_area.hpp>
#include "matching_areas.hpp"

#include <vector>

// Extract physical chemical features from a sanitized molecule with coordinates.
// Keep the interaction library's feature order; logical matching keys are built
// separately and refer back to these physical-area indices.
std::vector<interaction::interaction_area>
get_protein_areas(RDKit::RWMol& protein);

std::vector<interaction::interaction_area>
get_ligand_areas(RDKit::RWMol& ligand);

// Keep an area only if every feature atom matches a pocket atom within 1e-3 A.
// Matching uses coordinates, not atom indices; retained areas keep their order.
std::vector<interaction::interaction_area>
filter_protein_areas_in_pocket(
    const std::vector<interaction::interaction_area>& areas,
    const RDKit::ROMol& pocket
);

// Keep centroids within max_distance (A) of any pocket atom, preserving order.
// This is a proximity filter, distinct from full feature membership above.
std::vector<MatchingArea>
filter_matching_areas_near_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket,
    double max_distance
);

#endif
