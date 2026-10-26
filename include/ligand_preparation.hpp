#ifndef LIGAND_PREPARATION_HPP
#define LIGAND_PREPARATION_HPP

#include "matching_areas.hpp"
#include "triplet_generation.hpp"

#include <GraphMol/RWMol.h>
#include <interaction/interaction_area.hpp>

#include <vector>

// Feature-coordinate snapshot for one conformer. Rebuild it after changing
// coordinates or chemistry; source indices identify this molecule's areas.
struct PreparedLigand
{
    std::vector<interaction::interaction_area> areas;
    std::vector<MatchingArea> matching_areas;
    // Maximum distance between chemical-area centroids, in angstroms.
    double diameter = 0.0;
    std::vector<Triplet> triplets;
};

// Prepares an already loaded and sanitized molecule without taking ownership.
// Resets the output and fills it in order; completed stages remain available
// on failure so callers can report partial preparation counts.
// Rejects fewer than three physical areas or a nonpositive/nonfinite diameter;
// valid areas can still produce no triplets after geometric exclusions.
void prepare_ligand(RDKit::RWMol& molecule, PreparedLigand& prepared);

#endif // LIGAND_PREPARATION_HPP
