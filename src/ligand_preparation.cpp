#include "ligand_preparation.hpp"

#include "geometry.hpp"
#include "interaction_areas.hpp"

#include <cmath>
#include <stdexcept>

void prepare_ligand(RDKit::RWMol& molecule, PreparedLigand& prepared)
{
    prepared = PreparedLigand{};
    prepared.areas = get_ligand_areas(molecule);
    prepared.matching_areas = build_ligand_matching_areas(prepared.areas);
    prepared.diameter = compute_ligand_diameter(prepared.areas);

    if (prepared.areas.size() < 3 || !std::isfinite(prepared.diameter) ||
        prepared.diameter <= 0.0) {
        throw std::runtime_error("Insufficient interaction areas / invalid diameter");
    }

    // Preserve the ligand diameter as the triplet distance limit.
    prepared.triplets = generate_triplets(prepared.matching_areas, prepared.diameter);
}
