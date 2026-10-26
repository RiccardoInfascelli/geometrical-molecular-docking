// src/interaction_areas.cpp
#include "interaction_areas.hpp"

#include <interaction/find_interaction_area.hpp>

std::vector<interaction::interaction_area>
get_protein_areas(RDKit::RWMol& protein)
{
    return interaction::find_protein_areas(protein);
}

std::vector<interaction::interaction_area>
get_ligand_areas(RDKit::RWMol& ligand)
{
    return interaction::find_ligand_areas(ligand);
}

std::vector<interaction::interaction_area>
filter_protein_areas_in_pocket(
    const std::vector<interaction::interaction_area>& areas,
    const RDKit::ROMol& pocket)
{
    std::vector<interaction::interaction_area> result;

    const auto& conformer =
        pocket.getConformer();

    constexpr double tolerance = 1e-3;
    constexpr double tolerance2 =
        tolerance * tolerance;

    for (const auto& area : areas) {

        bool all_atoms_in_pocket = true;

        for (std::size_t i = 0;
             i < area.num_atoms;
             ++i) {

            bool found = false;

            for (unsigned int j = 0;
                 j < pocket.getNumAtoms();
                 ++j) {

                const auto p =
                    conformer.getAtomPos(j);

                const double dx =
                    area.x[i] - p.x;

                const double dy =
                    area.y[i] - p.y;

                const double dz =
                    area.z[i] - p.z;

                if (
                    dx * dx +
                    dy * dy +
                    dz * dz <= tolerance2
                ) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                all_atoms_in_pocket = false;
                break;
            }
        }

        if (all_atoms_in_pocket)
            result.push_back(area);
    }

    return result;
}

std::vector<MatchingArea>
filter_matching_areas_near_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket,
    double max_distance)
{
    std::vector<MatchingArea> result;

    const auto& conformer =
        pocket.getConformer();

    const double max_distance2 =
        max_distance * max_distance;

    for (const auto& area :
         matching_areas) {

        bool keep = false;

        for (unsigned int i = 0;
             i < pocket.getNumAtoms();
             ++i) {

            const auto p =
                conformer.getAtomPos(i);

            const double dx =
                static_cast<double>(area.centroid.x) - p.x;

            const double dy =
                static_cast<double>(area.centroid.y) - p.y;

            const double dz =
                static_cast<double>(area.centroid.z) - p.z;

            const double distance2 =
                dx * dx +
                dy * dy +
                dz * dz;

            if (distance2 <= max_distance2) {
                keep = true;
                break;
            }
        }

        if (keep)
            result.push_back(area);
    }

    return result;
}