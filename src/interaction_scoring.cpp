#include "interaction_scoring.hpp"

#include <interaction/find_interaction.hpp>
#include <interaction/interactions.hpp>

#include <algorithm>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace
{

bool is_aromatic(
    const interaction::interaction_area& area)
{
    return
        area.pattern_class ==
        interaction::interaction_class_pattern::Aromatic;
}

bool is_cation(
    const interaction::interaction_area& area)
{
    return
        area.pattern ==
        interaction::interaction_smarts_pattern::Cation;
}

bool is_cation_pi_pair(
    const interaction::interaction_area& protein_area,
    const interaction::interaction_area& ligand_area)
{
    return
        (is_aromatic(protein_area) && is_cation(ligand_area))
        ||
        (is_cation(protein_area) && is_aromatic(ligand_area));
}

double weighted_interaction_weight(
    std::string_view type_name,
    const interaction::interaction_area& protein_area,
    const interaction::interaction_area& ligand_area,
    const WeightedInteractionWeights& weights)
{
    if (type_name == "Hydrophobic")
        return weights.hydrophobic;

    if (type_name == "Hydrogen Bond")
        return weights.hydrogen_bond;

    if (type_name == "Halogen Bond")
        return weights.halogen_bond;

    if (type_name == "Pi Stacking")
        return weights.pi_stacking;

    if (type_name == "Metal")
        return weights.metal;

    if (type_name == "Ionic") {

        // The interaction library also labels cation-pi contacts as Ionic.
        if (is_cation_pi_pair(protein_area, ligand_area))
            return weights.cation_pi;

        return weights.ionic;
    }

    return 0.0;
}

} // namespace


double fast_interaction_weight(
    InteractionKey key,
    const FastScoreWeights& weights)
{
    switch (key) {

        case InteractionKey::Hydrophobic:
            return weights.hydrophobic;

        case InteractionKey::HydrogenDonor:
        case InteractionKey::HydrogenAcceptor:
            return weights.hydrogen_bond;

        case InteractionKey::HalogenDonor:
        case InteractionKey::HalogenAcceptor:
            return weights.halogen_bond;

        case InteractionKey::Cation:
            return weights.cation;

        case InteractionKey::Anion:
            return weights.anion;

        case InteractionKey::Aromatic:
            return weights.aromatic;

        case InteractionKey::Metal:
            return weights.metal;

        case InteractionKey::Chelated:
            return weights.chelated;
    }

    return 0.0;
}


FastInteractionScore score_matching_areas_weighted(
    const std::vector<MatchingArea>& ligand_areas,
    const ProteinGrid& grid,
    const FastScoreWeights& weights)
{
    FastInteractionScore score;

    score.total_areas =
        ligand_areas.size();

    for (const auto& area : ligand_areas) {

        const int gx =
            grid.to_grid(
                area.centroid.x,
                grid.xmin
            );

        const int gy =
            grid.to_grid(
                area.centroid.y,
                grid.ymin
            );

        const int gz =
            grid.to_grid(
                area.centroid.z,
                grid.zmin
            );

        if (!grid.in_bounds(gx, gy, gz))
            continue;

        ++score.areas_in_grid;

        const auto& voxel =
            grid.voxels[
                grid.index(gx, gy, gz)
            ];

        if (voxel.occupied) {
            ++score.areas_on_occupied_voxel;
            continue;
        }

        if (
            (voxel.interactions &
             interaction_key_bit(area.key)) == 0
        ) {
            continue;
        }

        ++score.compatible_areas;

        score.weighted_score +=
            fast_interaction_weight(
                area.key,
                weights
            );
    }

    return score;
}


WeightedInteractionScore score_weighted_interactions(
    const std::vector<interaction::interaction_area>& protein_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const WeightedInteractionWeights& weights)
{
    const auto interactions =
        interaction::find_interactions(
            protein_areas,
            ligand_areas
        );

    WeightedInteractionScore score;

    score.total_interactions =
        interactions.size();

    std::unordered_set<std::size_t>
        satisfied_ligand_areas;

    std::unordered_set<std::size_t>
        satisfied_protein_areas;

    // Contact rewards compete within each physical ligand area. Changing a
    // weight can expose another winner, so weighted counts cannot replace this max.
    std::vector<double>
        best_weight_by_ligand_area(
            ligand_areas.size(),
            0.0
        );

    for (const auto& formed : interactions) {

        const std::size_t protein_index =
            formed.index_protein_interaction_area;

        const std::size_t ligand_index =
            formed.index_ligand_interaction_area;

        if (
            protein_index >= protein_areas.size()
            ||
            ligand_index >= ligand_areas.size()
        ) {
            continue;
        }

        satisfied_protein_areas.insert(
            protein_index
        );

        satisfied_ligand_areas.insert(
            ligand_index
        );

        const std::string_view type_name =
            interaction::get(
                formed.type
            ).name;

        if (type_name == "Hydrophobic") {
            ++score.hydrophobic;
        }
        else if (type_name == "Hydrogen Bond") {
            ++score.hydrogen_bonds;
        }
        else if (type_name == "Halogen Bond") {
            ++score.halogen_bonds;
        }
        else if (type_name == "Ionic") {

            if (
                is_cation_pi_pair(
                    protein_areas[protein_index],
                    ligand_areas[ligand_index]
                )
            ) {
                ++score.cation_pi;
            }
            else {
                ++score.ionic;
            }
        }
        else if (type_name == "Pi Stacking") {
            ++score.pi_stacking;
        }
        else if (type_name == "Metal") {
            ++score.metal;
        }

        const double contribution =
            weighted_interaction_weight(
                type_name,
                protein_areas[protein_index],
                ligand_areas[ligand_index],
                weights
            );

        best_weight_by_ligand_area[
            ligand_index
        ] =
            std::max(
                best_weight_by_ligand_area[
                    ligand_index
                ],
                contribution
            );
    }

    score.satisfied_ligand_areas =
        satisfied_ligand_areas.size();

    score.satisfied_protein_areas =
        satisfied_protein_areas.size();

    for (const double weight :
         best_weight_by_ligand_area) {

        score.weighted_score +=
            weight;
    }

    return score;
}
