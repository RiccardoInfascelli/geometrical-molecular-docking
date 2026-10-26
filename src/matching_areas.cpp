#include "matching_areas.hpp"

namespace
{

void add_area(
    std::vector<MatchingArea>& result,
    std::size_t source_index,
    const interaction::point3D& centroid,
    InteractionKey key,
    double interaction_radius = 0.0)
{
    result.push_back({
        source_index,
        centroid,
        key,
        interaction_radius
    });
}

} // namespace


std::vector<MatchingArea> build_ligand_matching_areas(
    const std::vector<interaction::interaction_area>& areas)
{
    using P = interaction::interaction_smarts_pattern;
    using C = interaction::interaction_class_pattern;

    std::vector<MatchingArea> result;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        const auto& area = areas[i];

        if (area.pattern_class == C::Aromatic) {
            add_area(result, i, area.centroid, InteractionKey::Aromatic);
            continue;
        }

        switch (area.pattern) {

            case P::Hydrophobic:
                add_area(result, i, area.centroid,
                         InteractionKey::Hydrophobic);
                break;

            case P::Hydrogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenDonor);
                break;

            case P::Hydrogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenAcceptor);
                break;

            case P::Halogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenDonor);
                break;

            case P::Halogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenAcceptor);
                break;

            case P::Cation:
                add_area(result, i, area.centroid,
                         InteractionKey::Cation);
                break;

            case P::Anion:
                add_area(result, i, area.centroid,
                         InteractionKey::Anion);
                break;

            case P::Metal:
                add_area(result, i, area.centroid,
                         InteractionKey::Metal);
                break;

            case P::Chelated:
                add_area(result, i, area.centroid,
                         InteractionKey::Chelated);
                break;

            default:
                break;
        }
    }

    return result;
}

std::vector<MatchingArea> build_protein_matching_areas(
    const std::vector<interaction::interaction_area>& areas)
{
    using P = interaction::interaction_smarts_pattern;
    using C = interaction::interaction_class_pattern;

    std::vector<MatchingArea> result;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        const auto& area = areas[i];

        // One physical aromatic area exposes both stacking and cation-pi keys.
        // Shared source indices prevent using both as vertices of one triplet.
        if (area.pattern_class == C::Aromatic) {

            add_area(
                result,
                i,
                area.centroid,
                InteractionKey::Aromatic,
                6.5
            );

            add_area(
                result,
                i,
                area.centroid,
                InteractionKey::Cation,
                4.5
            );

            continue;
        }

        switch (area.pattern) {

            case P::Hydrophobic:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Hydrophobic,
                    4.5
                );
                break;

            // Protein keys encode the complementary ligand type.
            case P::Hydrogen_donor:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::HydrogenAcceptor,
                    3.5
                );
                break;

            case P::Hydrogen_acceptor:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::HydrogenDonor,
                    3.5
                );
                break;

            case P::Halogen_donor:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::HalogenAcceptor,
                    3.5
                );
                break;

            case P::Halogen_acceptor:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::HalogenDonor,
                    3.5
                );
                break;

            case P::Anion:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Cation,
                    4.5
                );
                break;

            // Keep ionic and cation-pi as separate logical entries of this area.
            case P::Cation:

                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Anion,
                    4.5
                );

                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Aromatic,
                    4.5
                );

                break;

            case P::Metal:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Chelated,
                    3.5
                );
                break;

            case P::Chelated:
                add_area(
                    result,
                    i,
                    area.centroid,
                    InteractionKey::Metal,
                    3.5
                );
                break;

            default:
                break;
        }
    }

    return result;
}
