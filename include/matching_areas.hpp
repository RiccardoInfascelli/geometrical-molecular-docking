#ifndef MATCHING_AREAS_HPP
#define MATCHING_AREAS_HPP

#include <interaction/interaction_area.hpp>

#include <cstddef>
#include <vector>
#include <cstdint>

enum class InteractionKey
{
    Hydrophobic,
    HydrogenDonor,
    HydrogenAcceptor,
    HalogenDonor,
    HalogenAcceptor,
    Cation,
    Anion,
    Aromatic,
    Metal,
    Chelated
};

using InteractionMask = std::uint16_t;

constexpr InteractionMask interaction_key_bit(
    InteractionKey key)
{
    return static_cast<InteractionMask>(
        InteractionMask{1} <<
        static_cast<unsigned int>(key)
    );
}

constexpr bool has_interaction(
    InteractionMask mask,
    InteractionKey key)
{
    return
        (mask & interaction_key_bit(key)) != 0;
}

constexpr void add_interaction(
    InteractionMask& mask,
    InteractionKey key)
{
    mask |= interaction_key_bit(key);
}

struct MatchingArea
{
    // Several logical areas can refer to the same physical chemical area.
    std::size_t source_area_index;
    interaction::point3D centroid;
    InteractionKey key;

    // Protein target-region radius in angstroms; ligand entries leave it at zero.
    double interaction_radius{0.0};
};

// Ligand keys describe the ligand's own interaction types.
std::vector<MatchingArea> build_ligand_matching_areas(
    const std::vector<interaction::interaction_area>& areas
);

// Protein keys describe compatible ligand types, with protein interaction radii.
// The order of these logical areas is retained by target and triplet generation.
std::vector<MatchingArea> build_protein_matching_areas(
    const std::vector<interaction::interaction_area>& areas
);

#endif
