#include "cuda_weighted_interaction_scoring.hpp"
#include "interaction_scoring.hpp"

#include <interaction/interaction_area.hpp>

#include <limits>
#include <stdexcept>

std::vector<GpuWeightedInteractionArea> make_gpu_weighted_interaction_areas(
    const std::vector<interaction::interaction_area>& areas)
{
    if (areas.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Too many WeightedInteraction interaction areas for CUDA");
    std::vector<GpuWeightedInteractionArea> packed;
    packed.reserve(areas.size());
    for (const auto& area : areas) {
        GpuWeightedInteractionArea result;
        using Pattern = interaction::interaction_smarts_pattern;
        switch (area.pattern) {
        case Pattern::Hydrophobic: result.pattern = GpuWeightedInteractionPattern::Hydrophobic; break;
        case Pattern::Hydrogen_donor: result.pattern = GpuWeightedInteractionPattern::HydrogenDonor; break;
        case Pattern::Hydrogen_acceptor: result.pattern = GpuWeightedInteractionPattern::HydrogenAcceptor; break;
        case Pattern::Halogen_donor: result.pattern = GpuWeightedInteractionPattern::HalogenDonor; break;
        case Pattern::Halogen_acceptor: result.pattern = GpuWeightedInteractionPattern::HalogenAcceptor; break;
        case Pattern::Anion: result.pattern = GpuWeightedInteractionPattern::Anion; break;
        case Pattern::Cation: result.pattern = GpuWeightedInteractionPattern::Cation; break;
        case Pattern::Aromatic_ring_5: result.pattern = GpuWeightedInteractionPattern::Aromatic5; break;
        case Pattern::Aromatic_ring_6: result.pattern = GpuWeightedInteractionPattern::Aromatic6; break;
        case Pattern::Metal: result.pattern = GpuWeightedInteractionPattern::Metal; break;
        case Pattern::Chelated: result.pattern = GpuWeightedInteractionPattern::Chelated; break;
        default: throw std::invalid_argument("Unsupported WeightedInteraction interaction pattern");
        }
        if (area.num_atoms > area.max_num_atoms)
            throw std::invalid_argument("Invalid WeightedInteraction interaction area atom count");
        result.centroid = {area.centroid.x, area.centroid.y, area.centroid.z};
        result.aromatic = area.pattern_class == interaction::interaction_class_pattern::Aromatic;
        result.num_atoms = static_cast<std::uint32_t>(area.num_atoms);
        for (std::size_t i = 0; i < 2 && i < area.num_atoms; ++i)
            result.atoms[i] = {area.x[i], area.y[i], area.z[i]};
        packed.push_back(result);
    }
    return packed;
}

GpuWeightedInteractionWeights make_gpu_weighted_interaction_weights(const WeightedInteractionWeights& weights)
{
    return {weights.hydrophobic, weights.pi_stacking, weights.hydrogen_bond,
            weights.halogen_bond, weights.cation_pi, weights.ionic, weights.metal};
}

WeightedInteractionScore weighted_interaction_score_from_gpu(const GpuWeightedInteractionScore& score)
{
    if (!score.valid) throw std::runtime_error("Invalid CUDA WeightedInteraction score");
    WeightedInteractionScore result;
    result.total_interactions = score.total_interactions;
    result.satisfied_ligand_areas = score.satisfied_ligand_areas;
    result.satisfied_protein_areas = score.satisfied_protein_areas;
    result.hydrophobic = score.hydrophobic;
    result.hydrogen_bonds = score.hydrogen_bonds;
    result.halogen_bonds = score.halogen_bonds;
    result.ionic = score.ionic;
    result.cation_pi = score.cation_pi;
    result.pi_stacking = score.pi_stacking;
    result.metal = score.metal;
    result.weighted_score = score.weighted_score;
    return result;
}
