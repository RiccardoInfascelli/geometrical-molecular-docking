#ifndef CUDA_WEIGHTED_INTERACTION_SCORING_HPP
#define CUDA_WEIGHTED_INTERACTION_SCORING_HPP

#include "cuda_triplet_types.hpp"

#include <cstdint>
#include <vector>

namespace interaction { struct interaction_area; }
struct WeightedInteractionWeights;
struct WeightedInteractionScore;
struct GpuRigidTransformArrayDevice;

enum class GpuWeightedInteractionPattern : std::uint32_t {
    Hydrophobic, HydrogenDonor, HydrogenAcceptor, HalogenDonor,
    HalogenAcceptor, Anion, Cation, Aromatic5, Aromatic6, Metal, Chelated
};

struct GpuWeightedInteractionPoint { float x{}, y{}, z{}; };

// WeightedInteraction predicates need only the centroid and the first two orientation atoms.
// Coordinates stay float, matching chem-interactions rather than Fast scoring.
struct GpuWeightedInteractionArea {
    GpuWeightedInteractionPoint centroid{}, atoms[2]{};
    GpuWeightedInteractionPattern pattern{};
    std::uint32_t aromatic{};
    std::uint32_t num_atoms{};
};

struct GpuWeightedInteractionWeights {
    double hydrophobic{}, pi_stacking{}, hydrogen_bond{}, halogen_bond{};
    double cation_pi{}, ionic{}, metal{};
};

struct GpuWeightedInteractionScore {
    std::uint64_t total_interactions{};
    std::uint64_t satisfied_ligand_areas{}, satisfied_protein_areas{};
    std::uint64_t hydrophobic{}, hydrogen_bonds{}, halogen_bonds{};
    std::uint64_t ionic{}, cation_pi{}, pi_stacking{}, metal{};
    double weighted_score{};
    std::uint32_t valid{};
};

struct GpuWeightedInteractionAreasDevice {
    GpuWeightedInteractionArea* areas{};
    std::uint32_t count{};
};

struct GpuWeightedInteractionScoresDevice {
    GpuWeightedInteractionScore* scores{};
    std::uint32_t count{};
};

// Preserve physical interaction-area order, without expanding MatchingArea
// roles or selecting a protein region. Unsupported patterns throw.
std::vector<GpuWeightedInteractionArea> make_gpu_weighted_interaction_areas(
    const std::vector<interaction::interaction_area>& areas);
GpuWeightedInteractionWeights make_gpu_weighted_interaction_weights(const WeightedInteractionWeights& weights);
WeightedInteractionScore weighted_interaction_score_from_gpu(const GpuWeightedInteractionScore& score);

bool cuda_upload_weighted_interaction_areas(const std::vector<GpuWeightedInteractionArea>& host,
                             GpuWeightedInteractionAreasDevice& device);
void cuda_free_weighted_interaction_areas(GpuWeightedInteractionAreasDevice& device);
void cuda_free_weighted_interaction_scores(GpuWeightedInteractionScoresDevice& device);

// One thread scores one selected survivor. IDs index the compacted transforms;
// nullptr means identity selection. Output order follows the supplied IDs.
// Contacts are counted separately, but the weighted score adds only the maximum
// non-negative interaction weight for each physical ligand area, as on CPU.
// Protein-area coverage uses checked temporary storage, never a truncated pool.
// This blocking call finishes the kernel before releasing its scratch buffer.
// A completed call may contain invalid score rows; ranking rejects those rows.
bool cuda_score_weighted_interaction_device(
    const GpuWeightedInteractionAreasDevice& protein, const GpuWeightedInteractionAreasDevice& ligand,
    const GpuRigidTransformArrayDevice& transforms,
    const std::uint32_t* selected_pose_ids, std::uint32_t selected_count,
    const GpuWeightedInteractionWeights& weights, GpuWeightedInteractionScoresDevice& output);

#endif
