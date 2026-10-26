#ifndef GPU_DOCKING_OUTPUT_HPP
#define GPU_DOCKING_OUTPUT_HPP

#include "cuda_pose_pipeline.hpp"
#include "interaction_scoring.hpp"
#include "vinardo_scoring.hpp"

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace RDKit {
class ROMol;
class RWMol;
}

// Scores for one exported pose. Entries are paired with the ranked pose vector,
// while pose_id retains the original, zero-based clash-survivor sequence ID.
// Missing scores are exported as empty fields and omitted SDF properties.
struct GpuDockingPoseScores {
    std::size_t pose_id{};
    std::optional<WeightedInteractionScore> weighted_interaction;
    std::optional<VinardoLikeScore> vinardo;
    std::optional<double> vinardo_adjusted;
    std::optional<double> hybrid;
};

// Rank valid survivors by decreasing Fast score, then increasing alignment ID.
// Invalid transform/clash/Fast flags are excluded. Any nonfinite Fast score in
// a valid pose is an error, including poses outside the requested top-k.
// k must be positive. No clustering or RMSD participates in selection.
std::vector<GpuPipelinePose> select_top_gpu_poses(
    const std::vector<GpuPipelinePose>& poses,
    std::size_t k
);

// Check coordinate availability and the fixed atom mapping without calculating
// RMSD, including when no docking poses survive for a reference record.
void validate_gpu_docking_reference(
    const RDKit::ROMol& ligand,
    const RDKit::ROMol& reference
);

// Direct heavy-atom RMSD in the protein coordinate frame: no superposition,
// symmetry search, or atom remapping. The complete atom order, element,
// isotope, formal charge, aromaticity and bond graph must match, including H.
// The caller supplies atom correspondence; no graph isomorphism is inferred.
double gpu_pose_heavy_atom_rmsd(
    const RDKit::ROMol& ligand,
    const GpuRigidTransform& transform,
    const RDKit::ROMol& reference
);

void write_gpu_docking_pose_header(
    std::ostream& csv, bool with_rmsd = false, bool extended_scores = false);

// Export already ranked poses, including full transformed atom coordinates in
// SDF. Rotations are row-major and applied as R*x+t. Only these selected poses
// receive an RMSD evaluation. The with_rmsd argument must match the CSV header;
// when true without a reference, rows retain an empty RMSD field.
// A non-null scores vector enables the extended score columns and properties,
// and must have one entry per ranked pose in the same order. The header's
// extended_scores flag must match. Without scores the original output is kept.
// Pairing is positional: the caller must preserve score/pose identity.
void write_gpu_docking_poses(
    std::ostream& csv,
    std::ostream& sdf,
    const RDKit::RWMol& ligand,
    const std::vector<GpuPipelinePose>& ranked,
    std::size_t record_index,
    const std::string& molecule_id,
    bool with_rmsd = false,
    const RDKit::ROMol* reference = nullptr,
    const std::vector<GpuDockingPoseScores>* scores = nullptr
);

#endif
