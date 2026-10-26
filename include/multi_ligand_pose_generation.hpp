#pragma once

#include "pose_generation.hpp"
#include "triplet_generation.hpp"

// Historical declaration only: no implementation is provided in this repository.
// Current screening uses generate_candidate_poses_from_index(), which keeps the
// supplied index and does not reapply the ligand diameter during lookup.
PoseGenerationResult generate_candidate_poses_with_index(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    double ligand_diameter,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds,
    const ProteinTripletHashTable* shared_index,
    bool compute_oracle_rmsd
);
