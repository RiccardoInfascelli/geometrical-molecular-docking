#ifndef CUDA_GEOMETRY_HPP
#define CUDA_GEOMETRY_HPP

#include <GraphMol/ROMol.h>

#include "cuda_triplet_types.hpp"
#include "matching_areas.hpp"
#include "pose_targets.hpp"

#include <vector>


// Pack host coordinates without reordering: triplet indices must still address
// the corresponding MatchingArea or PoseTarget. These functions do not upload.
std::vector<GpuPoint3>
make_gpu_ligand_matching_points(
    const std::vector<MatchingArea>& areas
);


std::vector<GpuPoint3>
make_gpu_pose_target_points(
    const std::vector<PoseTarget>& targets
);

// Copy every atom from the current conformer in RDKit atom order, including
// any hydrogens retained by the reader; clash uses this full atom count.
std::vector<GpuPoint3>
make_gpu_ligand_atom_points(
    const RDKit::ROMol& ligand
);


#endif