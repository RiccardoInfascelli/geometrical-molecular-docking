#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "cuda_triplet_types.hpp"
#include "cuda_grid.hpp"
#include "cuda_fast_scoring.hpp"

#include <cstdint>
#include <vector>


inline constexpr std::uint32_t
GPU_TRIPLET_KEY_NOT_FOUND = 0xFFFFFFFFu;


// ============================================================
// Device-resident intermediate arrays
//
// These buffers allow the pose pipeline to keep intermediate
// results on the GPU instead of downloading them after every
// stage.
// They are plain handles: copying one aliases the same allocation.
// Initialize owning outputs with {} and release them with the matching
// cuda_free_* function; do not free a borrowed view separately.
// Device inputs are borrowed. A device result does not imply an asynchronous
// call: implementations may synchronize or copy small status values to the host.
// ============================================================

struct GpuClashScoreArrayDevice
{
    GpuClashScore* scores{nullptr};
    std::uint32_t count{};
};

struct GpuClashSurvivorArrayDevice
{
    GpuRigidTransform* transforms{nullptr};
    GpuClashScore* clash_scores{nullptr};
    std::uint32_t* alignment_indices{nullptr};
    std::uint32_t count{};
};


struct GpuFastScoreArrayDevice
{
    GpuFastScore* scores{nullptr};
    std::uint32_t count{};
};

struct GpuAlignmentCandidateArrayDevice
{
    GpuAlignmentCandidate* alignments{nullptr};
    std::uint32_t count{};
};


bool cuda_expand_alignment_candidates_device(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const std::vector<std::uint32_t>& alignment_offsets,
    GpuAlignmentCandidateArrayDevice& alignments
);


bool cuda_download_alignment_candidates(
    const GpuAlignmentCandidateArrayDevice& device_alignments,
    std::vector<GpuAlignmentCandidate>& host_alignments
);


void cuda_free_alignment_candidates(
    GpuAlignmentCandidateArrayDevice& alignments
);
// ============================================================
// General
// ============================================================

bool cuda_smoke_test();


// ============================================================
// Triplet index
// ============================================================

bool cuda_upload_triplet_index(
    const GpuFlatTripletIndexHost& host_index,
    GpuFlatTripletIndexDevice& device_index
);


void cuda_free_triplet_index(
    GpuFlatTripletIndexDevice& device_index
);


// Return one key index per query, or GPU_TRIPLET_KEY_NOT_FOUND.
bool cuda_lookup_triplet_keys_exact(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& key_indices
);


// Return 27 slots per query for distance-bin offsets [-1, 1]^3, with d2
// varying fastest. Chemical keys stay fixed; missing and negative-bin slots
// use GPU_TRIPLET_KEY_NOT_FOUND. This does not check real triangle side lengths.
bool cuda_lookup_triplet_key_neighbors(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& neighbor_key_indices
);


struct GpuTripletNeighborCountArrayDevice
{
    std::uint32_t* counts{nullptr};
    std::uint32_t count{};
};


// Count records in all 27 neighboring bins for each query. Count, scan and
// expansion must use the same queries and the same unchanged protein index.
bool cuda_count_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    GpuTripletNeighborCountArrayDevice& counts
);


bool cuda_download_triplet_neighbor_counts(
    const GpuTripletNeighborCountArrayDevice& device_counts,
    std::vector<std::uint32_t>& host_counts
);


void cuda_free_triplet_neighbor_counts(
    GpuTripletNeighborCountArrayDevice& counts
);


bool cuda_count_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& record_counts
);


// Produce counts.size() + 1 offsets: [offsets[i], offsets[i + 1]) is the
// output range for input i, and the final offset is the total record count.
bool cuda_exclusive_prefix_sum(
    const std::vector<std::uint32_t>& counts,
    std::vector<std::uint32_t>& offsets
);


struct GpuTripletCandidateArrayDevice
{
    GpuTripletCandidate* candidates{nullptr};
    std::uint32_t count{};
};


bool cuda_expand_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    GpuTripletCandidateArrayDevice& candidates
);


bool cuda_download_triplet_candidates(
    const GpuTripletCandidateArrayDevice& device_candidates,
    std::vector<GpuTripletCandidate>& host_candidates
);


void cuda_free_triplet_candidates(
    GpuTripletCandidateArrayDevice& candidates
);


bool cuda_expand_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    std::vector<GpuTripletCandidate>& candidates
);


struct GpuAlignmentCountArrayDevice
{
    std::uint32_t* counts{nullptr};
    std::uint32_t count{};
};


bool cuda_count_alignment_candidates_device(
    const GpuTripletCandidateArrayDevice& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    GpuAlignmentCountArrayDevice& counts
);


bool cuda_download_alignment_counts(
    const GpuAlignmentCountArrayDevice& device_counts,
    std::vector<std::uint32_t>& host_counts
);


void cuda_free_alignment_counts(
    GpuAlignmentCountArrayDevice& counts
);


struct GpuAlignmentOffsetArrayDevice
{
    std::uint32_t* offsets{nullptr};
    std::uint32_t count{};
};


// Offsets include a final total. Counts and their cumulative total must fit
// uint32_t; overflow fails rather than truncating the alignment population.
bool cuda_exclusive_prefix_sum_device(
    const GpuAlignmentCountArrayDevice& counts,
    GpuAlignmentOffsetArrayDevice& offsets
);


bool cuda_download_alignment_offsets(
    const GpuAlignmentOffsetArrayDevice& device_offsets,
    std::vector<std::uint32_t>& host_offsets
);


void cuda_free_alignment_offsets(
    GpuAlignmentOffsetArrayDevice& offsets
);


struct GpuTripletNeighborOffsetArrayDevice
{
    std::uint32_t* offsets{nullptr};
    std::uint32_t count{};
};


bool cuda_exclusive_triplet_neighbor_prefix_sum_device(
    const GpuTripletNeighborCountArrayDevice& counts,
    GpuTripletNeighborOffsetArrayDevice& offsets
);


bool cuda_download_triplet_neighbor_offsets(
    const GpuTripletNeighborOffsetArrayDevice& device_offsets,
    std::vector<std::uint32_t>& host_offsets
);


void cuda_free_triplet_neighbor_offsets(
    GpuTripletNeighborOffsetArrayDevice& offsets
);


bool cuda_expand_triplet_neighbor_records_device(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const GpuTripletNeighborCountArrayDevice& device_counts,
    const GpuTripletNeighborOffsetArrayDevice& device_offsets,
    GpuTripletCandidateArrayDevice& candidates
);


bool cuda_expand_alignment_candidates_device(
    const GpuTripletCandidateArrayDevice& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const GpuAlignmentOffsetArrayDevice& alignment_offsets,
    GpuAlignmentCandidateArrayDevice& alignments
);




bool cuda_count_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    std::vector<std::uint32_t>& alignment_counts
);


bool cuda_expand_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const std::vector<std::uint32_t>& alignment_offsets,
    std::vector<GpuAlignmentCandidate>& alignments
);


// ============================================================
// Point arrays
// ============================================================

bool cuda_upload_points(
    const std::vector<GpuPoint3>& host_points,
    GpuPointArrayDevice& device_points
);


bool cuda_download_points(
    const GpuPointArrayDevice& device_points,
    std::vector<GpuPoint3>& host_points
);


void cuda_free_points(
    GpuPointArrayDevice& device_points
);


// ============================================================
// Kabsch
//
// Legacy API:
//     GPU computation -> host vector.
//
// Device API:
//     GPU computation -> GPU buffer.
// One transform is produced per alignment, in input order. Device-native
// alignment indices must address the supplied point arrays. SVD is the default;
// DOCKING_KABSCH_METHOD=quaternion selects the double-precision Jacobi alternative.
// The valid flag reports cuSOLVER status or the quaternion norm guard, not rank
// or finiteness. Both methods synchronize before returning the device buffer.
// ============================================================

bool cuda_compute_kabsch_transforms(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    std::vector<GpuRigidTransform>& transforms
);

bool cuda_compute_kabsch_transforms_device(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const GpuAlignmentCandidateArrayDevice& alignments,
    GpuRigidTransformArrayDevice& transforms
);


bool cuda_compute_kabsch_transforms_device(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    GpuRigidTransformArrayDevice& transforms
);


bool cuda_download_rigid_transforms(
    const GpuRigidTransformArrayDevice& device_transforms,
    std::vector<GpuRigidTransform>& host_transforms
);


void cuda_free_rigid_transforms(
    GpuRigidTransformArrayDevice& device_transforms
);


// ============================================================
// Protein grid
// ============================================================

bool cuda_upload_protein_grid(
    const GpuProteinGridHost& host_grid,
    GpuProteinGridDevice& device_grid
);


void cuda_free_protein_grid(
    GpuProteinGridDevice& device_grid
);


// ============================================================
// Grid clash
//
// Legacy API:
//     host transforms -> GPU -> host scores.
//
// Device API:
//     GPU transforms -> GPU scores.
// Each thread transforms and checks every atom of one pose. Out-of-grid atoms
// remain in the denominator but add no clash; no threshold is applied here.
// The device wrapper synchronizes before returning its score buffer.
// ============================================================

bool cuda_score_grid_clash(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuClashScore>& scores
);


bool cuda_score_grid_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores
);


bool cuda_download_clash_scores(
    const GpuClashScoreArrayDevice& device_scores,
    std::vector<GpuClashScore>& host_scores
);


void cuda_free_clash_scores(
    GpuClashScoreArrayDevice& device_scores
);


// Keep valid transforms with valid clash scores and fraction <= threshold.
// Preserve input order and retain original alignment indices for the survivors.
// valid_transform_count also includes valid transforms rejected by clash.
bool cuda_compact_clash_survivors(
    const GpuRigidTransformArrayDevice& transforms,
    const GpuClashScoreArrayDevice& clash_scores,
    double clash_threshold,
    GpuClashSurvivorArrayDevice& survivors,
    std::uint32_t& valid_transform_count
);


bool cuda_download_clash_survivors(
    const GpuClashSurvivorArrayDevice& device_survivors,
    std::vector<GpuRigidTransform>& host_transforms,
    std::vector<GpuClashScore>& host_clash_scores,
    std::vector<std::uint32_t>& host_alignment_indices
);


void cuda_free_clash_survivors(
    GpuClashSurvivorArrayDevice& survivors
);

// ============================================================
// Fast matching areas
// ============================================================

bool cuda_upload_fast_matching_areas(
    const std::vector<GpuFastMatchingArea>& host_areas,
    GpuFastMatchingAreaArrayDevice& device_areas
);


void cuda_free_fast_matching_areas(
    GpuFastMatchingAreaArrayDevice& device_areas
);


// ============================================================
// Fast interaction scoring
// One thread sums logical-area weights for one transform, in input area order.
// Device output is complete on return; ranking and normalization are separate.
// ============================================================

bool cuda_score_fast_interactions_device(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuFastScoreArrayDevice& scores
);


bool cuda_download_fast_scores(
    const GpuFastScoreArrayDevice& device_scores,
    std::vector<GpuFastScore>& host_scores
);


void cuda_free_fast_scores(
    GpuFastScoreArrayDevice& scores
);


// Legacy host-vector API.
bool cuda_score_fast_interactions(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuFastScore>& scores
);


#endif
