#ifndef CUDA_LIGAND_BATCH_HPP
#define CUDA_LIGAND_BATCH_HPP

#include "cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// Contiguous slices in concatenated ligand arrays and alignment order. Empty
// ligands retain a segment so output identities do not depend on batch contents.
struct GpuLigandBatchSegment {
    std::uint32_t alignment_begin{}, alignment_end{};
    std::uint32_t atom_begin{}, atom_count{};
    std::uint32_t area_begin{}, area_count{};
};

// Plain owning handle; copying aliases the allocation. Release it once with
// cuda_free_ligand_batch_segments, after all borrowing calls have completed.
struct GpuLigandBatchSegmentsDevice {
    GpuLigandBatchSegment* segments{nullptr};
    std::uint32_t count{};
    std::uint32_t alignment_count{}, atom_count{}, area_count{};
};

bool cuda_upload_ligand_batch_segments(
    const std::vector<GpuLigandBatchSegment>& segments,
    GpuLigandBatchSegmentsDevice& device);
void cuda_free_ligand_batch_segments(GpuLigandBatchSegmentsDevice& device);

// Query the existing compaction scan's scratch requirement before allocation.
// Uses the current device and active count; no scan kernel is launched.
bool cuda_ligand_batch_scan_storage_bytes(std::uint32_t count, std::size_t& bytes);

// One thread per concatenated transform. Clash uses the same device primitive
// as the single-ligand path; valid_counts includes valid transforms even when
// rejected by clash. This does not filter or reorder transforms.
bool cuda_score_ligand_batch_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuLigandBatchSegmentsDevice& segments,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores,
    std::vector<std::uint32_t>& valid_counts);

// Survivor alignment IDs still address the concatenated alignment array. Each
// thread scores the corresponding ligand's areas without altering pose order.
bool cuda_score_ligand_batch_fast_device(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuLigandBatchSegmentsDevice& segments,
    const GpuClashSurvivorArrayDevice& survivors,
    GpuFastScoreArrayDevice& scores);

#endif
