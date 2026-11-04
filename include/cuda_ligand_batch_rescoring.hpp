#ifndef CUDA_LIGAND_BATCH_RESCORING_HPP
#define CUDA_LIGAND_BATCH_RESCORING_HPP

#include "cuda_pose_rescoring.hpp"
#include "gpu_ligand_batch.hpp"

// Borrows the complete resident pool in ligand/survivor order. Chemical input
// uploads and scoring are aggregated; chemistry preparation and CUDA ranking operate on
// each ligand's own population. No normalization mixes different ligands.
// retained_scratch_bytes covers the target's live compaction workspace; shared
// receptor/context and algorithm-internal library temporaries are excluded.
std::vector<GpuDockingResult> rescore_ligand_batch_cuda(
    const GpuPosePipelineDeviceResult& pool,
    const std::vector<GpuPreparedLigand*>& ligands,
    const std::vector<GpuLigandSearchResult>& searches,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t top_k, std::size_t max_device_bytes,
    std::size_t retained_scratch_bytes, GpuDockingBatchTimings* timings = nullptr,
    std::vector<std::exception_ptr>* record_errors = nullptr);

// Return raw selected scores in local survivor order, without ranking or Hybrid
// normalization. Hybrid requests both components only. Download every survivor
// into searches[i].pipeline.poses so the caller can evaluate an unfiltered Oracle.
// Shared scoring/download durations belong to timings, not to each result row;
// selection_seconds is measured separately for each ligand. Inputs stay owned by
// the caller and the blocking operation retains no device resources.
std::vector<PoseRescoringResult> score_ligand_batch_cuda(
    const GpuPosePipelineDeviceResult& pool,
    const std::vector<GpuPreparedLigand*>& ligands,
    std::vector<GpuLigandSearchResult>& searches,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t max_device_bytes, std::size_t retained_scratch_bytes,
    GpuDockingBatchTimings* timings = nullptr,
    std::vector<std::exception_ptr>* record_errors = nullptr);

#endif
