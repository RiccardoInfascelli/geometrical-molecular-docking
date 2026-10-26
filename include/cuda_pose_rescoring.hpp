#ifndef CUDA_POSE_RESCORING_HPP
#define CUDA_POSE_RESCORING_HPP

#include "gpu_docking.hpp"
#include "cuda_pose_ranking.hpp"
#include "pose_rescoring.hpp"

// Persistent, typed receptor data. Chemistry is prepared on the host once; no
// receptor molecule or RDKit pointer is dereferenced by a CUDA kernel.
struct CudaRescoringTarget {
    GpuWeightedInteractionAreasDevice weighted_interaction{};
    GpuVinardoAtomsDevice vinardo{};
    CudaRescoringTarget(const RescoringTarget& target, PoseRescoringMethods methods);
    ~CudaRescoringTarget();
    CudaRescoringTarget(const CudaRescoringTarget&) = delete;
    CudaRescoringTarget& operator=(const CudaRescoringTarget&) = delete;
};

// Borrow the complete device survivor pool. Only the explicitly requested Fast
// prefilter may reduce it before WeightedInteraction/Vinardo; ranking downloads top-k only.
// Inputs remain owned by the caller. The blocking call releases only its own
// ligand/scoring buffers and throws on CUDA failure or invalid ranking inputs.
GpuDockingResult rescore_pose_pool_cuda(const GpuPosePipelineDeviceResult& pool,
    const RDKit::ROMol& molecule, const PreparedLigand& ligand,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t top_k);

#endif
