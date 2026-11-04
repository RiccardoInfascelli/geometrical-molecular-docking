#include "cuda_ligand_batch.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

__device__ std::uint32_t find_segment(
    const GpuLigandBatchSegment* segments, std::uint32_t count,
    std::uint32_t alignment_index)
{
    std::uint32_t first = 0, last = count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        if (segments[middle].alignment_end <= alignment_index)
            first = middle + 1;
        else
            last = middle;
    }
    return first;
}

__global__ void batch_clash_kernel(
    const GpuPoint3* atoms, GpuProteinGridDevice grid,
    const GpuLigandBatchSegment* segments, std::uint32_t segment_count,
    const GpuRigidTransform* transforms, std::uint32_t transform_count,
    GpuClashScore* scores, std::uint32_t* valid_counts)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= transform_count) return;
    const auto ligand = find_segment(segments, segment_count, index);
    if (ligand == segment_count) {
        scores[index] = GpuClashScore{};
        return;
    }
    const auto segment = segments[ligand];
    const auto transform = transforms[index];
    if (transform.valid) atomicAdd(valid_counts + ligand, 1u);
    const auto* ligand_atoms = segment.atom_count ? atoms + segment.atom_begin : nullptr;
    scores[index] = docking_device::grid_clash(
        ligand_atoms, segment.atom_count, grid, transform);
}

__global__ void batch_fast_kernel(
    const GpuFastMatchingArea* areas, GpuProteinGridDevice grid,
    const GpuLigandBatchSegment* segments, std::uint32_t segment_count,
    const GpuRigidTransform* transforms, const std::uint32_t* alignment_indices,
    std::uint32_t survivor_count, GpuFastScore* scores)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= survivor_count) return;
    const auto ligand = find_segment(segments, segment_count, alignment_indices[index]);
    if (ligand == segment_count) {
        scores[index] = GpuFastScore{};
        return;
    }
    const auto segment = segments[ligand];
    const auto* ligand_areas = segment.area_count ? areas + segment.area_begin : nullptr;
    scores[index] = docking_device::fast_score(
        ligand_areas, segment.area_count, grid, transforms[index]);
}

bool has_segments(const GpuLigandBatchSegmentsDevice& segments)
{
    return segments.count != 0 && segments.segments != nullptr &&
           segments.alignment_count <= static_cast<std::uint32_t>(INT_MAX);
}

} // namespace

bool cuda_ligand_batch_scan_storage_bytes(std::uint32_t count, std::size_t& bytes)
{
    bytes = 0;
    if (count > static_cast<std::uint32_t>(INT_MAX)) return false;
    if (count == 0) return true;
    std::size_t required = 0;
    if (cub::DeviceScan::ExclusiveSum(nullptr, required,
            static_cast<std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr),
            static_cast<int>(count)) != cudaSuccess)
        return false;
    bytes = required;
    return true;
}

void cuda_free_ligand_batch_segments(GpuLigandBatchSegmentsDevice& device)
{
    if (device.segments) cudaFree(device.segments);
    device = {};
}

bool cuda_upload_ligand_batch_segments(
    const std::vector<GpuLigandBatchSegment>& segments,
    GpuLigandBatchSegmentsDevice& device)
{
    if (segments.size() > std::numeric_limits<std::uint32_t>::max() ||
        segments.size() > std::numeric_limits<std::size_t>::max() / sizeof(GpuLigandBatchSegment))
        return false;
    if (segments.empty()) {
        cuda_free_ligand_batch_segments(device);
        return true;
    }

    GpuLigandBatchSegmentsDevice replacement;
    for (const auto& segment : segments) {
        if (segment.alignment_begin != replacement.alignment_count ||
            segment.alignment_end < segment.alignment_begin ||
            segment.alignment_end > static_cast<std::uint32_t>(INT_MAX) ||
            segment.atom_begin != replacement.atom_count ||
            segment.area_begin != replacement.area_count ||
            segment.atom_count > std::numeric_limits<std::uint32_t>::max() - segment.atom_begin ||
            segment.area_count > std::numeric_limits<std::uint32_t>::max() - segment.area_begin)
            return false;
        replacement.alignment_count = segment.alignment_end;
        replacement.atom_count = segment.atom_begin + segment.atom_count;
        replacement.area_count = segment.area_begin + segment.area_count;
    }

    const auto bytes = segments.size() * sizeof(GpuLigandBatchSegment);
    if (cudaMalloc(reinterpret_cast<void**>(&replacement.segments), bytes) != cudaSuccess)
        return false;
    if (cudaMemcpy(replacement.segments, segments.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_free_ligand_batch_segments(replacement);
        return false;
    }
    replacement.count = static_cast<std::uint32_t>(segments.size());
    cuda_free_ligand_batch_segments(device);
    device = replacement;
    return true;
}

bool cuda_score_ligand_batch_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuLigandBatchSegmentsDevice& segments,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores,
    std::vector<std::uint32_t>& valid_counts)
{
    cuda_free_clash_scores(scores);
    valid_counts.clear();
    if (transforms.count != segments.alignment_count ||
        (segments.count && !segments.segments))
        return false;
    valid_counts.assign(segments.count, 0);
    if (transforms.count == 0) return true;
    if (!has_segments(segments) || !transforms.transforms ||
        ligand_atoms.count != segments.atom_count ||
        (ligand_atoms.count && !ligand_atoms.points) ||
        !protein_grid.voxels || protein_grid.voxel_count == 0) {
        valid_counts.clear();
        return false;
    }

    GpuClashScore* device_scores = nullptr;
    std::uint32_t* device_valid_counts = nullptr;
    const auto cleanup = [&] {
        if (device_valid_counts) cudaFree(device_valid_counts);
        if (device_scores) cudaFree(device_scores);
    };
    const auto score_bytes = std::size_t(transforms.count) * sizeof(GpuClashScore);
    const auto count_bytes = std::size_t(segments.count) * sizeof(std::uint32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&device_scores), score_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_valid_counts), count_bytes) != cudaSuccess ||
        cudaMemset(device_valid_counts, 0, count_bytes) != cudaSuccess) {
        cleanup();
        valid_counts.clear();
        return false;
    }

    constexpr std::uint32_t block_size = 256;
    const auto blocks = (transforms.count + block_size - 1) / block_size;
    batch_clash_kernel<<<blocks, block_size>>>(
        ligand_atoms.points, protein_grid, segments.segments, segments.count,
        transforms.transforms, transforms.count, device_scores, device_valid_counts);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(valid_counts.data(), device_valid_counts, count_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        valid_counts.clear();
        return false;
    }
    if (cudaFree(device_valid_counts) != cudaSuccess) {
        cudaFree(device_scores);
        valid_counts.clear();
        return false;
    }
    scores = {device_scores, transforms.count};
    return true;
}

bool cuda_score_ligand_batch_fast_device(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuLigandBatchSegmentsDevice& segments,
    const GpuClashSurvivorArrayDevice& survivors,
    GpuFastScoreArrayDevice& scores)
{
    cuda_free_fast_scores(scores);
    if (survivors.count == 0) return true;
    if (!has_segments(segments) || !survivors.transforms || !survivors.alignment_indices ||
        survivors.count > segments.alignment_count || ligand_areas.count != segments.area_count ||
        (ligand_areas.count && !ligand_areas.areas) ||
        !protein_grid.voxels || protein_grid.voxel_count == 0)
        return false;

    GpuFastScore* device_scores = nullptr;
    const auto bytes = std::size_t(survivors.count) * sizeof(GpuFastScore);
    if (cudaMalloc(reinterpret_cast<void**>(&device_scores), bytes) != cudaSuccess)
        return false;
    constexpr std::uint32_t block_size = 256;
    const auto blocks = (survivors.count + block_size - 1) / block_size;
    batch_fast_kernel<<<blocks, block_size>>>(
        ligand_areas.areas, protein_grid, segments.segments, segments.count,
        survivors.transforms, survivors.alignment_indices, survivors.count, device_scores);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(device_scores);
        return false;
    }
    scores = {device_scores, survivors.count};
    return true;
}
