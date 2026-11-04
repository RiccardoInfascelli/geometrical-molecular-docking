#include "cuda_ligand_batch.hpp"
#include "cuda_pose_pipeline.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void same_clash(const GpuClashScore& a, const GpuClashScore& b) {
    require(a.valid == b.valid && a.total_atoms == b.total_atoms &&
            a.occupied_atoms == b.occupied_atoms && std::abs(a.fraction - b.fraction) <= 1e-12,
            "Segmented clash differs from the single-ligand kernel");
}
void same_fast(const GpuFastScore& a, const GpuFastScore& b) {
    require(a.valid == b.valid && a.total_areas == b.total_areas &&
            a.areas_in_grid == b.areas_in_grid &&
            a.areas_on_occupied_voxel == b.areas_on_occupied_voxel &&
            a.compatible_areas == b.compatible_areas &&
            std::abs(a.weighted_score - b.weighted_score) <= 1e-12,
            "Segmented Fast differs from the single-ligand kernel");
}
struct Buffers {
    GpuProteinGridDevice grid{};
    GpuPointArrayDevice atoms{};
    GpuFastMatchingAreaArrayDevice areas{};
    GpuRigidTransformArrayDevice transforms{};
    GpuLigandBatchSegmentsDevice segments{};
    GpuClashScoreArrayDevice clash{};
    GpuPosePipelineDeviceResult pool;
    ~Buffers() {
        pool.clear();
        cuda_free_clash_scores(clash);
        cuda_free_ligand_batch_segments(segments);
        cuda_free_rigid_transforms(transforms);
        cuda_free_fast_matching_areas(areas);
        cuda_free_points(atoms);
        cuda_free_protein_grid(grid);
    }
};
struct ReferenceScores {
    GpuClashScoreArrayDevice clash{};
    GpuFastScoreArrayDevice fast{};
    ~ReferenceScores() {
        cuda_free_fast_scores(fast);
        cuda_free_clash_scores(clash);
    }
};
GpuRigidTransform transform(double x, bool valid = true) {
    GpuRigidTransform result{};
    result.rotation[0] = result.rotation[4] = result.rotation[8] = 1.0;
    result.translation[0] = x;
    result.valid = valid;
    return result;
}
}

int main() {
    try {
        std::size_t scan_bytes = 99;
        require(cuda_ligand_batch_scan_storage_bytes(0, scan_bytes) && scan_bytes == 0 &&
                !cuda_ligand_batch_scan_storage_bytes(std::numeric_limits<std::uint32_t>::max(), scan_bytes) &&
                scan_bytes == 0, "Scan budget accepted invalid counts or retained a stale size");
        Buffers device;
        GpuProteinGridHost grid;
        grid.voxel_size = 1.0;
        grid.nx = grid.ny = grid.nz = 4;
        grid.voxels.resize(64);
        const auto cell = [](int x, int y, int z) { return x + 4 * (y + 4 * z); };
        for (const auto index : {cell(0, 0, 0), cell(1, 0, 0), cell(0, 2, 0)})
            grid.voxels[index].interactions = 1;
        grid.voxels[cell(1, 1, 1)].occupied = 1;
        require(cuda_upload_protein_grid(grid, device.grid), "Grid upload failed");

        std::vector<GpuPoint3> atoms(43);
        atoms[1] = atoms[3] = {1.0, 1.0, 1.0};
        atoms[2] = {3.0, 1.0, 1.0};
        const std::vector<GpuFastMatchingArea> areas{
            {{0, 0, 0}, 1, {}, 100.0}, {{0, 0, 0}, 1, {}, 2.25},
            {{0, 0, 0}, 1, {}, 1.5}, {{0, 2, 0}, 1, {}, 3.0}};
        const std::vector<GpuRigidTransform> transforms{
            transform(0), transform(1), transform(0, false), transform(0), transform(1)};
        // Empty segments before, between and after active ligands exercise the
        // prefix-boundary lookup. The active ligands have 2 and 40 atoms.
        const std::vector<GpuLigandBatchSegment> segments{
            {0, 0, 0, 1, 0, 1}, {0, 3, 1, 2, 1, 1}, {3, 3, 3, 0, 2, 0},
            {3, 5, 3, 40, 2, 2}, {5, 5, 43, 0, 4, 0}};
        require(cuda_upload_points(atoms, device.atoms) &&
                cuda_upload_fast_matching_areas(areas, device.areas) &&
                cuda_upload_ligand_batch_segments(segments, device.segments),
                "Ligand batch upload failed");
        require(cudaMalloc(reinterpret_cast<void**>(&device.transforms.transforms),
                    transforms.size() * sizeof(GpuRigidTransform)) == cudaSuccess &&
                cudaMemcpy(device.transforms.transforms, transforms.data(),
                    transforms.size() * sizeof(GpuRigidTransform), cudaMemcpyHostToDevice) == cudaSuccess,
                "Transform upload failed");
        device.transforms.count = static_cast<std::uint32_t>(transforms.size());

        std::vector<std::uint32_t> valid_counts;
        require(cuda_score_ligand_batch_clash_device(device.atoms, device.grid,
                    device.segments, device.transforms, device.clash, valid_counts),
                "Segmented clash failed");
        require(valid_counts == std::vector<std::uint32_t>({0, 2, 0, 2, 0}),
                "Per-ligand valid counts differ");
        std::vector<GpuClashScore> clash;
        require(cuda_download_clash_scores(device.clash, clash) && clash.size() == transforms.size(),
                "Clash download failed");
        for (const auto& segment : segments) {
            if (segment.alignment_begin == segment.alignment_end) continue;
            ReferenceScores reference;
            const GpuPointArrayDevice atom_view{
                device.atoms.points + segment.atom_begin, segment.atom_count};
            const GpuRigidTransformArrayDevice transform_view{
                device.transforms.transforms + segment.alignment_begin,
                segment.alignment_end - segment.alignment_begin};
            std::vector<GpuClashScore> expected;
            require(cuda_score_grid_clash_device(atom_view, device.grid, transform_view, reference.clash) &&
                    cuda_download_clash_scores(reference.clash, expected), "Reference clash failed");
            for (std::size_t i = 0; i < expected.size(); ++i)
                same_clash(clash[segment.alignment_begin + i], expected[i]);
        }

        GpuClashCompactionWorkspace workspace;
        std::uint32_t valid = 0;
        require(cuda_compact_clash_survivors(device.transforms, device.clash, 0.05,
                    device.pool.survivors, valid, &workspace) && valid == 4,
                "Shared compaction failed");
        std::vector<GpuRigidTransform> surviving_transforms;
        std::vector<GpuClashScore> surviving_clash;
        std::vector<std::uint32_t> ids;
        require(cuda_download_clash_survivors(device.pool.survivors, surviving_transforms,
                    surviving_clash, ids) && ids == std::vector<std::uint32_t>({1, 3, 4}),
                "Compaction lost cross-ligand survivor identity or order");
        require(cuda_score_ligand_batch_fast_device(device.areas, device.grid,
                    device.segments, device.pool.survivors, device.pool.fast_scores),
                "Segmented Fast failed");
        std::vector<GpuFastScore> scores;
        require(cuda_download_fast_scores(device.pool.fast_scores, scores) && scores.size() == ids.size(),
                "Fast download failed");
        const std::vector<double> expected_scores{2.25, 4.5, 1.5};
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& segment = segments[ids[i] < 3 ? 1 : 3];
            ReferenceScores reference;
            const GpuFastMatchingAreaArrayDevice area_view{
                device.areas.areas + segment.area_begin, segment.area_count};
            const GpuRigidTransformArrayDevice transform_view{device.pool.survivors.transforms + i, 1};
            std::vector<GpuFastScore> expected;
            require(cuda_score_fast_interactions_device(area_view, device.grid, transform_view, reference.fast) &&
                    cuda_download_fast_scores(reference.fast, expected) && expected.size() == 1,
                    "Reference Fast failed");
            same_fast(scores[i], expected.front());
            require(scores[i].valid && scores[i].weighted_score == expected_scores[i],
                    "Fast used another ligand's areas");
        }

        const auto* previous = device.segments.segments;
        for (int kind = 0; kind < 3; ++kind) {
            auto malformed = segments;
            if (kind == 0) malformed[1].alignment_begin = 1;
            if (kind == 1) malformed[1].atom_begin = 0;
            if (kind == 2) malformed[3].area_count = std::numeric_limits<std::uint32_t>::max();
            require(!cuda_upload_ligand_batch_segments(malformed, device.segments) &&
                    device.segments.segments == previous, "Invalid segments replaced an owning buffer");
        }
        auto wrong_atoms = device.atoms;
        --wrong_atoms.count;
        require(!cuda_score_ligand_batch_clash_device(wrong_atoms, device.grid,
                    device.segments, device.transforms, device.clash, valid_counts) &&
                !device.clash.scores && valid_counts.empty(), "Mismatched atom range was accepted");
        auto missing_ids = device.pool.survivors;
        missing_ids.alignment_indices = nullptr;
        require(!cuda_score_ligand_batch_fast_device(device.areas, device.grid,
                    device.segments, missing_ids, device.pool.fast_scores) &&
                !device.pool.fast_scores.scores, "Missing survivor identities were accepted");

        auto empty_segments = segments;
        for (auto& segment : empty_segments) segment.alignment_begin = segment.alignment_end = 0;
        require(cuda_upload_ligand_batch_segments(empty_segments, device.segments) &&
                cuda_score_ligand_batch_clash_device({}, {}, device.segments, {}, device.clash, valid_counts) &&
                valid_counts == std::vector<std::uint32_t>(segments.size(), 0),
                "Zero-alignment ligands lost their empty counts");
        require(cuda_score_ligand_batch_fast_device({}, {}, {}, {}, device.pool.fast_scores),
                "Empty survivor input failed");
        require(cuda_upload_ligand_batch_segments({}, device.segments) &&
                !device.segments.segments && device.segments.count == 0 &&
                cuda_score_ligand_batch_clash_device({}, {}, {}, {}, device.clash, valid_counts) &&
                valid_counts.empty(), "Empty batch left owning outputs");
        cuda_free_ligand_batch_segments(device.segments);
        cuda_free_ligand_batch_segments(device.segments);
        std::cout << "Segmented CUDA clash, shared compaction and Fast equivalence passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
