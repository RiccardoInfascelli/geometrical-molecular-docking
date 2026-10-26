#include <iostream>
#include <cstdlib>
#include <chrono>
#include "cuda_pose_pipeline.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

void GpuPosePipelineDeviceResult::clear() noexcept
{
    // An unused owner must not add cudaFree(nullptr) calls to Fast/CPU paths.
    if (fast_scores.scores) cuda_free_fast_scores(fast_scores);
    else fast_scores = {};
    if (survivors.transforms || survivors.clash_scores || survivors.alignment_indices)
        cuda_free_clash_survivors(survivors);
    else survivors = {};
}

GpuPosePipelineDeviceResult::~GpuPosePipelineDeviceResult() { clear(); }


bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuAlignmentCandidateArrayDevice& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result,
    GpuPosePipelineTimings* timings,
    GpuPosePipelineDeviceResult* device_result
)
{
    if (device_result) device_result->clear();
    if (timings) {
        const bool collect_stages = timings->collect_stages;
        *timings = GpuPosePipelineTimings{};
        timings->collect_stages = collect_stages;
    }

    // Optional diagnostics add synchronization around each stage; their timings
    // should not be mixed with measurements from normal screening runs.
    const bool enable_stage_timings =
        std::getenv("DOCKING_STAGE_TIMINGS") != nullptr;
    const bool collect_stages = timings && timings->collect_stages;

    const auto timed_pose_stage =
        [&](const char* label, std::size_t elements, const char* scope,
            auto&& operation) -> bool
        {
            if (!enable_stage_timings && !collect_stages)
                return operation();

            if (cudaDeviceSynchronize() != cudaSuccess)
                return false;

            const auto start =
                std::chrono::steady_clock::now();

            const bool success = operation();

            const cudaError_t sync_status =
                cudaDeviceSynchronize();

            const auto end =
                std::chrono::steady_clock::now();

            const double milliseconds =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            if (enable_stage_timings) {
                std::cout
                    << "CUDA_POSE_MS,"
                    << label
                    << ","
                    << milliseconds
                    << '\n';
            }

            if (collect_stages && success && sync_status == cudaSuccess)
                timings->stages.push_back({label, milliseconds, elements, scope});

            return success && sync_status == cudaSuccess;
        };


    result =
        GpuPosePipelineResult{};


    if (
        !std::isfinite(
            clash_threshold
        ) ||
        clash_threshold < 0.0 ||
        clash_threshold > 1.0
    ) {
        return false;
    }


    result.input_alignment_count =
        alignments.count;


    if (alignments.count == 0) {
        return true;
    }


    if (alignments.alignments == nullptr) {
        return false;
    }


    // ========================================================
    // 1. Kabsch - consume alignment candidates directly on GPU
    // ========================================================

    GpuRigidTransformArrayDevice
        device_transforms{};


    if (
        !timed_pose_stage("kabsch", alignments.count,
            "device_wrapper_allocation_gather_and_sync", [&]() { return cuda_compute_kabsch_transforms_device(
            ligand_matching_points,
            protein_target_points,
            alignments,
            device_transforms
        ); })
    ) {
        return false;
    }


    if (
        device_transforms.count !=
        alignments.count
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    // ========================================================
    // 2. Grid clash - consume Kabsch output directly on GPU
    // ========================================================

    GpuClashScoreArrayDevice
        device_clash_scores{};


    if (
        !timed_pose_stage("grid_clash", device_transforms.count,
            "device_wrapper_allocation_atom_transform_and_sync", [&]() { return cuda_score_grid_clash_device(
            ligand_atoms,
            protein_grid,
            device_transforms,
            device_clash_scores
        ); })
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    if (
        device_clash_scores.count !=
        device_transforms.count
    ) {
        cuda_free_clash_scores(
            device_clash_scores
        );

        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    // ========================================================
    // 3. GPU clash compaction
    // ========================================================

    GpuClashSurvivorArrayDevice
        device_survivors{};


    std::uint32_t
        valid_transform_count = 0;


    if (
        !timed_pose_stage("clash_compaction", device_transforms.count,
            "device_wrapper_allocation_scan_scatter_status_d2h_and_sync", [&]() { return cuda_compact_clash_survivors(
            device_transforms,
            device_clash_scores,
            clash_threshold,
            device_survivors,
            valid_transform_count
        ); })
    ) {
        cuda_free_clash_scores(
            device_clash_scores
        );

        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    result.valid_transform_count =
        valid_transform_count;


    result.clash_survivor_count =
        device_survivors.count;


    // Full pre-compaction arrays are no longer needed.
    cuda_free_clash_scores(
        device_clash_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    if (
        device_survivors.count == 0
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return true;
    }


    // ========================================================
    // 4. Device-resident fast interaction scoring
    // ========================================================

    // Borrow the compacted transforms; device_survivors remains their owner.
    const GpuRigidTransformArrayDevice
        survivor_transform_view{
            device_survivors.transforms,
            device_survivors.count
        };


    GpuFastScoreArrayDevice
        device_fast_scores{};


    if (
        !timed_pose_stage("fast_scoring", device_survivors.count,
            "device_wrapper_allocation_area_transform_and_sync", [&]() { return cuda_score_fast_interactions_device(
            ligand_fast_areas,
            protein_grid,
            survivor_transform_view,
            device_fast_scores
        ); })
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    if (
        device_fast_scores.count !=
        device_survivors.count
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    // ========================================================
    // Final survivor download
    // ========================================================

    if (device_result) {
        // Transfer these allocations to the result owner. They no longer depend
        // on the temporary alignment and ligand buffers released by the caller.
        device_result->survivors = device_survivors;
        device_result->fast_scores = device_fast_scores;
        return true;
    }

    std::vector<GpuRigidTransform>
        survivor_transforms;


    std::vector<std::uint32_t>
        survivor_alignment_indices;


    std::vector<GpuClashScore>
        survivor_clash_scores;


    std::vector<GpuFastScore>
        fast_scores;


    if (
        !timed_pose_stage("survivor_download", device_survivors.count,
            "d2h_wrapper_host_allocation_and_sync", [&]() {
            const auto start = timings
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};

            const bool success = cuda_download_clash_survivors(
                device_survivors,
                survivor_transforms,
                survivor_clash_scores,
                survivor_alignment_indices
            );

            if (timings) {
                timings->survivor_download_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start
                    ).count();
            }

            return success;
        })
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    if (
        !timed_pose_stage("fast_score_download", device_survivors.count,
            "d2h_wrapper_host_allocation_and_sync", [&]() {
            const auto start = timings
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};

            const bool success = cuda_download_fast_scores(
                device_fast_scores,
                fast_scores
            );

            if (timings) {
                timings->fast_score_download_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start
                    ).count();
            }

            return success;
        })
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    cuda_free_fast_scores(
        device_fast_scores
    );


    cuda_free_clash_survivors(
        device_survivors
    );


    if (
        survivor_transforms.size() !=
            result.clash_survivor_count ||

        survivor_clash_scores.size() !=
            result.clash_survivor_count ||

        survivor_alignment_indices.size() !=
            result.clash_survivor_count ||

        fast_scores.size() !=
            result.clash_survivor_count
    ) {
        return false;
    }


    // ========================================================
    // 5. Assemble surviving poses
    // ========================================================

    const auto assembly_start =
        std::chrono::steady_clock::now();


    result.poses.reserve(
        survivor_transforms.size()
    );


    for (
        std::size_t i = 0;
        i < survivor_transforms.size();
        ++i
    ) {
        if (
            fast_scores[i].valid == 0
        ) {
            return false;
        }


        GpuPipelinePose pose;


        pose.alignment_index =
            survivor_alignment_indices[i];


        pose.transform =
            survivor_transforms[i];


        pose.clash =
            survivor_clash_scores[i];


        pose.fast_score =
            fast_scores[i];


        result.poses.push_back(
            pose
        );
    }


    if (enable_stage_timings || collect_stages) {
        const auto assembly_end =
            std::chrono::steady_clock::now();

        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                assembly_end - assembly_start
            ).count();

        if (enable_stage_timings) {
            std::cout
                << "CUDA_POSE_MS,pose_assembly,"
                << milliseconds
                << '\n';
        }
        if (collect_stages)
            timings->stages.push_back({"pose_assembly", milliseconds,
                result.poses.size(), "host_allocation_and_copy"});
    }


    return true;
}


// ============================================================
// Legacy host-vector API
//
// Used by the existing synthetic pipeline test.
// The production/device path above does not perform this
// host-to-device alignment copy.
// ============================================================

bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuAlignmentCandidate>& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result,
    GpuPosePipelineTimings* timings
)
{
    if (timings) {
        const bool collect_stages = timings->collect_stages;
        *timings = GpuPosePipelineTimings{};
        timings->collect_stages = collect_stages;
    }

    if (
        alignments.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    // Reject alignment indices outside the uploaded point arrays.
    for (const auto& alignment :
         alignments) {

        for (int i = 0; i < 3; ++i) {

            if (
                alignment.ligand_indices[i] >=
                    ligand_matching_points.count ||

                alignment.protein_indices[i] >=
                    protein_target_points.count
            ) {
                return false;
            }
        }
    }


    GpuAlignmentCandidateArrayDevice
        device_alignments{};


    if (!alignments.empty()) {

        const std::size_t bytes =
            alignments.size() *
            sizeof(GpuAlignmentCandidate);


        if (
            cudaMalloc(
                reinterpret_cast<void**>(
                    &device_alignments.alignments
                ),
                bytes
            ) != cudaSuccess
        ) {
            return false;
        }


        device_alignments.count =
            static_cast<std::uint32_t>(
                alignments.size()
            );


        if (
            cudaMemcpy(
                device_alignments.alignments,
                alignments.data(),
                bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            cuda_free_alignment_candidates(
                device_alignments
            );

            return false;
        }
    }


    const bool ok =
        cuda_run_pose_pipeline(
            ligand_matching_points,
            protein_target_points,
            ligand_atoms,
            ligand_fast_areas,
            protein_grid,
            device_alignments,
            clash_threshold,
            result,
            timings
        );


    cuda_free_alignment_candidates(
        device_alignments
    );


    return ok;
}
