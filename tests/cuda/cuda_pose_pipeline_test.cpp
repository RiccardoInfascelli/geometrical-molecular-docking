#include "cuda_pose_pipeline.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>


namespace
{

bool close(
    double a,
    double b,
    double tolerance = 1e-10
)
{
    return
        std::abs(a - b) <=
        tolerance;
}


std::size_t grid_index(
    int x,
    int y,
    int z
)
{
    constexpr int nx = 4;
    constexpr int ny = 4;

    return
        static_cast<std::size_t>(
            x +
            nx * (
                y +
                ny * z
            )
        );
}

bool same_results(const GpuPosePipelineResult& a, const GpuPosePipelineResult& b)
{
    if (a.input_alignment_count != b.input_alignment_count ||
        a.valid_transform_count != b.valid_transform_count ||
        a.clash_survivor_count != b.clash_survivor_count ||
        a.poses.size() != b.poses.size()) return false;
    for (std::size_t i = 0; i < a.poses.size(); ++i) {
        const auto& x = a.poses[i];
        const auto& y = b.poses[i];
        if (x.alignment_index != y.alignment_index ||
            x.transform.valid != y.transform.valid || x.clash.valid != y.clash.valid ||
            x.clash.total_atoms != y.clash.total_atoms ||
            x.clash.occupied_atoms != y.clash.occupied_atoms ||
            !close(x.clash.fraction, y.clash.fraction) ||
            x.fast_score.valid != y.fast_score.valid ||
            x.fast_score.total_areas != y.fast_score.total_areas ||
            x.fast_score.areas_in_grid != y.fast_score.areas_in_grid ||
            x.fast_score.areas_on_occupied_voxel != y.fast_score.areas_on_occupied_voxel ||
            x.fast_score.compatible_areas != y.fast_score.compatible_areas ||
            !close(x.fast_score.weighted_score, y.fast_score.weighted_score)) return false;
        for (int j = 0; j < 9; ++j) {
            if (!close(x.transform.rotation[j], y.transform.rotation[j])) return false;
        }
        for (int j = 0; j < 3; ++j) {
            if (!close(x.transform.translation[j], y.transform.translation[j])) return false;
        }
    }
    return true;
}

bool same_profile_stages(const StageProfile& stages, std::size_t expected_count,
                         std::size_t elements)
{
    const char* expected[] = {
        "kabsch", "grid_clash", "clash_compaction", "fast_scoring",
        "survivor_download", "fast_score_download", "pose_assembly"
    };
    if (stages.size() != expected_count) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].phase != expected[i] || stages[i].elements != elements ||
            !std::isfinite(stages[i].milliseconds) || stages[i].milliseconds < 0.0 ||
            stages[i].scope.empty()) return false;
    }
    return true;
}

bool test_reused_clash_compaction()
{
    constexpr std::size_t maximum_count = 65537;
    struct Inputs {
        GpuRigidTransformArrayDevice transforms{};
        GpuClashScoreArrayDevice clashes{};
        ~Inputs() {
            cuda_free_clash_scores(clashes);
            cuda_free_rigid_transforms(transforms);
        }
    } input;
    if (cudaMalloc(reinterpret_cast<void**>(&input.transforms.transforms),
                   maximum_count * sizeof(GpuRigidTransform)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&input.clashes.scores),
                   maximum_count * sizeof(GpuClashScore)) != cudaSuccess)
        return false;

    GpuClashCompactionWorkspace workspace;
    const auto run = [&](std::size_t count, bool invalidate_all = false) {
        std::vector<GpuRigidTransform> transforms(count);
        std::vector<GpuClashScore> clashes(count);
        std::vector<std::uint32_t> expected_ids;
        std::uint32_t expected_valid = 0;
        for (std::size_t i = 0; i < count; ++i) {
            auto& transform = transforms[i];
            transform.rotation[0] = transform.rotation[4] = transform.rotation[8] = 1.0;
            transform.translation[0] = static_cast<double>(i);
            transform.translation[1] = -static_cast<double>(i);
            transform.valid = !invalidate_all && i % 4 != 0;
            clashes[i].total_atoms = 4;
            clashes[i].occupied_atoms = static_cast<std::uint32_t>(i % 5);
            clashes[i].fraction = clashes[i].occupied_atoms / 4.0;
            clashes[i].valid = i % 7 != 0;
            expected_valid += transform.valid;
            if (transform.valid && clashes[i].valid && clashes[i].fraction <= 0.5)
                expected_ids.push_back(static_cast<std::uint32_t>(i));
        }
        input.transforms.count = input.clashes.count = static_cast<std::uint32_t>(count);
        if (count && (cudaMemcpy(input.transforms.transforms, transforms.data(),
                count * sizeof(GpuRigidTransform), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(input.clashes.scores, clashes.data(),
                count * sizeof(GpuClashScore), cudaMemcpyHostToDevice) != cudaSuccess))
            return false;

        const auto* previous_flags = workspace.flags;
        const auto* previous_offsets = workspace.offsets;
        const auto* previous_valid_count = workspace.valid_count;
        const auto previous_capacity = workspace.capacity;
        GpuPosePipelineDeviceResult legacy, reused;
        std::uint32_t legacy_valid = 99, reused_valid = 99;
        if (!cuda_compact_clash_survivors(input.transforms, input.clashes, 0.5,
                legacy.survivors, legacy_valid) ||
            !cuda_compact_clash_survivors(input.transforms, input.clashes, 0.5,
                reused.survivors, reused_valid, &workspace) ||
            legacy_valid != expected_valid || reused_valid != expected_valid ||
            legacy.survivors.count != expected_ids.size() ||
            reused.survivors.count != expected_ids.size())
            return false;
        if (count <= previous_capacity &&
            (workspace.flags != previous_flags || workspace.offsets != previous_offsets ||
             workspace.valid_count != previous_valid_count || workspace.capacity != previous_capacity))
            return false;

        GpuPosePipelineResult expected_host, legacy_host, reused_host;
        for (const auto id : expected_ids)
            expected_host.poses.push_back({id, transforms[id], clashes[id], {}});
        for (auto* owner : {&legacy, &reused}) {
            std::vector<GpuRigidTransform> output_transforms;
            std::vector<GpuClashScore> output_clashes;
            std::vector<std::uint32_t> ids;
            if (!cuda_download_clash_survivors(owner->survivors, output_transforms,
                    output_clashes, ids) || ids != expected_ids ||
                output_transforms.size() != ids.size() || output_clashes.size() != ids.size())
                return false;
            auto& host = owner == &legacy ? legacy_host : reused_host;
            for (std::size_t i = 0; i < ids.size(); ++i)
                host.poses.push_back({ids[i], output_transforms[i], output_clashes[i], {}});
        }
        return same_results(legacy_host, reused_host) && same_results(expected_host, reused_host);
    };

    // Grow across block boundaries, then reuse the storage for smaller and empty
    // inputs. An all-invalid call catches a stale valid-transform counter.
    if (!run(1) || !run(8) || !run(3) || !run(769) || !run(maximum_count)) return false;
    const auto* scan_storage = workspace.scan_storage;
    const auto scan_capacity = workspace.scan_capacity_bytes;
    if (!run(maximum_count) || !run(7, true) || !run(0) || !run(7) ||
        workspace.scan_storage != scan_storage || workspace.scan_capacity_bytes != scan_capacity)
        return false;

    const auto* flags = workspace.flags;
    const auto* offsets = workspace.offsets;
    const auto* valid_count = workspace.valid_count;
    const auto capacity = workspace.capacity;
    GpuPosePipelineDeviceResult invalid;
    std::uint32_t valid = 99;
    if (cuda_compact_clash_survivors(input.transforms, input.clashes,
            std::numeric_limits<double>::quiet_NaN(), invalid.survivors, valid, &workspace) ||
        valid != 0 || invalid.survivors.count ||
        workspace.reserve(std::numeric_limits<std::size_t>::max()) ||
        workspace.flags != flags || workspace.offsets != offsets || workspace.valid_count != valid_count ||
        workspace.capacity != capacity || workspace.scan_storage != scan_storage ||
        workspace.scan_capacity_bytes != scan_capacity || !run(7))
        return false;
    workspace.clear();
    workspace.clear();
    return !workspace.flags && !workspace.offsets && !workspace.valid_count &&
           !workspace.scan_storage && workspace.capacity == 0 && workspace.scan_capacity_bytes == 0;
}

} // namespace


int main()
{
    if (!test_reused_clash_compaction()) {
        std::cerr << "Reused clash compaction changed results or workspace ownership\n";
        return 1;
    }

    // ========================================================
    // Ligand matching triangle
    // ========================================================

    const std::vector<GpuPoint3>
        ligand_matching_points = {

            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}
        };


    GpuPointArrayDevice
        device_ligand_matching_points;


    if (
        !cuda_upload_points(
            ligand_matching_points,
            device_ligand_matching_points
        )
    ) {
        std::cerr
            << "Ligand matching point upload failed\n";

        return 1;
    }


    // ========================================================
    // Protein target triangles
    //
    // 0,1,2 -> ligand triangle translated +1 X
    // 3,4,5 -> ligand triangle translated +2 X
    // ========================================================

    const std::vector<GpuPoint3>
        protein_target_points = {

            {1.0, 0.0, 0.0},
            {2.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},

            {2.0, 0.0, 0.0},
            {3.0, 0.0, 0.0},
            {2.0, 1.0, 0.0}
        };


    GpuPointArrayDevice
        device_protein_target_points;


    if (
        !cuda_upload_points(
            protein_target_points,
            device_protein_target_points
        )
    ) {
        std::cerr
            << "Protein target point upload failed\n";

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Alignment candidates
    // ========================================================

    GpuAlignmentCandidate
        alignment0;

    alignment0.ligand_indices[0] = 0;
    alignment0.ligand_indices[1] = 1;
    alignment0.ligand_indices[2] = 2;

    alignment0.protein_indices[0] = 0;
    alignment0.protein_indices[1] = 1;
    alignment0.protein_indices[2] = 2;


    GpuAlignmentCandidate
        alignment1;

    alignment1.ligand_indices[0] = 0;
    alignment1.ligand_indices[1] = 1;
    alignment1.ligand_indices[2] = 2;

    alignment1.protein_indices[0] = 3;
    alignment1.protein_indices[1] = 4;
    alignment1.protein_indices[2] = 5;


    const std::vector<GpuAlignmentCandidate>
        alignments = {

            alignment0,
            alignment1
        };


    // ========================================================
    // Ligand atoms
    //
    // For +1 X:
    //
    // atom 0 -> voxel (1,0,0), free
    // atom 1 -> voxel (2,0,0), occupied
    // atom 2 -> outside grid
    //
    // clash = 1/3
    //
    // For +2 X:
    //
    // atom 0 -> voxel (2,0,0), occupied
    // atom 1 -> voxel (3,0,0), occupied
    // atom 2 -> outside grid
    //
    // clash = 2/3
    // ========================================================

    const std::vector<GpuPoint3>
        ligand_atoms = {

            {0.2, 0.2, 0.2},
            {1.2, 0.2, 0.2},
            {10.0, 10.0, 10.0}
        };


    GpuPointArrayDevice
        device_ligand_atoms;


    if (
        !cuda_upload_points(
            ligand_atoms,
            device_ligand_atoms
        )
    ) {
        std::cerr
            << "Ligand atom upload failed\n";

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Protein grid
    // ========================================================

    GpuProteinGridHost
        host_grid;


    host_grid.xmin = 0.0;
    host_grid.ymin = 0.0;
    host_grid.zmin = 0.0;

    host_grid.voxel_size = 1.0;

    host_grid.nx = 4;
    host_grid.ny = 4;
    host_grid.nz = 4;

    host_grid.voxels.resize(
        4 * 4 * 4
    );


    constexpr std::uint16_t
        interaction_bit =
            std::uint16_t{1} << 0;


    /*
     * Free interaction voxel used by
     * surviving +1 X pose.
     */
    host_grid
        .voxels[
            grid_index(
                1,
                0,
                0
            )
        ]
        .interactions =
            interaction_bit;


    /*
     * Clash voxels.
     */
    host_grid
        .voxels[
            grid_index(
                2,
                0,
                0
            )
        ]
        .occupied = 1;


    host_grid
        .voxels[
            grid_index(
                3,
                0,
                0
            )
        ]
        .occupied = 1;


    GpuProteinGridDevice
        device_grid;


    if (
        !cuda_upload_protein_grid(
            host_grid,
            device_grid
        )
    ) {
        std::cerr
            << "Protein grid upload failed\n";

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Fast interaction areas
    //
    // For +1 X:
    //
    // area 0 -> (1,0,0): free + compatible
    // area 1 -> (2,0,0): occupied
    // area 2 -> outside
    // ========================================================

    const std::vector<GpuFastMatchingArea>
        host_fast_areas = {

            {
                {0.2, 0.2, 0.2},
                interaction_bit,
                {},
                1.5
            },

            {
                {1.2, 0.2, 0.2},
                interaction_bit,
                {},
                2.5
            },

            {
                {10.0, 10.0, 10.0},
                interaction_bit,
                {},
                3.5
            }
        };


    GpuFastMatchingAreaArrayDevice
        device_fast_areas;


    if (
        !cuda_upload_fast_matching_areas(
            host_fast_areas,
            device_fast_areas
        )
    ) {
        std::cerr
            << "Fast-area upload failed\n";

        cuda_free_protein_grid(
            device_grid
        );

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Pipeline
    //
    // threshold 0.5:
    //
    // +1 X -> 1/3 -> survives
    // +2 X -> 2/3 -> rejected
    // ========================================================

    GpuPosePipelineResult
        result;


    if (
        !cuda_run_pose_pipeline(
            device_ligand_matching_points,
            device_protein_target_points,
            device_ligand_atoms,
            device_fast_areas,
            device_grid,
            alignments,
            0.5,
            result
        )
    ) {
        std::cerr
            << "CUDA pose pipeline failed\n";

        cuda_free_fast_matching_areas(
            device_fast_areas
        );

        cuda_free_protein_grid(
            device_grid
        );

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // A retained workspace must also preserve the complete pose/Fast pipeline.
    {
        GpuClashCompactionWorkspace workspace;
        for (int repetition = 0; repetition < 2; ++repetition) {
            GpuPosePipelineResult reused_result;
            if (!cuda_run_pose_pipeline(
                    device_ligand_matching_points, device_protein_target_points,
                    device_ligand_atoms, device_fast_areas, device_grid, alignments,
                    0.5, reused_result, nullptr, &workspace) ||
                !same_results(result, reused_result)) {
                std::cerr << "Reused compaction scratch changed pose/Fast results\n";
                return 1;
            }
        }
    }

    // Pipeline-level checks
    // ========================================================

    if (
        result.input_alignment_count != 2 ||
        result.valid_transform_count != 2 ||
        result.clash_survivor_count != 1 ||
        result.poses.size() != 1
    ) {
        std::cerr
            << "Unexpected pipeline counts\n";

        return 1;
    }


    const auto& pose =
        result.poses[0];


    if (
        pose.alignment_index != 0
    ) {
        std::cerr
            << "Unexpected surviving alignment\n";

        return 1;
    }


    // ========================================================
    // Kabsch check
    // ========================================================

    if (
        !close(
            pose.transform.translation[0],
            1.0
        ) ||
        !close(
            pose.transform.translation[1],
            0.0
        ) ||
        !close(
            pose.transform.translation[2],
            0.0
        )
    ) {
        std::cerr
            << "Unexpected pipeline Kabsch transform\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline Kabsch stage passed\n";


    // ========================================================
    // Clash check
    // ========================================================

    if (
        pose.clash.occupied_atoms != 1 ||
        pose.clash.total_atoms != 3 ||
        !close(
            pose.clash.fraction,
            1.0 / 3.0
        )
    ) {
        std::cerr
            << "Unexpected pipeline clash result\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline clash filtering passed\n";


    // ========================================================
    // Fast scoring check
    // ========================================================

    if (
        pose.fast_score.total_areas != 3 ||
        pose.fast_score.areas_in_grid != 2 ||
        pose.fast_score
            .areas_on_occupied_voxel != 1 ||
        pose.fast_score.compatible_areas != 1 ||
        !close(
            pose.fast_score.weighted_score,
            1.5
        )
    ) {
        std::cerr
            << "Unexpected pipeline fast score\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline fast scoring passed\n";

    // Keep both alignments to compare survivor order as well as every score.
    GpuPosePipelineResult without_timings, with_timings;
    GpuPosePipelineTimings timings{99.0, 99.0, false, {}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            1.0, without_timings) ||
        !cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            1.0, with_timings, &timings) ||
        !same_results(without_timings, with_timings) || with_timings.poses.size() != 2 ||
        with_timings.poses[0].alignment_index != 0 ||
        with_timings.poses[1].alignment_index != 1 ||
        timings.collect_stages || !timings.stages.empty() ||
        !std::isfinite(timings.survivor_download_seconds) ||
        !std::isfinite(timings.fast_score_download_seconds) ||
        timings.survivor_download_seconds < 0.0 || timings.fast_score_download_seconds < 0.0) {
        std::cerr << "Download timers changed pipeline results or returned invalid times\n";
        return 1;
    }

    // Retaining the pool must preserve the original host-output path, including
    // Fast and stable alignment IDs, after temporary alignment inputs are freed.
    GpuAlignmentCandidateArrayDevice resident_alignments{};
    if (cudaMalloc(reinterpret_cast<void**>(&resident_alignments.alignments),
                   alignments.size() * sizeof(GpuAlignmentCandidate)) != cudaSuccess) {
        std::cerr << "Resident-pool alignment upload allocation failed\n";
        return 1;
    }
    resident_alignments.count = static_cast<std::uint32_t>(alignments.size());
    GpuClashCompactionWorkspace resident_workspace;
    GpuPosePipelineDeviceResult retained;
    GpuPosePipelineResult resident_result;
    GpuPosePipelineTimings resident_timings{99.0, 99.0, true, {}};
    const bool resident_ok = cudaMemcpy(resident_alignments.alignments, alignments.data(),
        alignments.size() * sizeof(GpuAlignmentCandidate), cudaMemcpyHostToDevice) == cudaSuccess &&
        cuda_run_pose_pipeline(device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, resident_alignments,
            1.0, resident_result, &resident_timings, &retained, &resident_workspace);
    cuda_free_alignment_candidates(resident_alignments);
    if (!resident_ok || !resident_result.poses.empty() ||
        retained.survivors.count != without_timings.clash_survivor_count ||
        retained.fast_scores.count != retained.survivors.count ||
        resident_timings.survivor_download_seconds != 0.0 ||
        resident_timings.fast_score_download_seconds != 0.0 ||
        !same_profile_stages(resident_timings.stages, 4, alignments.size())) {
        std::cerr << "Resident output lost counters, downloaded poses or reported invalid stages\n";
        return 1;
    }

    // A later search overwrites scratch, then releases it. Retained survivor
    // and Fast buffers must remain independent and readable in full.
    GpuPosePipelineResult next_result;
    if (!cuda_run_pose_pipeline(device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            0.5, next_result, nullptr, &resident_workspace) ||
        !same_results(result, next_result)) {
        std::cerr << "Reusing scratch changed the next search\n";
        return 1;
    }
    resident_workspace.clear();

    std::vector<GpuRigidTransform> retained_transforms;
    std::vector<GpuClashScore> retained_clash;
    std::vector<std::uint32_t> retained_alignment_ids;
    std::vector<GpuFastScore> retained_fast;
    if (!cuda_download_clash_survivors(retained.survivors, retained_transforms,
            retained_clash, retained_alignment_ids) ||
        !cuda_download_fast_scores(retained.fast_scores, retained_fast) ||
        retained_transforms.size() != retained.survivors.count ||
        retained_clash.size() != retained.survivors.count ||
        retained_alignment_ids.size() != retained.survivors.count ||
        retained_fast.size() != retained.survivors.count) {
        std::cerr << "Retained pool did not outlive temporary alignment inputs\n";
        return 1;
    }
    for (std::size_t i = 0; i < retained_transforms.size(); ++i)
        resident_result.poses.push_back({retained_alignment_ids[i], retained_transforms[i],
                                        retained_clash[i], retained_fast[i]});
    if (!same_results(without_timings, resident_result)) {
        std::cerr << "Resident and host-output paths produced different poses\n";
        return 1;
    }
    resident_timings = {99.0, 99.0, false, {}};
    if (!cuda_run_pose_pipeline(device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, GpuAlignmentCandidateArrayDevice{},
            1.0, resident_result, &resident_timings, &retained, &resident_workspace) ||
        resident_result.input_alignment_count || resident_result.valid_transform_count ||
        resident_result.clash_survivor_count || !resident_result.poses.empty() ||
        resident_timings.survivor_download_seconds != 0.0 ||
        resident_timings.fast_score_download_seconds != 0.0) {
        std::cerr << "Empty resident invocation did not clear previous outputs\n";
        return 1;
    }
    retained.clear(); retained.clear();
    if (retained.survivors.transforms || retained.survivors.clash_scores ||
        retained.survivors.alignment_indices || retained.survivors.count ||
        retained.fast_scores.scores || retained.fast_scores.count) {
        std::cerr << "Repeated resident clear must leave an empty owner\n";
        return 1;
    }

    // Structured collection must preserve every pose and its original order.
    // Calling through the host-vector overload also checks that its reset keeps
    // the collection request before forwarding to the device-input overload.
    GpuPosePipelineResult profiled_result;
    timings = {99.0, 99.0, true, {{"stale", 99.0, 99, "stale"}}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            1.0, profiled_result, &timings) ||
        !same_results(without_timings, profiled_result) || !timings.collect_stages ||
        !same_profile_stages(timings.stages, 7, alignments.size())) {
        std::cerr << "Stage profiling changed results or returned invalid stages\n";
        return 1;
    }

    const auto timers_are_zero = [&]() {
        return timings.survivor_download_seconds == 0.0 &&
               timings.fast_score_download_seconds == 0.0;
    };
    GpuPosePipelineResult empty_result;
    timings = {99.0, 99.0, false, {}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid,
            std::vector<GpuAlignmentCandidate>{}, 0.5, empty_result, &timings) ||
        !empty_result.poses.empty() || !timers_are_zero()) {
        std::cerr << "Empty alignment input must reset download timers\n";
        return 1;
    }

    timings = {99.0, 99.0, false, {}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            0.0, empty_result, &timings) ||
        empty_result.clash_survivor_count != 0 || !empty_result.poses.empty() ||
        !timers_are_zero()) {
        std::cerr << "No clash survivors must leave download timers at zero\n";
        return 1;
    }

    auto invalid_alignments = alignments;
    invalid_alignments[0].ligand_indices[0] = device_ligand_matching_points.count;
    timings = {99.0, 99.0, false, {}};
    if (cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, invalid_alignments,
            0.5, empty_result, &timings) || !timers_are_zero()) {
        std::cerr << "Host alignment validation failure must reset download timers\n";
        return 1;
    }

    timings = {99.0, 99.0, true, {{"stale", 99.0, 99, "stale"}}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid,
            std::vector<GpuAlignmentCandidate>{}, 0.5, empty_result, &timings) ||
        !timings.collect_stages || !timings.stages.empty() || !timers_are_zero()) {
        std::cerr << "Empty input must clear stages and retain collection request\n";
        return 1;
    }

    timings = {99.0, 99.0, true, {{"stale", 99.0, 99, "stale"}}};
    if (!cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, alignments,
            0.0, empty_result, &timings) ||
        !timings.collect_stages || !same_profile_stages(timings.stages, 3, alignments.size()) ||
        !empty_result.poses.empty() || !timers_are_zero()) {
        std::cerr << "No survivors must report only completed stages\n";
        return 1;
    }

    timings = {99.0, 99.0, true, {{"stale", 99.0, 99, "stale"}}};
    if (cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid, invalid_alignments,
            0.5, empty_result, &timings) ||
        !timings.collect_stages || !timings.stages.empty() || !timers_are_zero()) {
        std::cerr << "Host validation failure must not report completed stages\n";
        return 1;
    }

    timings = {99.0, 99.0, true, {{"stale", 99.0, 99, "stale"}}};
    if (cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid,
            GpuAlignmentCandidateArrayDevice{}, -1.0, empty_result, &timings) ||
        !timings.collect_stages || !timings.stages.empty() || !timers_are_zero()) {
        std::cerr << "Device validation failure must not report completed stages\n";
        return 1;
    }

    timings = {99.0, 99.0, false, {}};
    if (cuda_run_pose_pipeline(
            device_ligand_matching_points, device_protein_target_points,
            device_ligand_atoms, device_fast_areas, device_grid,
            GpuAlignmentCandidateArrayDevice{}, -1.0, empty_result, &timings) ||
        !timers_are_zero()) {
        std::cerr << "Device pipeline validation failure must reset download timers\n";
        return 1;
    }


    // ========================================================
    // Cleanup
    // ========================================================

    cuda_free_fast_matching_areas(
        device_fast_areas
    );

    cuda_free_protein_grid(
        device_grid
    );

    cuda_free_points(
        device_ligand_atoms
    );

    cuda_free_points(
        device_protein_target_points
    );

    cuda_free_points(
        device_ligand_matching_points
    );


    std::cout
        << "CUDA pose pipeline tests passed\n";


    return 0;
}
