#include <cuda_runtime_api.h>
#include <cstdlib>
#include <chrono>
#include "clash.hpp"
#include "cuda_backend.hpp"
#include "cuda_fast_scoring.hpp"
#include "cuda_geometry.hpp"
#include "cuda_grid.hpp"
#include "cuda_pose_pipeline.hpp"
#include "cuda_triplet_index.hpp"

#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "rigid_alignment.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"
#include "triplet_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


namespace
{

RigidTransform make_cpu_transform(
    const GpuAlignmentCandidate& alignment,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<PoseTarget>& pose_targets
)
{
    std::array<interaction::point3D, 3>
        source_points;

    std::array<interaction::point3D, 3>
        target_points;


    for (
        std::size_t i = 0;
        i < 3;
        ++i
    ) {
        const std::size_t ligand_index =
            alignment.ligand_indices[i];

        const std::size_t protein_index =
            alignment.protein_indices[i];


        if (
            ligand_index >=
                ligand_matching_areas.size()
        ) {
            throw std::runtime_error(
                "Alignment contains invalid ligand index"
            );
        }


        if (
            protein_index >=
                pose_targets.size()
        ) {
            throw std::runtime_error(
                "Alignment contains invalid protein index"
            );
        }


        source_points[i] =
            ligand_matching_areas[
                ligand_index
            ].centroid;

        target_points[i] =
            pose_targets[
                protein_index
            ].position;
    }


    RigidTransform transform;


    if (
        !rigid_transform_from_points(
            source_points,
            target_points,
            transform
        )
    ) {
        throw std::runtime_error(
            "CPU Kabsch failed for CUDA alignment candidate"
        );
    }


    return transform;
}


double gpu_rotation_determinant(
    const GpuRigidTransform& transform
)
{
    const double* r =
        transform.rotation;


    return
        r[0] * (
            r[4] * r[8] -
            r[5] * r[7]
        )
        -
        r[1] * (
            r[3] * r[8] -
            r[5] * r[6]
        )
        +
        r[2] * (
            r[3] * r[7] -
            r[4] * r[6]
        );
}


double gpu_orthogonality_error(
    const GpuRigidTransform& transform
)
{
    double maximum_error = 0.0;


    for (
        std::size_t row = 0;
        row < 3;
        ++row
    ) {
        for (
            std::size_t column = 0;
            column < 3;
            ++column
        ) {
            double value = 0.0;


            for (
                std::size_t k = 0;
                k < 3;
                ++k
            ) {
                value +=
                    transform.rotation[
                        k * 3 + row
                    ]
                    *
                    transform.rotation[
                        k * 3 + column
                    ];
            }


            const double expected =
                row == column
                    ? 1.0
                    : 0.0;


            maximum_error =
                std::max(
                    maximum_error,
                    std::abs(
                        value -
                        expected
                    )
                );
        }
    }


    return maximum_error;
}


double transformed_point_difference(
    const interaction::point3D& point,
    const RigidTransform& cpu_transform,
    const GpuRigidTransform& gpu_transform
)
{
    const auto cpu_point =
        transform_point(
            point,
            cpu_transform
        );


    const double x =
        gpu_transform.rotation[0] *
            static_cast<double>(point.x)
        +
        gpu_transform.rotation[1] *
            static_cast<double>(point.y)
        +
        gpu_transform.rotation[2] *
            static_cast<double>(point.z)
        +
        gpu_transform.translation[0];


    const double y =
        gpu_transform.rotation[3] *
            static_cast<double>(point.x)
        +
        gpu_transform.rotation[4] *
            static_cast<double>(point.y)
        +
        gpu_transform.rotation[5] *
            static_cast<double>(point.z)
        +
        gpu_transform.translation[1];


    const double z =
        gpu_transform.rotation[6] *
            static_cast<double>(point.x)
        +
        gpu_transform.rotation[7] *
            static_cast<double>(point.y)
        +
        gpu_transform.rotation[8] *
            static_cast<double>(point.z)
        +
        gpu_transform.translation[2];


    const double dx =
        static_cast<double>(cpu_point.x) -
        x;

    const double dy =
        static_cast<double>(cpu_point.y) -
        y;

    const double dz =
        static_cast<double>(cpu_point.z) -
        z;


    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


double transform_difference(
    const GpuRigidTransform& a,
    const GpuRigidTransform& b
)
{
    double maximum_error = 0.0;


    for (
        std::size_t i = 0;
        i < 9;
        ++i
    ) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    a.rotation[i] -
                    b.rotation[i]
                )
            );
    }


    for (
        std::size_t i = 0;
        i < 3;
        ++i
    ) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    a.translation[i] -
                    b.translation[i]
                )
            );
    }


    return maximum_error;
}

} // namespace


int main(
    int argc,
    char* argv[]
)
{
    // Optional stage-level timing for GPU profiling.
    // Normal correctness execution is unchanged.
    const bool enable_stage_timings =
        std::getenv("DOCKING_STAGE_TIMINGS") != nullptr;

    const auto timed_cuda_stage =
        [&](const char* label, auto&& operation) -> bool
        {
            if (!enable_stage_timings)
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

            std::cout
                << "CUDA_STAGE_MS,"
                << label
                << ","
                << milliseconds
                << '\n';

            return success && sync_status == cudaSuccess;
        };


    GpuFlatTripletIndexDevice
        device_triplet_index{};

    GpuPointArrayDevice
        device_ligand_matching_points{};

    GpuPointArrayDevice
        device_protein_target_points{};

    GpuPointArrayDevice
        device_ligand_atoms{};

    GpuProteinGridDevice
        device_grid{};

    GpuFastMatchingAreaArrayDevice
        device_fast_areas{};

    GpuTripletNeighborCountArrayDevice
        device_neighbor_counts{};

    GpuTripletNeighborOffsetArrayDevice
        device_neighbor_offsets{};

    GpuTripletCandidateArrayDevice
        device_triplet_candidates{};

    GpuAlignmentCountArrayDevice
        device_alignment_counts{};

    GpuAlignmentOffsetArrayDevice
        device_alignment_offsets{};

    GpuAlignmentCandidateArrayDevice
        device_alignments{};


    const auto cleanup =
        [&]()
        {
            cuda_free_alignment_candidates(
                device_alignments
            );

            cuda_free_alignment_offsets(
                device_alignment_offsets
            );

            cuda_free_alignment_counts(
                device_alignment_counts
            );

            cuda_free_triplet_candidates(
                device_triplet_candidates
            );

            cuda_free_triplet_neighbor_offsets(
                device_neighbor_offsets
            );

            cuda_free_triplet_neighbor_counts(
                device_neighbor_counts
            );

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

            cuda_free_triplet_index(
                device_triplet_index
            );
        };


    try {
        const RunConfig config =
            parse_run_config(
                argc,
                argv
            );


        // ====================================================
        // CPU target preparation
        // ====================================================

        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );


        const auto protein_areas =
            get_protein_areas(
                *protein
            );


        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );


        auto protein_grid =
            build_empty_protein_grid(
                *pocket,
                config.grid_voxel_size,
                config.grid_margin
            );


        mark_protein_occupancy(
            *protein,
            protein_grid
        );


        mark_interaction_regions(
            protein_matching_areas,
            protein_grid
        );


        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas,
                *pocket,
                config.pocket_centroid_distance
            );


        const auto pose_targets =
            generate_pose_targets(
                pocket_matching_areas,
                protein_grid,
                *protein,
                *pocket
            );


        // ====================================================
        // CPU ligand preparation
        // ====================================================

        auto ligand =
            load_ligand(
                config.ligand_path.string()
            );


        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        const double ligand_diameter =
            compute_ligand_diameter(
                ligand_areas
            );


        const auto ligand_triplets =
            generate_triplets(
                ligand_matching_areas,
                ligand_diameter
            );


        // ====================================================
        // Protein triplet index
        // ====================================================

        const auto protein_triplet_hash =
            build_protein_triplet_hash_table(
                pose_targets,
                ligand_diameter,
                config.triplet_bin_size
            );


        const auto flat_index =
            flatten_protein_triplet_index(
                protein_triplet_hash
            );


        if (
            !validate_flat_triplet_index(
                protein_triplet_hash,
                flat_index
            )
        ) {
            throw std::runtime_error(
                "CPU flat triplet index validation failed"
            );
        }


        const auto gpu_host_index =
            make_gpu_flat_triplet_index(
                flat_index
            );


        if (
            !validate_gpu_flat_triplet_index(
                flat_index,
                gpu_host_index
            )
        ) {
            throw std::runtime_error(
                "GPU flat triplet index conversion failed"
            );
        }


        if (
            !cuda_upload_triplet_index(
                gpu_host_index,
                device_triplet_index
            )
        ) {
            throw std::runtime_error(
                "CUDA triplet index upload failed"
            );
        }


        // ====================================================
        // Ligand triplet queries
        // ====================================================

        std::vector<GpuTripletKey>
            queries;

        std::vector<GpuLigandTripletOrders>
            ligand_orders;


        make_gpu_ligand_triplet_queries(
            ligand_triplets,
            config.triplet_bin_size,
            queries,
            ligand_orders
        );


        if (
            queries.size() !=
            ligand_orders.size()
        ) {
            throw std::runtime_error(
                "Ligand query/order size mismatch"
            );
        }


        // ====================================================
        // CUDA triplet lookup
        // ====================================================

        if (
            !timed_cuda_stage("neighbor_count", [&]() { return cuda_count_triplet_neighbor_records_device(
                device_triplet_index,
                queries,
                device_neighbor_counts
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device candidate counting failed"
            );
        }


        // Temporary download for CPU/GPU numerical
        // correctness validation.
        std::vector<std::uint32_t>
            candidate_counts;


        if (
            !cuda_download_triplet_neighbor_counts(
                device_neighbor_counts,
                candidate_counts
            )
        ) {
            throw std::runtime_error(
                "CUDA candidate count download failed"
            );
        }


        if (
            device_neighbor_counts.count != queries.size()
            ||
            candidate_counts.size() != queries.size()
        ) {
            throw std::runtime_error(
                "CUDA candidate count size mismatch"
            );
        }


        if (
            !timed_cuda_stage("neighbor_scan", [&]() { return cuda_exclusive_triplet_neighbor_prefix_sum_device(
                device_neighbor_counts,
                device_neighbor_offsets
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device candidate prefix sum failed"
            );
        }


        // Temporary host copy for candidate expansion
        // and CPU/GPU correctness validation.
        std::vector<std::uint32_t>
            candidate_offsets;


        if (
            !cuda_download_triplet_neighbor_offsets(
                device_neighbor_offsets,
                candidate_offsets
            )
        ) {
            throw std::runtime_error(
                "CUDA candidate offset download failed"
            );
        }


        if (
            device_neighbor_offsets.count !=
                candidate_counts.size() + 1
            ||
            candidate_offsets.size() !=
                candidate_counts.size() + 1
        ) {
            throw std::runtime_error(
                "CUDA candidate offset size mismatch"
            );
        }


        // Independent CPU validation of every GPU offset.
        std::uint64_t expected_candidate_offset = 0;

        for (
            std::size_t i = 0;
            i < candidate_counts.size();
            ++i
        ) {
            if (
                candidate_offsets[i] !=
                expected_candidate_offset
            ) {
                throw std::runtime_error(
                    "CPU/GPU candidate prefix mismatch"
                );
            }

            expected_candidate_offset +=
                candidate_counts[i];
        }


        if (
            candidate_offsets.back() !=
            expected_candidate_offset
        ) {
            throw std::runtime_error(
                "CPU/GPU final candidate offset mismatch"
            );
        }


        if (
            !timed_cuda_stage("candidate_expansion", [&]() { return cuda_expand_triplet_neighbor_records_device(
                device_triplet_index,
                queries,
                device_neighbor_counts,
                device_neighbor_offsets,
                device_triplet_candidates
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device triplet candidate expansion failed"
            );
        }


        // Temporary host copy needed by alignment counting
        // and expansion, and by CPU/GPU validation.
        std::vector<GpuTripletCandidate>
            triplet_candidates;


        if (
            !cuda_download_triplet_candidates(
                device_triplet_candidates,
                triplet_candidates
            )
        ) {
            throw std::runtime_error(
                "CUDA triplet candidate download failed"
            );
        }


        if (
            candidate_offsets.empty()
            ||
            device_triplet_candidates.count !=
                candidate_offsets.back()
            ||
            triplet_candidates.size() !=
                device_triplet_candidates.count
        ) {
            throw std::runtime_error(
                "CUDA triplet candidate count mismatch"
            );
        }


        // ====================================================
        // Equivalent-order expansion
        // ====================================================

        if (
            !timed_cuda_stage("alignment_count", [&]() { return cuda_count_alignment_candidates_device(
                device_triplet_candidates,
                ligand_orders,
                device_alignment_counts
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device alignment counting failed"
            );
        }


        // Temporary download for CPU/GPU validation
        // and the existing alignment prefix scan.
        std::vector<std::uint32_t>
            alignment_counts;


        if (
            !cuda_download_alignment_counts(
                device_alignment_counts,
                alignment_counts
            )
        ) {
            throw std::runtime_error(
                "CUDA alignment count download failed"
            );
        }


        if (
            device_alignment_counts.count !=
                triplet_candidates.size()
            ||
            alignment_counts.size() !=
                triplet_candidates.size()
        ) {
            throw std::runtime_error(
                "CUDA alignment count size mismatch"
            );
        }


        if (
            !timed_cuda_stage("alignment_scan", [&]() { return cuda_exclusive_prefix_sum_device(
                device_alignment_counts,
                device_alignment_offsets
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device alignment prefix sum failed"
            );
        }


        // Temporary download for alignment expansion
        // and numerical correctness validation.
        std::vector<std::uint32_t>
            alignment_offsets;


        if (
            !cuda_download_alignment_offsets(
                device_alignment_offsets,
                alignment_offsets
            )
        ) {
            throw std::runtime_error(
                "CUDA alignment offset download failed"
            );
        }


        if (
            device_alignment_offsets.count !=
                alignment_counts.size() + 1
            ||
            alignment_offsets.size() !=
                alignment_counts.size() + 1
        ) {
            throw std::runtime_error(
                "CUDA alignment offset size mismatch"
            );
        }


        // Validate every GPU prefix offset against
        // an independent CPU cumulative sum.
        std::uint64_t expected_offset = 0;

        for (
            std::size_t i = 0;
            i < alignment_counts.size();
            ++i
        ) {
            if (
                alignment_offsets[i] !=
                expected_offset
            ) {
                throw std::runtime_error(
                    "CPU/GPU alignment prefix mismatch"
                );
            }

            expected_offset +=
                alignment_counts[i];
        }


        if (
            alignment_offsets.back() !=
            expected_offset
        ) {
            throw std::runtime_error(
                "CPU/GPU final alignment offset mismatch"
            );
        }





        if (
            !timed_cuda_stage("alignment_expansion", [&]() { return cuda_expand_alignment_candidates_device(
                device_triplet_candidates,
                ligand_orders,
                device_alignment_offsets,
                device_alignments
            ); })
        ) {
            throw std::runtime_error(
                "CUDA device alignment expansion failed"
            );
        }


        /*
        * Host copy used only by the CPU/GPU validation code below.
        *
        * The CUDA pose pipeline itself consumes device_alignments
        * directly and therefore does not upload these alignments
        * back to the GPU.
        */
        std::vector<GpuAlignmentCandidate>
            alignments;


        if (
            !cuda_download_alignment_candidates(
                device_alignments,
                alignments
            )
        ) {
            throw std::runtime_error(
                "CUDA alignment download failed"
            );
        }


        // ====================================================
        // Geometry upload
        // ====================================================

        const auto ligand_matching_points =
            make_gpu_ligand_matching_points(
                ligand_matching_areas
            );


        const auto protein_target_points =
            make_gpu_pose_target_points(
                pose_targets
            );


        const auto ligand_atom_points =
            make_gpu_ligand_atom_points(
                *ligand
            );


        if (
            !cuda_upload_points(
                ligand_matching_points,
                device_ligand_matching_points
            )
        ) {
            throw std::runtime_error(
                "Ligand matching-point upload failed"
            );
        }


        if (
            !cuda_upload_points(
                protein_target_points,
                device_protein_target_points
            )
        ) {
            throw std::runtime_error(
                "Protein target-point upload failed"
            );
        }


        if (
            !cuda_upload_points(
                ligand_atom_points,
                device_ligand_atoms
            )
        ) {
            throw std::runtime_error(
                "Ligand atom upload failed"
            );
        }


        // ====================================================
        // Protein grid upload
        // ====================================================

        const auto gpu_host_grid =
            make_gpu_protein_grid(
                protein_grid
            );


        if (
            !validate_gpu_protein_grid(
                protein_grid,
                gpu_host_grid
            )
        ) {
            throw std::runtime_error(
                "GPU protein grid conversion failed"
            );
        }


        if (
            !cuda_upload_protein_grid(
                gpu_host_grid,
                device_grid
            )
        ) {
            throw std::runtime_error(
                "CUDA protein grid upload failed"
            );
        }


        // ====================================================
        // Fast scoring areas
        // ====================================================

        const FastScoreWeights
            fast_weights{};


        const auto fast_areas =
            make_gpu_fast_matching_areas(
                ligand_matching_areas,
                fast_weights
            );


        if (
            !cuda_upload_fast_matching_areas(
                fast_areas,
                device_fast_areas
            )
        ) {
            throw std::runtime_error(
                "CUDA fast-area upload failed"
            );
        }


        // ====================================================
        // Direct CUDA Kabsch + clash
        //
        // We keep these results so every alignment can be
        // compared directly against the CPU implementation.
        // ====================================================

        std::vector<GpuRigidTransform>
            gpu_transforms;


        if (
            !cuda_compute_kabsch_transforms(
                device_ligand_matching_points,
                device_protein_target_points,
                alignments,
                gpu_transforms
            )
        ) {
            throw std::runtime_error(
                "CUDA Kabsch stage failed"
            );
        }


        if (
            gpu_transforms.size() !=
            alignments.size()
        ) {
            throw std::runtime_error(
                "CUDA Kabsch result size mismatch"
            );
        }


        std::vector<GpuClashScore>
            gpu_clash_scores;


        if (
            !cuda_score_grid_clash(
                device_ligand_atoms,
                device_grid,
                gpu_transforms,
                gpu_clash_scores
            )
        ) {
            throw std::runtime_error(
                "CUDA clash stage failed"
            );
        }


        if (
            gpu_clash_scores.size() !=
            alignments.size()
        ) {
            throw std::runtime_error(
                "CUDA clash result size mismatch"
            );
        }


        // ====================================================
        // CPU vs GPU validation: Kabsch + clash
        // ====================================================

        double max_rotation_error = 0.0;
        double max_translation_error = 0.0;

        double max_transformed_point_error =
            0.0;

        double max_determinant_error = 0.0;

        double max_orthogonality_error = 0.0;

        double max_clash_fraction_error = 0.0;


        std::size_t
            transform_validity_mismatches = 0;

        std::size_t
            clash_validity_mismatches = 0;

        std::size_t
            clash_count_mismatches = 0;

        std::size_t
            survivor_decision_mismatches = 0;

        std::size_t
            cpu_survivors = 0;

        std::size_t
            gpu_survivors = 0;


        for (
            std::size_t alignment_index = 0;
            alignment_index <
                alignments.size();
            ++alignment_index
        ) {
            const auto& alignment =
                alignments[
                    alignment_index
                ];


            const RigidTransform cpu_transform =
                make_cpu_transform(
                    alignment,
                    ligand_matching_areas,
                    pose_targets
                );


            const auto& gpu_transform =
                gpu_transforms[
                    alignment_index
                ];


            if (!gpu_transform.valid) {
                ++transform_validity_mismatches;
                continue;
            }


            for (
                std::size_t i = 0;
                i < 9;
                ++i
            ) {
                max_rotation_error =
                    std::max(
                        max_rotation_error,
                        std::abs(
                            cpu_transform.rotation[i] -
                            gpu_transform.rotation[i]
                        )
                    );
            }


            for (
                std::size_t i = 0;
                i < 3;
                ++i
            ) {
                max_translation_error =
                    std::max(
                        max_translation_error,
                        std::abs(
                            cpu_transform.translation[i] -
                            gpu_transform.translation[i]
                        )
                    );
            }


            for (
                std::size_t i = 0;
                i < 3;
                ++i
            ) {
                const auto& point =
                    ligand_matching_areas[
                        alignment.ligand_indices[i]
                    ].centroid;


                max_transformed_point_error =
                    std::max(
                        max_transformed_point_error,
                        transformed_point_difference(
                            point,
                            cpu_transform,
                            gpu_transform
                        )
                    );
            }


            max_determinant_error =
                std::max(
                    max_determinant_error,
                    std::abs(
                        gpu_rotation_determinant(
                            gpu_transform
                        ) -
                        1.0
                    )
                );


            max_orthogonality_error =
                std::max(
                    max_orthogonality_error,
                    gpu_orthogonality_error(
                        gpu_transform
                    )
                );


            const auto cpu_clash =
                score_fast_grid_clash(
                    *ligand,
                    cpu_transform,
                    protein_grid
                );


            const auto& gpu_clash =
                gpu_clash_scores[
                    alignment_index
                ];


            if (!gpu_clash.valid) {
                ++clash_validity_mismatches;
                continue;
            }


            if (
                cpu_clash.occupied_atoms !=
                    gpu_clash.occupied_atoms
                ||
                cpu_clash.total_atoms !=
                    gpu_clash.total_atoms
            ) {
                ++clash_count_mismatches;
            }


            max_clash_fraction_error =
                std::max(
                    max_clash_fraction_error,
                    std::abs(
                        cpu_clash.fraction -
                        gpu_clash.fraction
                    )
                );


            const bool cpu_survives =
                cpu_clash.fraction <=
                config.operational_clash_threshold;


            const bool gpu_survives =
                gpu_clash.fraction <=
                config.operational_clash_threshold;


            if (cpu_survives)
                ++cpu_survivors;


            if (gpu_survives)
                ++gpu_survivors;


            if (
                cpu_survives !=
                gpu_survives
            ) {
                ++survivor_decision_mismatches;
            }
        }


        // ====================================================
        // Full CUDA pose pipeline
        // ====================================================

        GpuPosePipelineResult result;


        if (
            !timed_cuda_stage("pose_pipeline", [&]() { return cuda_run_pose_pipeline(
                device_ligand_matching_points,
                device_protein_target_points,
                device_ligand_atoms,
                device_fast_areas,
                device_grid,
                device_alignments,
                config.operational_clash_threshold,
                result
            ); })
        ) {
            throw std::runtime_error(
                "CUDA real pose pipeline failed"
            );
        }


        // ====================================================
        // CPU vs GPU validation: surviving fast scores
        // ====================================================

        std::size_t
            fast_score_structure_mismatches = 0;

        std::size_t
            pipeline_clash_mismatches = 0;

        std::size_t
            invalid_pipeline_alignment_indices = 0;


        double max_fast_score_error = 0.0;

        double max_pipeline_transform_repeat_error =
            0.0;

        double max_pipeline_clash_fraction_repeat_error =
            0.0;


        for (const auto& pose :
             result.poses) {

            const std::size_t alignment_index =
                pose.alignment_index;


            if (
                alignment_index >=
                    alignments.size()
            ) {
                ++invalid_pipeline_alignment_indices;
                continue;
            }


            const auto& alignment =
                alignments[
                    alignment_index
                ];


            const RigidTransform cpu_transform =
                make_cpu_transform(
                    alignment,
                    ligand_matching_areas,
                    pose_targets
                );


            const auto transformed_matching =
                transform_matching_areas(
                    ligand_matching_areas,
                    cpu_transform
                );


            const auto cpu_fast_score =
                score_matching_areas_weighted(
                    transformed_matching,
                    protein_grid,
                    fast_weights
                );


            if (
                !pose.fast_score.valid
                ||
                cpu_fast_score.total_areas !=
                    pose.fast_score.total_areas
                ||
                cpu_fast_score.areas_in_grid !=
                    pose.fast_score.areas_in_grid
                ||
                cpu_fast_score
                        .areas_on_occupied_voxel !=
                    pose.fast_score
                        .areas_on_occupied_voxel
                ||
                cpu_fast_score.compatible_areas !=
                    pose.fast_score.compatible_areas
            ) {
                ++fast_score_structure_mismatches;
            }


            max_fast_score_error =
                std::max(
                    max_fast_score_error,
                    std::abs(
                        cpu_fast_score.weighted_score -
                        pose.fast_score.weighted_score
                    )
                );


            const auto& direct_transform =
                gpu_transforms[
                    alignment_index
                ];


            max_pipeline_transform_repeat_error =
                std::max(
                    max_pipeline_transform_repeat_error,
                    transform_difference(
                        pose.transform,
                        direct_transform
                    )
                );


            const auto& direct_clash =
                gpu_clash_scores[
                    alignment_index
                ];


            if (
                pose.clash.valid !=
                    direct_clash.valid
                ||
                pose.clash.occupied_atoms !=
                    direct_clash.occupied_atoms
                ||
                pose.clash.total_atoms !=
                    direct_clash.total_atoms
            ) {
                ++pipeline_clash_mismatches;
            }


            max_pipeline_clash_fraction_repeat_error =
                std::max(
                    max_pipeline_clash_fraction_repeat_error,
                    std::abs(
                        pose.clash.fraction -
                        direct_clash.fraction
                    )
                );
        }


        // ====================================================
        // Report
        // ====================================================

        std::cout
            << std::setprecision(17);


        std::cout
            << "\n=== Real CUDA pipeline ===\n"

            << "Complex: "
            << config.complex_id
            << '\n'

            << "Pose targets: "
            << pose_targets.size()
            << '\n'

            << "Ligand matching areas: "
            << ligand_matching_areas.size()
            << '\n'

            << "Ligand triplets: "
            << ligand_triplets.size()
            << '\n'

            << "Protein triplet records: "
            << flat_index.records.size()
            << '\n'

            << "Candidate matches: "
            << triplet_candidates.size()
            << '\n'

            << "Alignment candidates: "
            << alignments.size()
            << '\n'

            << "Valid transforms: "
            << result.valid_transform_count
            << '\n'

            << "Clash survivors: "
            << result.clash_survivor_count
            << '\n'

            << "Pipeline poses: "
            << result.poses.size()
            << '\n';


        std::cout
            << "\n=== CPU vs GPU Kabsch ===\n"

            << "Validity mismatches: "
            << transform_validity_mismatches
            << '\n'

            << "Max rotation element error: "
            << max_rotation_error
            << '\n'

            << "Max translation error: "
            << max_translation_error
            << '\n'

            << "Max transformed source-point error: "
            << max_transformed_point_error
            << '\n'

            << "Max |det(R) - 1|: "
            << max_determinant_error
            << '\n'

            << "Max orthogonality error: "
            << max_orthogonality_error
            << '\n';


        std::cout
            << "\n=== CPU vs GPU clash ===\n"

            << "GPU validity mismatches: "
            << clash_validity_mismatches
            << '\n'

            << "Occupied/total atom mismatches: "
            << clash_count_mismatches
            << '\n'

            << "Max clash-fraction error: "
            << max_clash_fraction_error
            << '\n'

            << "CPU survivors: "
            << cpu_survivors
            << '\n'

            << "GPU survivors: "
            << gpu_survivors
            << '\n'

            << "Survivor decision mismatches: "
            << survivor_decision_mismatches
            << '\n';


        std::cout
            << "\n=== CPU vs GPU fast scoring ===\n"

            << "Structural score mismatches: "
            << fast_score_structure_mismatches
            << '\n'

            << "Max weighted-score error: "
            << max_fast_score_error
            << '\n';


        std::cout
            << "\n=== CUDA pipeline internal consistency ===\n"

            << "Invalid alignment indices: "
            << invalid_pipeline_alignment_indices
            << '\n'

            << "Max repeated-transform error: "
            << max_pipeline_transform_repeat_error
            << '\n'

            << "Clash metadata mismatches: "
            << pipeline_clash_mismatches
            << '\n'

            << "Max repeated clash-fraction error: "
            << max_pipeline_clash_fraction_repeat_error
            << '\n';


        // These tolerances are intentionally conservative.
        // First goal: detect meaningful numerical divergence,
        // not demand bitwise CPU/GPU equality.

        constexpr double
            transform_tolerance = 1e-6;

        // CPU transform_point() stores the transformed
        // coordinate back into interaction::coordinate_type,
        // which is float. A few micro-Angstroms are therefore
        // expected when comparing against the CUDA double path.
        constexpr double
            transformed_point_tolerance = 1e-5;

        constexpr double
            rotation_quality_tolerance = 1e-9;

        constexpr double
            fast_score_tolerance = 1e-9;


        const bool validation_passed =
            transform_validity_mismatches == 0
            &&
            clash_validity_mismatches == 0
            &&
            clash_count_mismatches == 0
            &&
            survivor_decision_mismatches == 0
            &&
            cpu_survivors ==
                result.clash_survivor_count
            &&
            gpu_survivors ==
                result.clash_survivor_count
            &&
            fast_score_structure_mismatches == 0
            &&
            invalid_pipeline_alignment_indices == 0
            &&
            pipeline_clash_mismatches == 0
            &&
            max_rotation_error <=
                transform_tolerance
            &&
            max_translation_error <=
                transform_tolerance
            &&
            max_transformed_point_error <=
                transformed_point_tolerance
            &&
            max_determinant_error <=
                rotation_quality_tolerance
            &&
            max_orthogonality_error <=
                rotation_quality_tolerance
            &&
            max_fast_score_error <=
                fast_score_tolerance
            &&
            max_pipeline_transform_repeat_error <=
                transform_tolerance
            &&
            max_pipeline_clash_fraction_repeat_error <=
                fast_score_tolerance;


        cleanup();


        if (!validation_passed) {
            std::cerr
                << "\nCPU/GPU numerical validation FAILED\n";

            return 1;
        }


        std::cout
            << "\nCPU/GPU numerical validation PASSED\n";


        return 0;
    }
    catch (const std::exception& error) {

        cleanup();

        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}
