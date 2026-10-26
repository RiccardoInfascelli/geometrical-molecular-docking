#include "serial_docking.hpp"

#include "clash.hpp"
#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "rigid_alignment.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "triplet_generation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>


namespace
{

struct SerialCandidatePose
{
    RigidTransform transform{};

    std::size_t compatible_areas{};
    std::size_t occupied_atoms{};

    double fast_score{};

    WeightedInteractionScore weighted_interaction_score{};
};

} // namespace


SerialDockingResult run_serial_docking(
    const RunConfig& config,
    std::size_t top_n)
{
    using Clock = std::chrono::steady_clock;

    SerialDockingResult result;

    const auto total_start =
        Clock::now();


    // ========================================================
    // OFFLINE TARGET PREPARATION
    // ========================================================

    const auto offline_start =
        Clock::now();


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


    const auto pocket_protein_areas =
        filter_protein_areas_in_pocket(
            protein_areas,
            *pocket
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


    const auto offline_end =
        Clock::now();


    result.timings
        .offline_target_preparation_seconds =
            std::chrono::duration<double>(
                offline_end -
                offline_start
            ).count();


    // ========================================================
    // ONLINE
    // ========================================================

    const auto online_start =
        Clock::now();


    // ========================================================
    // LIGAND PREPARATION
    // ========================================================

    const auto ligand_start =
        Clock::now();


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


    const auto ligand_end =
        Clock::now();


    result.timings
        .ligand_preparation_seconds =
            std::chrono::duration<double>(
                ligand_end -
                ligand_start
            ).count();


    // ========================================================
    // PROTEIN TRIPLET INDEX
    // ========================================================

    const auto index_start =
        Clock::now();


    const auto protein_triplet_hash =
        build_protein_triplet_hash_table(
            pose_targets,
            ligand_diameter,
            config.triplet_bin_size
        );


    const auto index_end =
        Clock::now();


    result.timings
        .protein_triplet_index_seconds =
            std::chrono::duration<double>(
                index_end -
                index_start
            ).count();


    // Enumerate hash buckets and equivalent point orders without oracle RMSD
    // or threshold sweeps. Grid support is retained for the legacy Fast tie-break.

    std::vector<SerialCandidatePose>
        surviving_poses;


    double grid_clash_seconds = 0.0;


    const auto generation_start =
        Clock::now();


    for (const auto& ligand_triplet :
         ligand_triplets) {

        const auto canonical =
            canonicalize_triplet(
                ligand_triplet,
                config.triplet_bin_size
            );


        const auto buckets =
            find_neighbor_triplet_buckets(
                protein_triplet_hash,
                canonical
            );


        if (buckets.empty())
            continue;


        for (const auto* bucket :
             buckets) {

            for (const auto& protein_record :
                 *bucket) {

                const std::array<
                    interaction::point3D,
                    3
                > target_points{

                    pose_targets[
                        protein_record.indices[0]
                    ].position,

                    pose_targets[
                        protein_record.indices[1]
                    ].position,

                    pose_targets[
                        protein_record.indices[2]
                    ].position
                };


                for (
                    std::size_t order_index = 0;
                    order_index <
                        canonical.equivalent_order_count;
                    ++order_index
                ) {

                    const auto& ligand_order =
                        canonical.equivalent_orders[
                            order_index
                        ];


                    const std::array<
                        interaction::point3D,
                        3
                    > source_points{

                        ligand_matching_areas[
                            ligand_order[0]
                        ].centroid,

                        ligand_matching_areas[
                            ligand_order[1]
                        ].centroid,

                        ligand_matching_areas[
                            ligand_order[2]
                        ].centroid
                    };


                    RigidTransform transform;


                    if (
                        !rigid_transform_from_points(
                            source_points,
                            target_points,
                            transform
                        )
                    ) {
                        continue;
                    }


                    // Support does not reject a pose here; the occupied-atom
                    // fraction below is the operational survival criterion.

                    const auto grid_clash_start =
                        Clock::now();


                    const auto transformed_matching =
                        transform_matching_areas(
                            ligand_matching_areas,
                            transform
                        );


                    const auto grid_score =
                        score_ligand_on_grid(
                            transformed_matching,
                            protein_grid
                        );


                    const auto clash =
                        score_fast_grid_clash(
                            *ligand,
                            transform,
                            protein_grid
                        );


                    const auto grid_clash_end =
                        Clock::now();


                    grid_clash_seconds +=
                        std::chrono::duration<double>(
                            grid_clash_end -
                            grid_clash_start
                        ).count();


                    // Only the operational clash threshold.
                    if (
                        clash.fraction >
                        config.operational_clash_threshold
                    ) {
                        continue;
                    }


                    SerialCandidatePose pose;

                    pose.transform =
                        transform;

                    pose.compatible_areas =
                        grid_score.compatible_areas;

                    pose.occupied_atoms =
                        clash.occupied_atoms;


                    surviving_poses.push_back(
                        pose
                    );
                }
            }
        }
    }


    const auto generation_end =
        Clock::now();


    const double generation_seconds =
        std::chrono::duration<double>(
            generation_end -
            generation_start
        ).count();


    result.timings.grid_clash_seconds =
        grid_clash_seconds;


    result.timings.matching_kabsch_seconds =
        std::max(
            0.0,
            generation_seconds -
                grid_clash_seconds
        );


    // ========================================================
    // FAST SCORING
    // ========================================================

    const auto fast_start =
        Clock::now();


    for (auto& pose :
         surviving_poses) {

        const auto transformed_matching =
            transform_matching_areas(
                ligand_matching_areas,
                pose.transform
            );


        const auto score =
            score_matching_areas_weighted(
                transformed_matching,
                protein_grid
            );


        pose.fast_score =
            score.weighted_score;
    }


    std::sort(
        surviving_poses.begin(),
        surviving_poses.end(),
        [](
            const SerialCandidatePose& a,
            const SerialCandidatePose& b)
        {
            if (a.fast_score != b.fast_score) {
                return
                    a.fast_score >
                    b.fast_score;
            }

            if (
                a.compatible_areas !=
                b.compatible_areas
            ) {
                return
                    a.compatible_areas >
                    b.compatible_areas;
            }

            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // Only the best FAST poses reach WeightedInteraction chemistry.
    if (
        surviving_poses.size() >
        config.fast_prefilter_top_n
    ) {
        surviving_poses.resize(
            config.fast_prefilter_top_n
        );
    }


    const auto fast_end =
        Clock::now();


    result.timings.fast_scoring_seconds =
        std::chrono::duration<double>(
            fast_end -
            fast_start
        ).count();


    // ========================================================
    // WEIGHTED INTERACTION SCORING
    // ========================================================

    const auto weighted_interaction_start =
        Clock::now();


    for (auto& pose :
         surviving_poses) {

        const auto transformed_areas =
            transform_interaction_areas(
                ligand_areas,
                pose.transform
            );


        pose.weighted_interaction_score =
            score_weighted_interactions(
                pocket_protein_areas,
                transformed_areas
            );
    }


    // ========================================================
    // FINAL RANKING
    // ========================================================

    std::sort(
        surviving_poses.begin(),
        surviving_poses.end(),
        [](
            const SerialCandidatePose& a,
            const SerialCandidatePose& b)
        {
            if (
                a.weighted_interaction_score.weighted_score !=
                b.weighted_interaction_score.weighted_score
            ) {
                return
                    a.weighted_interaction_score.weighted_score >
                    b.weighted_interaction_score.weighted_score;
            }

            if (
                a.weighted_interaction_score.satisfied_ligand_areas !=
                b.weighted_interaction_score.satisfied_ligand_areas
            ) {
                return
                    a.weighted_interaction_score.satisfied_ligand_areas >
                    b.weighted_interaction_score.satisfied_ligand_areas;
            }

            if (
                a.weighted_interaction_score.total_interactions !=
                b.weighted_interaction_score.total_interactions
            ) {
                return
                    a.weighted_interaction_score.total_interactions >
                    b.weighted_interaction_score.total_interactions;
            }

            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    const auto weighted_interaction_end =
        Clock::now();


    result.timings.weighted_interaction_scoring_seconds =
        std::chrono::duration<double>(
            weighted_interaction_end -
            weighted_interaction_start
        ).count();


    // ========================================================
    // FINAL TOP-N POSES
    // ========================================================

    const std::size_t final_count =
        std::min(
            top_n,
            surviving_poses.size()
        );


    result.top_poses.reserve(
        final_count
    );


    for (
        std::size_t i = 0;
        i < final_count;
        ++i
    ) {
        const auto& pose =
            surviving_poses[i];


        DockedPose output_pose;

        output_pose.transform =
            pose.transform;

        output_pose.fast_score =
            pose.fast_score;

        output_pose.weighted_interaction_score =
            pose.weighted_interaction_score;

        output_pose.occupied_atoms =
            pose.occupied_atoms;


        result.top_poses.push_back(
            output_pose
        );
    }


    const auto online_end =
        Clock::now();

    const auto total_end =
        Clock::now();


    result.timings.online_total_seconds =
        std::chrono::duration<double>(
            online_end -
            online_start
        ).count();


    result.timings.total_seconds =
        std::chrono::duration<double>(
            total_end -
            total_start
        ).count();


    return result;
}
