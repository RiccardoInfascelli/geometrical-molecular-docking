#include "pipeline.hpp"

#include "diagnostics.hpp"
#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "pose_generation.hpp"
#include "pose_ranking.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "protein_area_crop.hpp"
#include "triplet_generation.hpp"

#include <interaction/find_interaction.hpp>

#include <cstddef>
#include <unordered_set>
#include <chrono>
#include <stdexcept>


RunResult run_pipeline(
    const RunConfig& config)
{
    RunResult result;

    result.complex_id =
        config.complex_id;


    // Target-only work ends at pose-target selection. This diagnostic path also
    // retains the physical areas needed for WeightedInteraction scoring and reference checks.

    const auto offline_start =
        std::chrono::steady_clock::now();


    auto protein =
        load_protein(
            config.protein_path.string()
        );

    auto pocket =
        load_pocket(
            config.pocket_path.string()
        );


    result.protein_atoms =
        protein->getNumAtoms();

    result.pocket_atoms =
        pocket->getNumAtoms();


    const auto protein_areas = prepare_protein_areas(config.protein_path, *protein, *pocket);

    result.protein_areas =
        protein_areas.size();


    const auto pocket_protein_areas =
        filter_protein_areas_in_pocket(
            protein_areas,
            *pocket
        );

    result.pocket_protein_areas =
        pocket_protein_areas.size();


    const auto protein_matching_areas =
        build_protein_matching_areas(
            protein_areas
        );

    result.protein_matching_areas =
        protein_matching_areas.size();


    // Occupancy uses the full protein. Optional area cropping changes chemical
    // masks and targets, independently of the pocket feature filter below.

    const GridBounds pocket_bounds =
        compute_pocket_bounds(
            *pocket
        );


    result.pocket_size_x =
        pocket_bounds.xmax -
        pocket_bounds.xmin;

    result.pocket_size_y =
        pocket_bounds.ymax -
        pocket_bounds.ymin;

    result.pocket_size_z =
        pocket_bounds.zmax -
        pocket_bounds.zmin;


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


    result.grid_nx =
        protein_grid.nx;

    result.grid_ny =
        protein_grid.ny;

    result.grid_nz =
        protein_grid.nz;


    result.total_voxels =
        protein_grid.voxels.size();

    result.occupied_voxels =
        count_occupied_voxels(
            protein_grid
        );

    result.interaction_voxels =
        count_interaction_voxels(
            protein_grid
        );

    result.multi_interaction_voxels =
        count_multi_interaction_voxels(
            protein_grid
        );


    // Target proximity is measured to any pocket atom. WeightedInteraction uses the stricter
    // feature-membership subset prepared above, not this centroid filter.

    const auto pocket_matching_areas =
        filter_matching_areas_near_pocket(
            protein_matching_areas,
            *pocket,
            config.pocket_centroid_distance
        );


    result.pocket_matching_areas =
        pocket_matching_areas.size();


    // --------------------------------------------------------
    // Pose targets
    // --------------------------------------------------------

    const auto pose_targets =
        generate_pose_targets(
            pocket_matching_areas,
            protein_grid,
            *protein,
            *pocket
        );


    result.pose_targets =
        pose_targets.size();

    result.pose_targets_missing =
        pocket_matching_areas.size() -
        pose_targets.size();


    const auto offline_end =
        std::chrono::steady_clock::now();


    result.timings
        .offline_target_preparation_seconds =
            std::chrono::duration<double>(
                offline_end -
                offline_start
            ).count();


    // ========================================================
    // ONLINE
    //
    // Ligand-specific work starts here, including its protein triplet index.
    // ========================================================

    const auto online_start =
        std::chrono::steady_clock::now();


    // ========================================================
    // ONLINE - LIGAND PREPARATION
    // ========================================================

    const auto ligand_preparation_start =
        std::chrono::steady_clock::now();


    auto ligand =
        load_ligand(
            config.ligand_path.string()
        );


    result.ligand_atoms =
        ligand->getNumAtoms();


    const auto ligand_areas =
        get_ligand_areas(
            *ligand
        );

    result.ligand_areas =
        ligand_areas.size();


    const auto ligand_matching_areas =
        build_ligand_matching_areas(
            ligand_areas
        );

    result.ligand_matching_areas =
        ligand_matching_areas.size();


    const double ligand_diameter =
        compute_ligand_diameter(
            ligand_areas
        );

    result.ligand_diameter =
        ligand_diameter;


    const auto ligand_triplets =
        generate_triplets(
            ligand_matching_areas,
            ligand_diameter
        );

    result.ligand_triplets =
        ligand_triplets.size();


    const auto ligand_preparation_end =
        std::chrono::steady_clock::now();


    result.timings
        .online_ligand_preparation_seconds =
            std::chrono::duration<double>(
                ligand_preparation_end -
                ligand_preparation_start
            ).count();



    // ========================================================
    // ONLINE - POSE SEARCH
    //
    // Build the protein index with this ligand's diameter as the edge cutoff.
    // ========================================================

    const auto pose_search_start =
        std::chrono::steady_clock::now();


    // This reference pipeline indexes only triangles within the ligand diameter.
    // Shared-cutoff screening is a different candidate-selection policy.

    const auto index_start =
        std::chrono::steady_clock::now();


    const auto protein_triplet_hash =
        build_protein_triplet_hash_table(
            pose_targets,
            ligand_diameter,
            config.triplet_bin_size
        );

    const auto flat_triplet_index =
        flatten_protein_triplet_index(
            protein_triplet_hash
        );


    if (
        !validate_flat_triplet_index(
            protein_triplet_hash,
            flat_triplet_index
        )
    ) {
        throw std::runtime_error(
            "Flat protein triplet index validation failed"
        );
    }


    const auto index_end =
        std::chrono::steady_clock::now();


    const double protein_index_seconds =
        std::chrono::duration<double>(
            index_end -
            index_start
        ).count();


    // --------------------------------------------------------
    // Candidate pose generation
    // --------------------------------------------------------

    result.pose_generation =
        generate_candidate_poses_from_index(
            *ligand,
            ligand_matching_areas,
            ligand_triplets,
            pose_targets,
            protein_grid,
            flat_triplet_index,
            config.triplet_bin_size,
            config.operational_clash_threshold,
            config.clash_thresholds
        );


    result.pose_generation
        .timings
        .protein_triplet_index_seconds =
            protein_index_seconds;


    const auto pose_search_end =
        std::chrono::steady_clock::now();


    result.timings
        .online_pose_search_seconds =
            std::chrono::duration<double>(
                pose_search_end -
                pose_search_start
            ).count();


    // Fast scores every survivor; only its configured prefilter reaches WeightedInteraction.
    // The legacy ranking keeps its own tie-breaks and reference diagnostics.

    const auto scoring_start =
        std::chrono::steady_clock::now();


    result.pose_ranking =
        rank_poses(
            result.pose_generation
                .surviving_poses,
            ligand_matching_areas,
            ligand_areas,
            pocket_protein_areas,
            protein_grid,
            config.fast_prefilter_top_n
        );


    const auto scoring_end =
        std::chrono::steady_clock::now();


    result.timings
        .online_scoring_seconds =
            std::chrono::duration<double>(
                scoring_end -
                scoring_start
            ).count();


    result.timings
        .online_total_seconds =
            std::chrono::duration<double>(
                scoring_end -
                online_start
            ).count();


    result.timings.total_seconds =
        result.timings
            .offline_target_preparation_seconds +
        result.timings
            .online_total_seconds;


    // Reference checks below are outside the recorded totals. Per-pose RMSD
    // computed during generation is already inside the search interval.


    // --------------------------------------------------------
    // Crystal ligand / fast grid
    // --------------------------------------------------------

    const auto crystal_grid_score =
        score_ligand_on_grid(
            ligand_matching_areas,
            protein_grid
        );


    result.crystal_grid_total =
        crystal_grid_score.total_areas;

    result.crystal_grid_in_grid =
        crystal_grid_score.areas_in_grid;

    result.crystal_grid_occupied =
        crystal_grid_score
            .areas_on_occupied_voxel;

    result.crystal_grid_compatible =
        crystal_grid_score.compatible_areas;


    // --------------------------------------------------------
    // Crystal chemical interactions - detected protein areas (optionally cropped)
    // --------------------------------------------------------

    const auto interactions =
        interaction::find_interactions(
            protein_areas,
            ligand_areas
        );


    result.crystal_interactions =
        interactions.size();


    std::unordered_set<std::size_t>
        crystal_interacting_ligand_indices;


    for (const auto& formed :
         interactions) {

        crystal_interacting_ligand_indices.insert(
            formed
                .index_ligand_interaction_area
        );
    }


    result.crystal_interacting_ligand_areas =
        crystal_interacting_ligand_indices.size();


    // --------------------------------------------------------
    // Pocket-only chemical interaction check
    // --------------------------------------------------------

    const auto pocket_weighted_interaction_score =
        score_weighted_interactions(
            pocket_protein_areas,
            ligand_areas
        );


    result.pocket_interactions =
        pocket_weighted_interaction_score
            .total_interactions;

    result.pocket_interacting_ligand_areas =
        pocket_weighted_interaction_score
            .satisfied_ligand_areas;


    // --------------------------------------------------------
    // Matching-area / pocket diagnostic
    // --------------------------------------------------------

    result.matching_area_distance_diagnostic =
        compute_matching_area_distance_diagnostic(
            protein_matching_areas,
            *pocket
        );


    // --------------------------------------------------------
    // Pose-target / crystal diagnostic
    // --------------------------------------------------------

    result.pose_target_crystal_diagnostic =
        compute_pose_target_crystal_diagnostic(
            pose_targets,
            pocket_matching_areas,
            ligand_matching_areas,
            interactions
        );


    return result;
}
