#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include "diagnostics.hpp"
#include "pose_generation.hpp"
#include "pose_ranking.hpp"
#include "run_config.hpp"

#include <cstddef>
#include <string>


struct RunTimings
{
    // Reusable protein/pocket preparation through grid and pose targets.
    double offline_target_preparation_seconds{};

    // Ligand loading, areas, diameter and triplets.
    double online_ligand_preparation_seconds{};

    // Includes the ligand-diameter-specific protein index and oracle RMSD,
    // as well as lookup, rigid alignment and grid/clash evaluation.
    double online_pose_search_seconds{};

    // Fast scoring, prefilter, WeightedInteraction scoring and all associated rankings.
    double online_scoring_seconds{};

    // Ligand preparation + pose search + scoring.
    double online_total_seconds{};

    // Offline + online; excludes later reference diagnostics and file output.
    double total_seconds{};
};


struct RunResult
{
    std::string complex_id;

    std::size_t protein_atoms{};
    std::size_t pocket_atoms{};
    std::size_t ligand_atoms{};

    std::size_t protein_areas{};
    std::size_t pocket_protein_areas{};
    std::size_t ligand_areas{};

    std::size_t protein_matching_areas{};
    std::size_t ligand_matching_areas{};

    double ligand_diameter{};
    std::size_t ligand_triplets{};

    double pocket_size_x{};
    double pocket_size_y{};
    double pocket_size_z{};

    int grid_nx{};
    int grid_ny{};
    int grid_nz{};

    std::size_t total_voxels{};
    std::size_t occupied_voxels{};
    std::size_t interaction_voxels{};
    std::size_t multi_interaction_voxels{};

    std::size_t crystal_grid_total{};
    std::size_t crystal_grid_in_grid{};
    std::size_t crystal_grid_occupied{};
    std::size_t crystal_grid_compatible{};

    std::size_t crystal_interactions{};
    std::size_t crystal_interacting_ligand_areas{};

    std::size_t pocket_interactions{};
    std::size_t pocket_interacting_ligand_areas{};

    std::size_t pocket_matching_areas{};
    std::size_t pose_targets{};
    std::size_t pose_targets_missing{};

    MatchingAreaDistanceDiagnostic
        matching_area_distance_diagnostic;

    PoseTargetCrystalDiagnostic
        pose_target_crystal_diagnostic;

    PoseGenerationResult
        pose_generation;

    PoseRankingResult
        pose_ranking;

    RunTimings timings;
};


// Single-complex CPU search with Fast-to-WeightedInteraction ranking and reference diagnostics.
// Uses a ligand-diameter-specific index, unlike shared-cutoff screening.
// The input ligand coordinates are the RMSD reference, never a ranking criterion.
// Returns results in memory; reporting and file output belong to the caller.
RunResult run_pipeline(
    const RunConfig& config
);

#endif
