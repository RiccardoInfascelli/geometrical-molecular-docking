#include "protein_preparation.hpp"

#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"

#include <chrono>

PreparedProtein prepare_protein(
    const RunConfig& config,
    std::optional<double> minimum_target_clearance)
{
    auto protein = load_protein(config.protein_path.string());
    auto pocket = load_pocket(config.pocket_path.string());
    const auto protein_areas = get_protein_areas(*protein);
    const auto protein_matching_areas = build_protein_matching_areas(protein_areas);

    PreparedProtein prepared;
    prepared.grid = build_empty_protein_grid(
        *pocket, config.grid_voxel_size, config.grid_margin);
    mark_protein_occupancy(*protein, prepared.grid);
    // Grid interactions use all protein areas, before filtering target areas.
    mark_interaction_regions(protein_matching_areas, prepared.grid);

    // This distance is from each area centroid to the nearest pocket atom.
    const auto pocket_matching = filter_matching_areas_near_pocket(
        protein_matching_areas, *pocket, config.pocket_centroid_distance);
    // Preserve the 6.5 A pocket-region default, independently of grid_margin,
    // and the current selection of at most one target per MatchingArea.
    prepared.targets = generate_pose_targets(
        pocket_matching, prepared.grid, *protein, *pocket);
    prepared.targets_before_pruning = prepared.targets.size();
    if (minimum_target_clearance) {
        const auto start = std::chrono::steady_clock::now();
        prune_pose_targets(prepared.targets, *protein, minimum_target_clearance);
        prepared.target_pruning_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }

    return prepared;
}
