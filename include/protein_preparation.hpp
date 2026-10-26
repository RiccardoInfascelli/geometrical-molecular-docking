#ifndef PROTEIN_PREPARATION_HPP
#define PROTEIN_PREPARATION_HPP

#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"

#include <cstddef>
#include <optional>
#include <vector>

// Host data reused for every ligand against the same protein and pocket.
struct PreparedProtein
{
    ProteinGrid grid;
    std::vector<PoseTarget> targets;
    std::size_t targets_before_pruning = 0;
    double target_pruning_seconds = 0.0;
};

// Loads protein and pocket, builds the grid and selects the pose targets.
// Only the grid and targets are retained; molecule and area temporaries are
// released before return. Triplet indexing and its cutoff belong to the caller.
// Optional pruning runs after target selection and never modifies the grid.
// Clearance is in angstroms: nullopt disables the filter, while zero enables it.
// Filter before indexing; target indices refer to the final vector order.
PreparedProtein prepare_protein(
    const RunConfig& config,
    std::optional<double> minimum_target_clearance = std::nullopt);

#endif // PROTEIN_PREPARATION_HPP
