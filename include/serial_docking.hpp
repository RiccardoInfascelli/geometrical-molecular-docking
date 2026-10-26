#pragma once

#include "geometry.hpp"
#include "interaction_scoring.hpp"
#include "run_config.hpp"

#include <cstddef>
#include <vector>


struct DockedPose
{
    RigidTransform transform{};

    double fast_score{};

    WeightedInteractionScore weighted_interaction_score{};

    std::size_t occupied_atoms{};
};


struct SerialDockingTimings
{
    double offline_target_preparation_seconds{};

    double ligand_preparation_seconds{};

    double protein_triplet_index_seconds{};
    // Generation minus grid/clash: includes lookup and bookkeeping, not just Kabsch.
    double matching_kabsch_seconds{};
    double grid_clash_seconds{};

    // These intervals include sorting; Fast also includes prefilter selection.
    double fast_scoring_seconds{};
    double weighted_interaction_scoring_seconds{};

    double online_total_seconds{};
    double total_seconds{};
};


struct SerialDockingResult
{
    std::vector<DockedPose> top_poses;

    SerialDockingTimings timings;
};


// Hash-index CPU search without oracle RMSD or diagnostic threshold sweeps.
// The protein index cutoff is the ligand's area-centroid diameter, in angstroms.
// Fast prefilters candidates for WeightedInteraction; top_n limits the final returned ranking.
// Ties use the legacy support/contact/clash criteria, not the score-only pose IDs.
SerialDockingResult run_serial_docking(
    const RunConfig& config,
    std::size_t top_n = 20
);
