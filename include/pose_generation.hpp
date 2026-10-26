#ifndef POSE_GENERATION_HPP
#define POSE_GENERATION_HPP

#include "geometry.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
// Keep the alignment API available to existing users of this header.
#include "rigid_alignment.hpp"
#include "stage_profile.hpp"
#include "triplet_generation.hpp"

#include <GraphMol/ROMol.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>


struct RankedPoseDiagnostic
{
    std::size_t compatible_areas{};
    std::size_t occupied_atoms{};

    double clash_fraction{};
    double rmsd{};

    RigidTransform transform{};

    double fast_weighted_score{};

    WeightedInteractionScore weighted_interaction_score{};
};

struct PoseSearchTimings
{
    double protein_triplet_index_seconds{};

    // Residual after grid/clash and RMSD: includes lookup and bookkeeping,
    // so this is not an isolated measurement of rigid alignment.
    double matching_kabsch_seconds{};

    double grid_clash_seconds{};

    // Evaluation only; included in generation_seconds but never used for selection.
    double oracle_rmsd_seconds{};
};


struct PoseGenerationResult
{
    std::vector<RankedPoseDiagnostic>
        surviving_poses;


    // Record/key counts of the supplied flat index, retaining historical names.
    std::size_t protein_triplet_records{};
    std::size_t protein_hash_keys{};


    // A candidate match is one protein record; equivalent ligand orders can
    // produce several transforms from that match.
    std::size_t ligand_triplets_with_match{};
    std::size_t total_candidate_matches{};
    std::size_t largest_bucket{};


    // Transform generation
    std::size_t transforms_generated{};
    std::size_t transforms_valid{};
    std::size_t poses_with_grid_support{};
    std::size_t best_grid_score{};


    // Diagnostics are collected after the clash_thresholds.back() filter,
    // before the operational survivor filter.
    std::array<std::size_t, 4>
        poses_below_clash_threshold{};

    std::array<double, 4>
        best_rmsd_below_clash_threshold{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
        };


    // Absolute clash diagnostic
    std::size_t poses_zero_clash{};
    std::size_t poses_one_clash{};
    std::size_t poses_two_clashes{};
    std::size_t poses_three_to_five_clashes{};
    // Historical CSV meaning: also counts all early diagnostic-limit rejects,
    // even if their absolute number of occupied atoms is five or fewer.
    std::size_t poses_over_five_clashes{};

    std::size_t best_grid_score_zero_clash{};

    double best_zero_clash_rmsd =
        std::numeric_limits<double>::infinity();

    std::vector<std::size_t>
        zero_clash_grid_histogram;


    double generation_seconds{};

    PoseSearchTimings timings;
};


// Borrows an index built from these pose_targets and the same triplet_bin_size.
// Requires a nonempty ligand with coordinates and valid area/target indices.
// Keeps lookup/record/equivalent-order traversal order; no ranking or deduplication.
// No ligand-diameter filter is reapplied to records from the supplied index.
// clash_thresholds.back() gates diagnostics before the operational clash filter;
// it should cover the largest threshold to be reported and the operational limit.
// DOCKING_CPU_CLASH_METHOD=cached prepares atom coordinates once for this search;
// all other values use the original per-pose RDKit access.
// screening_only skips unweighted grid support and RMSD, without changing survivors.
// Support fields stay zero, pose RMSD is NaN and best-RMSD aggregates stay infinite;
// clash diagnostics remain, but the zero-clash support histogram uses only bin zero.
// The default keeps full diagnostics for ranking; only screening callers opt out.
PoseGenerationResult generate_candidate_poses_from_index(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    const FlatTripletIndex& protein_triplet_index,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds,
    bool screening_only = false
);


// Builds a ligand-specific index, then runs the same search. Index construction is
// timed separately from generation_seconds; RMSD is evaluated against the input pose.
PoseGenerationResult generate_candidate_poses(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    double ligand_diameter,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds
);

struct CpuPoseStageProfile
{
    StageProfile stages;
    std::size_t candidate_matches{};
    std::size_t transforms_generated{};
    std::size_t transforms_valid{};
    std::size_t survivors{};
    bool has_fast_score{};
    double best_fast_score{};
};

// Diagnostic batch replay of the screening search with the current alignment and
// clash selectors. Preserves candidate multiplicity, traversal and clash filters.
// Materializes intermediate arrays so each batch needs only one pair of clocks;
// these times have different locality and allocation costs from the online loop.
// RMSD, unweighted support and legacy per-pose diagnostics are not replayed.
CpuPoseStageProfile profile_cpu_pose_stages(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    const FlatTripletIndex& protein_triplet_index,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds
);

#endif
