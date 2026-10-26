#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include "matching_areas.hpp"
#include "pose_targets.hpp"
#include "pose_generation.hpp"
#include "pose_ranking.hpp"

#include <array>
#include <cstddef>
#include <GraphMol/ROMol.h>
#include <interaction/find_interaction.hpp>

#include <vector>

// Reference-based evaluation helpers; they do not select or reorder search poses.
void print_pose_target_crystal_diagnostic(
    const std::vector<PoseTarget>& pose_targets,
    const std::vector<MatchingArea>& protein_matching_areas,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::formed_interaction>& interactions
);

void print_matching_area_distance_to_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket
);

void print_pose_generation_diagnostics(
    const PoseGenerationResult& result,
    std::size_t pose_target_count,
    std::size_t ligand_triplet_count,
    std::size_t ligand_matching_area_count,
    std::size_t ligand_atom_count,
    const std::array<double, 4>& clash_thresholds
);

void print_ranking_diagnostic(
    const char* title,
    const std::vector<RankedPoseDiagnostic>& poses,
    bool weighted_interaction_available
);


void print_weighted_interaction_ranking_diagnostics(
    const WeightedInteractionRankingResult& result
);

void print_pose_ranking_diagnostics(
    const PoseRankingResult& result
);

struct MatchingAreaDistanceDiagnostic
{
    std::size_t matching_areas{};

    double mean_distance{};
    double min_distance{};
    double max_distance{};

    std::array<std::size_t, 7> histogram{};
};


// Summarizes each area centroid's distance to its nearest pocket atom.
// Histogram upper bounds are 2, 4, 6, 8, 10 and 15 angstroms, then an open tail.
// Inputs share a coordinate frame; an empty area list returns zero statistics.
MatchingAreaDistanceDiagnostic
compute_matching_area_distance_diagnostic(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket
);

void print_matching_area_distance_diagnostic(
    const MatchingAreaDistanceDiagnostic& diagnostic
);


struct PoseTargetCrystalDiagnostic
{
    std::size_t interactions{};

    std::size_t interactions_with_target{};
    std::size_t interactions_without_target{};

    std::size_t interacting_ligand_areas{};
    std::size_t ligand_areas_with_target{};

    double mean_target_error{};
    double max_target_error{};
    double mean_original_error{};

    std::size_t repositioning_samples{};

    std::size_t improved{};
    std::size_t worsened{};
    std::size_t unchanged{};
};


// Interaction indices must refer to the physical areas behind the MatchingAreas.
// Uses the first target with matching source index and key for each interaction.
// Errors are centroid distances in the reference frame, not whole-ligand RMSD.
PoseTargetCrystalDiagnostic
compute_pose_target_crystal_diagnostic(
    const std::vector<PoseTarget>& pose_targets,
    const std::vector<MatchingArea>& protein_matching_areas,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::formed_interaction>& interactions
);

void print_pose_target_crystal_diagnostic(
    const PoseTargetCrystalDiagnostic& diagnostic
);

#endif
