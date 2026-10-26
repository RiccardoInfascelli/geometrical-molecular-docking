#include "diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>


namespace
{

double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    const double dx =
        static_cast<double>(a.x) -
        static_cast<double>(b.x);

    const double dy =
        static_cast<double>(a.y) -
        static_cast<double>(b.y);

    const double dz =
        static_cast<double>(a.z) -
        static_cast<double>(b.z);

    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}

} // namespace



void print_pose_target_crystal_diagnostic(
    const std::vector<PoseTarget>& pose_targets,
    const std::vector<MatchingArea>& protein_matching_areas,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::formed_interaction>& interactions)
{
    print_pose_target_crystal_diagnostic(
        compute_pose_target_crystal_diagnostic(
            pose_targets,
            protein_matching_areas,
            ligand_matching_areas,
            interactions
        )
    );
}

   

void print_matching_area_distance_to_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket)
{
    print_matching_area_distance_diagnostic(
        compute_matching_area_distance_diagnostic(
            matching_areas,
            pocket
        )
    );
}


void print_pose_generation_diagnostics(
    const PoseGenerationResult& result,
    std::size_t pose_target_count,
    std::size_t ligand_triplet_count,
    std::size_t ligand_matching_area_count,
    std::size_t ligand_atom_count,
    const std::array<double, 4>& clash_thresholds)
{
    // ========================================================
    // Protein target triplets
    // ========================================================

    std::cout
        << "\n=== Protein target triplets ===\n"

        << "Pose targets: "
        << pose_target_count
        << '\n'

        << "Protein triplets indexed: "
        << result.protein_triplet_records
        << '\n'

        << "Hash keys: "
        << result.protein_hash_keys
        << '\n';


    // ========================================================
    // 27-neighbor lookup
    // ========================================================

    std::cout
        << "\n=== 27-neighbor triplet lookup ===\n"

        << "Ligand triplets: "
        << ligand_triplet_count
        << '\n'

        << "Ligand triplets with match: "
        << result.ligand_triplets_with_match
        << '\n'

        << "Candidate matches: "
        << result.total_candidate_matches
        << '\n'

        << "Largest bucket: "
        << result.largest_bucket
        << '\n';


    // ========================================================
    // Candidate generation
    // ========================================================

    std::cout
        << "\n=== Candidate pose generation ===\n"

        << "Transforms generated: "
        << result.transforms_generated
        << '\n'

        << "Valid transforms: "
        << result.transforms_valid
        << '\n'

        << "Poses with grid support: "
        << result.poses_with_grid_support
        << '\n'

        << "Best grid-compatible areas: "
        << result.best_grid_score
        << " / "
        << ligand_matching_area_count
        << '\n'

        << "Generation + grid + clash time: "
        << result.generation_seconds
        << " s\n";


    // ========================================================
    // Fast clash
    // ========================================================

    std::cout
        << "\n=== Fast atom clash diagnostic ===\n"

        << "0 occupied atoms: "
        << result.poses_zero_clash
        << '\n'

        << "1 occupied atom: "
        << result.poses_one_clash
        << '\n'

        << "2 occupied atoms: "
        << result.poses_two_clashes
        << '\n'

        << "3-5 occupied atoms: "
        << result.poses_three_to_five_clashes
        << '\n'

        << ">5 occupied atoms: "
        << result.poses_over_five_clashes
        << '\n'

        << "Best grid score with 0 occupied atoms: "
        << result.best_grid_score_zero_clash
        << " / "
        << ligand_matching_area_count
        << '\n';


    // ========================================================
    // Zero-clash quality
    // ========================================================

    std::cout
        << "\n=== Zero-clash pose quality ===\n"

        << "Zero-clash poses: "
        << result.poses_zero_clash
        << '\n'

        << "Best RMSD to crystal: "
        << result.best_zero_clash_rmsd
        << " A\n"

        << "\nGrid score distribution:\n";


    for (
        std::size_t score = 0;
        score <
            result.zero_clash_grid_histogram.size();
        ++score
    ) {
        if (
            result.zero_clash_grid_histogram[
                score
            ] == 0
        ) {
            continue;
        }

        std::cout
            << "  "
            << score
            << " / "
            << ligand_matching_area_count
            << ": "
            << result.zero_clash_grid_histogram[
                score
            ]
            << '\n';
    }


    // ========================================================
    // Relative clash thresholds
    // ========================================================

    std::cout
        << "\n=== Relative clash threshold diagnostic ===\n"

        << "Ligand atoms: "
        << ligand_atom_count
        << '\n';


    for (
        std::size_t i = 0;
        i < clash_thresholds.size();
        ++i
    ) {
        const std::size_t maximum_occupied_atoms =
            static_cast<std::size_t>(
                std::floor(
                    clash_thresholds[i] *
                    static_cast<double>(
                        ligand_atom_count
                    )
                )
            );


        std::cout
            << "\nThreshold: "
            << clash_thresholds[i] * 100.0
            << "%"

            << " (max "
            << maximum_occupied_atoms
            << " occupied atoms)\n"

            << "Surviving poses: "
            << result
                   .poses_below_clash_threshold[i]
            << '\n'

            << "Best RMSD: "
            << result
                   .best_rmsd_below_clash_threshold[i]
            << " A\n";
    }
}

void print_ranking_diagnostic(
    const char* title,
    const std::vector<RankedPoseDiagnostic>& poses,
    bool weighted_interaction_available)
{
    std::cout
        << "\n=== "
        << title
        << " ===\n"
        << "Poses: "
        << poses.size()
        << '\n';

    if (poses.empty())
        return;


    const auto best_rmsd_in_top =
        [&poses](std::size_t top_n)
        {
            double best =
                std::numeric_limits<double>::infinity();

            const std::size_t count =
                std::min(
                    top_n,
                    poses.size()
                );

            for (
                std::size_t i = 0;
                i < count;
                ++i
            ) {
                best =
                    std::min(
                        best,
                        poses[i].rmsd
                    );
            }

            return best;
        };


    double global_best_rmsd =
        std::numeric_limits<double>::infinity();

    for (const auto& pose : poses) {

        global_best_rmsd =
            std::min(
                global_best_rmsd,
                pose.rmsd
            );
    }


    std::size_t best_rank = 0;

    for (
        std::size_t i = 0;
        i < poses.size();
        ++i
    ) {
        if (
            std::abs(
                poses[i].rmsd -
                global_best_rmsd
            ) < 1e-12
        ) {
            best_rank = i + 1;
            break;
        }
    }


    const auto& best_pose =
        poses[best_rank - 1];


    const std::array<std::size_t, 8>
        top_values{
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000
        };


    for (
        const std::size_t top_n :
        top_values
    ) {
        if (top_n > poses.size())
            continue;

        std::cout
            << "Best RMSD top "
            << top_n
            << ": "
            << best_rmsd_in_top(top_n)
            << " A\n";
    }


    std::cout
        << "\nBest overall RMSD in this set: "
        << global_best_rmsd
        << " A\n"

        << "Rank of best-RMSD pose: "
        << best_rank
        << " / "
        << poses.size()
        << '\n'

        << "Grid compatible areas: "
        << best_pose.compatible_areas
        << '\n'

        << "Fast weighted score: "
        << best_pose.fast_weighted_score
        << '\n'

        << "Occupied atoms: "
        << best_pose.occupied_atoms
        << '\n';


    if (weighted_interaction_available) {

        std::cout
            << "WeightedInteraction score: "
            << best_pose.weighted_interaction_score.weighted_score
            << '\n'

            << "Chemical interactions: "
            << best_pose.weighted_interaction_score.total_interactions
            << '\n'

            << "Satisfied ligand areas: "
            << best_pose.weighted_interaction_score.satisfied_ligand_areas
            << '\n';
    }
}

void print_weighted_interaction_ranking_diagnostics(
    const WeightedInteractionRankingResult& result)
{
    std::cout
        << "\n=== WeightedInteraction scoring benchmark: fast top 500 only ===\n"

        << "Poses scored: "
        << result.scored_candidates.size()
        << '\n'

        << "Total chemical interactions found: "
        << result.total_interactions
        << '\n'

        << "WeightedInteraction scoring time: "
        << result.scoring_seconds
        << " s\n";


    if (!result.scored_candidates.empty()) {

        std::cout
            << "Time per pose: "
            << (
                result.scoring_seconds /
                static_cast<double>(
                    result.scored_candidates.size()
                ) *
                1000.0
            )
            << " ms\n";
    }
}


void print_pose_ranking_diagnostics(
    const PoseRankingResult& result)
{
    std::cout
        << "\n=== Fast weighted scoring benchmark ===\n"

        << "Poses scored: "
        << result.input_pose_count
        << '\n'

        << "Fast weighted time: "
        << result.fast_scoring_seconds
        << " s\n";


    if (result.input_pose_count > 0) {

        std::cout
            << "Time per pose: "
            << (
                result.fast_scoring_seconds /
                static_cast<double>(
                    result.input_pose_count
                ) *
                1000.0
            )
            << " ms\n";
    }


    print_ranking_diagnostic(
        "Unweighted grid ranking",
        result.unweighted_ranking,
        false
    );


    print_ranking_diagnostic(
        "Fast weighted grid ranking",
        result.fast_weighted_ranking,
        false
    );


    std::cout
        << "\n=== Fast weighted prefilter ===\n"

        << "Input poses: "
        << result.input_pose_count
        << '\n'

        << "Poses kept for WeightedInteraction scoring: "
        << result.weighted_interaction_candidates.size()
        << '\n'

        << "Best RMSD surviving prefilter: "
        << result.best_prefilter_rmsd
        << " A\n";


    std::cout
        << "\n=== WeightedInteraction scoring benchmark: fast top 500 only ===\n"

        << "Poses scored: "
        << result.weighted_interaction_candidates.size()
        << '\n'

        << "Total chemical interactions found: "
        << result.total_interactions
        << '\n'

        << "WeightedInteraction scoring time: "
        << result.weighted_interaction_scoring_seconds
        << " s\n";


    if (!result.weighted_interaction_candidates.empty()) {

        std::cout
            << "Time per pose: "
            << (
                result.weighted_interaction_scoring_seconds /
                static_cast<double>(
                    result.weighted_interaction_candidates.size()
                ) *
                1000.0
            )
            << " ms\n";
    }


    print_ranking_diagnostic(
        "Interaction-count ranking / fast top 500",
        result.interaction_count_ranking,
        true
    );


    print_ranking_diagnostic(
        "Ligand-coverage ranking / fast top 500",
        result.interaction_coverage_ranking,
        true
    );


    print_ranking_diagnostic(
        "WeightedInteraction ranking / fast top 500",
        result.weighted_interaction_ranking,
        true
    );
}

MatchingAreaDistanceDiagnostic
compute_matching_area_distance_diagnostic(
    const std::vector<MatchingArea>& matching_areas,
    const RDKit::ROMol& pocket)
{
    MatchingAreaDistanceDiagnostic result;

    result.matching_areas =
        matching_areas.size();

    if (matching_areas.empty())
        return result;


    double distance_sum = 0.0;

    result.min_distance =
        std::numeric_limits<double>::infinity();

    result.max_distance =
        0.0;


    const auto& conformer =
        pocket.getConformer();


    for (const auto& area : matching_areas) {

        double nearest_distance =
            std::numeric_limits<double>::infinity();


        for (
            unsigned int atom_index = 0;
            atom_index < pocket.getNumAtoms();
            ++atom_index
        ) {
            const auto p =
                conformer.getAtomPos(
                    atom_index
                );


            const double dx =
                static_cast<double>(
                    area.centroid.x
                ) - p.x;

            const double dy =
                static_cast<double>(
                    area.centroid.y
                ) - p.y;

            const double dz =
                static_cast<double>(
                    area.centroid.z
                ) - p.z;


            const double distance =
                std::sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                );


            nearest_distance =
                std::min(
                    nearest_distance,
                    distance
                );
        }


        distance_sum +=
            nearest_distance;

        result.min_distance =
            std::min(
                result.min_distance,
                nearest_distance
            );

        result.max_distance =
            std::max(
                result.max_distance,
                nearest_distance
            );


        if (nearest_distance < 2.0)
            ++result.histogram[0];

        else if (nearest_distance < 4.0)
            ++result.histogram[1];

        else if (nearest_distance < 6.0)
            ++result.histogram[2];

        else if (nearest_distance < 8.0)
            ++result.histogram[3];

        else if (nearest_distance < 10.0)
            ++result.histogram[4];

        else if (nearest_distance < 15.0)
            ++result.histogram[5];

        else
            ++result.histogram[6];
    }


    result.mean_distance =
        distance_sum /
        static_cast<double>(
            matching_areas.size()
        );


    return result;
}

PoseTargetCrystalDiagnostic
compute_pose_target_crystal_diagnostic(
    const std::vector<PoseTarget>& pose_targets,
    const std::vector<MatchingArea>& protein_matching_areas,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::formed_interaction>& interactions)
{
    PoseTargetCrystalDiagnostic result;

    result.interactions =
        interactions.size();


    std::unordered_set<std::size_t>
        interacting_ligand_areas;

    std::unordered_set<std::size_t>
        ligand_areas_with_target;


    double target_error_sum =
        0.0;

    double original_error_sum =
        0.0;


    for (const auto& formed :
         interactions) {

        const std::size_t protein_source =
            formed.index_protein_interaction_area;

        const std::size_t ligand_source =
            formed.index_ligand_interaction_area;


        interacting_ligand_areas.insert(
            ligand_source
        );


        const MatchingArea* ligand_area =
            nullptr;


        for (const auto& area :
             ligand_matching_areas) {

            if (
                area.source_area_index ==
                ligand_source
            ) {
                ligand_area =
                    &area;

                break;
            }
        }


        if (ligand_area == nullptr) {

            ++result.interactions_without_target;
            continue;
        }


        const MatchingArea* protein_area =
            nullptr;


        for (const auto& area :
             protein_matching_areas) {

            if (
                area.source_area_index ==
                    protein_source &&
                area.key ==
                    ligand_area->key
            ) {
                protein_area =
                    &area;

                break;
            }
        }


        const PoseTarget* target =
            nullptr;


        for (const auto& candidate :
             pose_targets) {

            if (
                candidate.source_area_index ==
                    protein_source &&
                candidate.key ==
                    ligand_area->key
            ) {
                target =
                    &candidate;

                break;
            }
        }


        if (
            target == nullptr ||
            protein_area == nullptr
        ) {
            ++result.interactions_without_target;
            continue;
        }


        ++result.interactions_with_target;


        ligand_areas_with_target.insert(
            ligand_source
        );


        const double original_error =
            point_distance(
                protein_area->centroid,
                ligand_area->centroid
            );


        const double target_error =
            point_distance(
                target->position,
                ligand_area->centroid
            );


        original_error_sum +=
            original_error;

        target_error_sum +=
            target_error;


        result.max_target_error =
            std::max(
                result.max_target_error,
                target_error
            );


        ++result.repositioning_samples;


        constexpr double EPS =
            1e-9;


        if (
            target_error <
            original_error - EPS
        ) {
            ++result.improved;
        }
        else if (
            target_error >
            original_error + EPS
        ) {
            ++result.worsened;
        }
        else {
            ++result.unchanged;
        }
    }


    result.interacting_ligand_areas =
        interacting_ligand_areas.size();

    result.ligand_areas_with_target =
        ligand_areas_with_target.size();


    if (result.repositioning_samples > 0) {

        const double n =
            static_cast<double>(
                result.repositioning_samples
            );

        result.mean_target_error =
            target_error_sum / n;

        result.mean_original_error =
            original_error_sum / n;
    }


    return result;
}

void print_pose_target_crystal_diagnostic(
    const PoseTargetCrystalDiagnostic& d)
{
    std::cout
        << "\n=== Pose target / crystal diagnostic ===\n"

        << "Chemical interactions: "
        << d.interactions
        << '\n'

        << "Chemical interactions with target: "
        << d.interactions_with_target
        << '\n'

        << "Chemical interactions without target: "
        << d.interactions_without_target
        << '\n'

        << "Interacting ligand areas with target: "
        << d.ligand_areas_with_target
        << " / "
        << d.interacting_ligand_areas
        << '\n'

        << "Mean target error: "
        << d.mean_target_error
        << " A\n"

        << "Max target error: "
        << d.max_target_error
        << " A\n";


    std::cout
        << "\n=== Repositioning quality ===\n"

        << "Mean original error: "
        << d.mean_original_error
        << " A\n"

        << "Mean target error: "
        << d.mean_target_error
        << " A\n"

        << "Improved interactions: "
        << d.improved
        << " / "
        << d.repositioning_samples
        << '\n'

        << "Worsened interactions: "
        << d.worsened
        << " / "
        << d.repositioning_samples
        << '\n'

        << "Unchanged interactions: "
        << d.unchanged
        << " / "
        << d.repositioning_samples
        << '\n';
}

void print_matching_area_distance_diagnostic(
    const MatchingAreaDistanceDiagnostic& d)
{
    std::cout
        << "\n=== Unfiltered matching-area distance to pocket ===\n"

        << "Matching areas: "
        << d.matching_areas
        << '\n'

        << "Mean distance: "
        << d.mean_distance
        << " A\n"

        << "Min distance: "
        << d.min_distance
        << " A\n"

        << "Max distance: "
        << d.max_distance
        << " A\n"

        << "\nDistance distribution:\n"

        << "  < 2 A:    "
        << d.histogram[0]
        << '\n'

        << "  2-4 A:    "
        << d.histogram[1]
        << '\n'

        << "  4-6 A:    "
        << d.histogram[2]
        << '\n'

        << "  6-8 A:    "
        << d.histogram[3]
        << '\n'

        << "  8-10 A:   "
        << d.histogram[4]
        << '\n'

        << "  10-15 A:  "
        << d.histogram[5]
        << '\n'

        << "  >= 15 A:  "
        << d.histogram[6]
        << '\n';
}