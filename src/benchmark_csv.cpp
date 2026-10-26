#include "benchmark_csv.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>


void append_benchmark_csv(
    const RunResult& result,
    const std::filesystem::path& csv_path)
{
    const auto parent =
        csv_path.parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(
            parent
        );
    }


    const bool write_header =
        !std::filesystem::exists(csv_path) ||
        std::filesystem::file_size(csv_path) == 0;


    std::ofstream file(
        csv_path,
        std::ios::app
    );


    if (!file) {
        throw std::runtime_error(
            "Could not open benchmark CSV: " +
            csv_path.string()
        );
    }


    // ========================================================
    // Oracle helper
    //
    // RMSD is used ONLY for evaluation.
    // It never participates in ranking.
    // ========================================================

    const auto best_rmsd_in_top =
        [](
            const auto& poses,
            std::size_t top_n)
        {
            if (
                poses.empty() ||
                top_n == 0
            ) {
                return
                    std::numeric_limits<double>::
                        quiet_NaN();
            }


            const std::size_t count =
                std::min(
                    top_n,
                    poses.size()
                );


            double best =
                std::numeric_limits<double>::
                    infinity();


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


    // ========================================================
    // Header
    // ========================================================

    if (write_header) {

        file
            << "complex,"

            // Search workload
            << "ligand_atoms,"
            << "ligand_areas,"
            << "ligand_triplets,"
            << "protein_triplets,"
            << "protein_hash_keys,"
            << "candidate_matches,"
            << "transforms_generated,"
            << "survivors_5pct,"

            // Generation / prefilter quality
            << "best_rmsd_5pct,"
            << "best_rmsd_10pct,"
            << "best_rmsd_fast_top500,"

            // FAST ONLY
            << "fast_top1,"
            << "fast_top5,"
            << "fast_top10,"
            << "fast_top20,"

            // WeightedInteraction on FAST TOP 50
            << "weighted_interaction50_top1,"
            << "weighted_interaction50_top5,"
            << "weighted_interaction50_top10,"
            << "weighted_interaction50_top20,"

            // WeightedInteraction on FAST TOP 500
            << "weighted_interaction500_top1,"
            << "weighted_interaction500_top5,"
            << "weighted_interaction500_top10,"
            << "weighted_interaction500_top20,"

            // Main timings
            << "offline_seconds,"
            << "ligand_prep_seconds,"
            << "pose_search_seconds,"
            << "scoring_seconds,"
            << "online_seconds,"
            << "total_seconds,"

            // Pose-search breakdown
            << "protein_index_seconds,"
            << "matching_kabsch_seconds,"
            << "grid_clash_seconds,"
            << "oracle_rmsd_seconds,"

            // Scoring breakdown
            << "fast_scoring_seconds,"
            << "weighted_interaction50_scoring_seconds,"
            << "weighted_interaction500_scoring_seconds"

            << '\n';
    }


    // ========================================================
    // Rankings
    // ========================================================

    const auto& fast =
        result.pose_ranking
            .fast_weighted_ranking;


    const auto& weighted_interaction50 =
        result.pose_ranking
            .weighted_interaction_top50_ranking;


    const auto& weighted_interaction500 =
        result.pose_ranking
            .weighted_interaction_ranking;


    // CSV labels assume the default clash sweep {0, 0.025, 0.05, 0.10}.
    // Changing the configuration does not update these labels or array indices.

    const double best_rmsd_5pct =
        result.pose_generation
            .best_rmsd_below_clash_threshold[2];


    const double best_rmsd_10pct =
        result.pose_generation
            .best_rmsd_below_clash_threshold[3];


    // ========================================================
    // Row
    // ========================================================

    file << std::setprecision(10);


    file
        << result.complex_id << ","

        // Search workload
        << result.ligand_atoms << ","
        << result.ligand_areas << ","
        << result.ligand_triplets << ","

        << result.pose_generation
               .protein_triplet_records
        << ","

        << result.pose_generation
               .protein_hash_keys
        << ","

        << result.pose_generation
               .total_candidate_matches
        << ","

        << result.pose_generation
               .transforms_generated
        << ","

        << result.pose_generation
               .surviving_poses.size()
        << ","

        // Generation quality
        << best_rmsd_5pct << ","
        << best_rmsd_10pct << ","

        << result.pose_ranking
               .best_prefilter_rmsd
        << ","

        // FAST ONLY
        << best_rmsd_in_top(fast, 1) << ","
        << best_rmsd_in_top(fast, 5) << ","
        << best_rmsd_in_top(fast, 10) << ","
        << best_rmsd_in_top(fast, 20) << ","

        // WeightedInteraction TOP 50
        << best_rmsd_in_top(weighted_interaction50, 1) << ","
        << best_rmsd_in_top(weighted_interaction50, 5) << ","
        << best_rmsd_in_top(weighted_interaction50, 10) << ","
        << best_rmsd_in_top(weighted_interaction50, 20) << ","

        // WeightedInteraction TOP 500
        << best_rmsd_in_top(weighted_interaction500, 1) << ","
        << best_rmsd_in_top(weighted_interaction500, 5) << ","
        << best_rmsd_in_top(weighted_interaction500, 10) << ","
        << best_rmsd_in_top(weighted_interaction500, 20) << ","

        // Main timings
        << result.timings
               .offline_target_preparation_seconds
        << ","

        << result.timings
               .online_ligand_preparation_seconds
        << ","

        << result.timings
               .online_pose_search_seconds
        << ","

        << result.timings
               .online_scoring_seconds
        << ","

        << result.timings
               .online_total_seconds
        << ","

        << result.timings
               .total_seconds
        << ","

        // Pose-search timings
        << result.pose_generation.timings
               .protein_triplet_index_seconds
        << ","

        << result.pose_generation.timings
               .matching_kabsch_seconds
        << ","

        << result.pose_generation.timings
               .grid_clash_seconds
        << ","

        << result.pose_generation.timings
               .oracle_rmsd_seconds
        << ","

        // Scoring timings
        << result.pose_ranking
               .fast_scoring_seconds
        << ","

        << result.pose_ranking
               .weighted_interaction_top50_scoring_seconds
        << ","

        << result.pose_ranking
               .weighted_interaction_scoring_seconds

        << '\n';
}