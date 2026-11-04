#ifndef RUN_CONFIG_HPP
#define RUN_CONFIG_HPP

#include "pose_export.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>

enum class OutputMode
{
    Quiet,
    Summary,
    Debug
};

// Geometric lengths are in angstroms; clash thresholds are atom fractions.
struct RunConfig
{
    // Input
    std::filesystem::path complex_directory;
    std::string complex_id;

    std::filesystem::path protein_path;
    std::filesystem::path ligand_path;
    std::filesystem::path pocket_path;

    // Output
    std::filesystem::path output_directory =
        "results";

    OutputMode output_mode =
        OutputMode::Summary;

    // Export only; these options never change the pose pool or ranking.
    PoseExportOptions pose_export;
    bool export_structures = true;
    bool export_timings = false;
    bool pose_center_distance = false;
    std::filesystem::path center_reference_mol2;

    // Grid
    double grid_voxel_size =
        0.5;

    double grid_margin =
        6.5;

    // Triplets
    double triplet_bin_size =
        0.5;

    // Maximum area-centroid distance to any pocket atom, not the pocket centroid.
    double pocket_centroid_distance =
        2.0;

    // Maximum occupied-atom fraction accepted as an operational survivor.
    double operational_clash_threshold =
        0.05;

    // The last value is the early diagnostic rejection limit; keep it largest.
    std::array<double, 4> clash_thresholds{
        0.00,
        0.025,
        0.05,
        0.10
    };

    // Number of Fast-ranked candidates sent to WeightedInteraction scoring; zero selects none.
    std::size_t fast_prefilter_top_n =
        500;
};

// Derives input paths from the complex directory and parses output options.
// Scientific parameters retain the defaults above; they are not general CLI flags.
// Does not open or validate molecule files; loading belongs to the pipeline.
RunConfig parse_run_config(
    int argc,
    char* argv[]
);

// Call only from geometric_docking before alignment. Preserve an explicit
// environment selector; other applications and direct APIs keep their policy.
void apply_cpu_docking_environment();

#endif
