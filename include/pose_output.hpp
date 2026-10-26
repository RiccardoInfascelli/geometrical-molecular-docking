#ifndef POSE_OUTPUT_HPP
#define POSE_OUTPUT_HPP

#include "pipeline.hpp"
#include "run_config.hpp"

#include <cstddef>

// Export the WeightedInteraction top_n poses, ranking CSV, references and PyMOL script
// under output_directory/complex_id. The input ligand is reloaded and transformed.
// Existing output files are overwritten; old extra pose files are not removed.
void save_pose_outputs(
    const RunResult& result,
    const RunConfig& config,
    std::size_t top_n = 20
);

#endif