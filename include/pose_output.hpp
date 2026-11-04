#ifndef POSE_OUTPUT_HPP
#define POSE_OUTPUT_HPP

#include "pipeline.hpp"
#include "run_config.hpp"

#include <cstddef>

// Export the available WeightedInteraction ranking under output_directory/complex_id.
// TopK writes at most top_n entries; All writes this whole ranking, after the Fast
// prefilter, without rescoring or restoring previously excluded poses. None returns
// before file access unless center-distance diagnostics require reference validation.
// export_structures=false writes only the ranking CSV.
// Unset detail preserves the legacy CSV schema; Full adds the rigid transforms.
// Structure export reloads the input ligand and applies each stored transform.
// Existing output files are overwritten; old extra pose files are not removed.
void save_pose_outputs(
    const RunResult& result,
    const RunConfig& config,
    std::size_t top_n = 20
);

#endif
