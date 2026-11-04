#ifndef GPU_DOCKING_CONFIG_HPP
#define GPU_DOCKING_CONFIG_HPP

#include "gpu_docking.hpp"
#include "pose_export.hpp"

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <optional>
#include <string>

struct GpuDockingCliOptions {
    RunConfig config;
    GpuDockingSettings settings;
    std::filesystem::path input, output, references, center_reference;
    bool pose_center_distance = false;
    // Export limit only; settings.fast_prefilter controls the rescoring pool.
    std::size_t top_k = 20;
    PoseExportOptions pose_export;
    // Explicit output controls enable pool-wide reference metrics independently
    // of how many poses are written. Legacy invocations keep their old output.
    bool pose_output_controls = false;
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    // Recover molecular record failures; reference pairing is excluded.
    bool continue_on_error = false;
    bool no_sdf = false;
    // Unset preserves the original serial driver. Explicit OFF/ON uses the
    // same prepared inputs, with or without one CPU prefetch worker.
    std::optional<bool> cpu_gpu_overlap;
    bool overlap_trace = false;
    bool gpu_batch = false;
    std::size_t batch_size = 16;
    std::size_t batch_memory_bytes = std::size_t{2048} * 1024 * 1024;
    std::string kabsch_method, query_method;
    // Each source is "cli", "environment", or "default".
    std::string kabsch_source, query_source, diagnostics_source;
    std::string ranking_requested = "fast";
};

// Parse values without opening files or changing the process environment.
// CLI selectors override environment selectors; otherwise Kabsch/query accept
// only canonical values (including rejection of an empty set variable).
// DOCKING_PROFILE_STAGES retains its historical exact-"1" activation rule.
GpuDockingCliOptions parse_gpu_docking_options(int argc, char** argv);

// Call before constructing any CUDA target. Sets only the two environment
// selectors read by existing Kabsch/query implementations; diagnostic selection
// remains explicit in settings.diagnostic_stages. Throws on setenv failure.
void apply_gpu_docking_environment(const GpuDockingCliOptions& options);

void write_gpu_docking_usage(std::ostream& output);

#endif
