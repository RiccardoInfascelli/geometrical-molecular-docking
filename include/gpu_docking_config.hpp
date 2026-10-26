#ifndef GPU_DOCKING_CONFIG_HPP
#define GPU_DOCKING_CONFIG_HPP

#include "gpu_docking.hpp"

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <string>

struct GpuDockingCliOptions {
    RunConfig config;
    GpuDockingSettings settings;
    std::filesystem::path input, output, references;
    // Export limit only; settings.fast_prefilter controls the rescoring pool.
    std::size_t top_k = 20;
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    // Recover per-record parsing/docking failures; reference pairing is excluded.
    bool continue_on_error = false;
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
