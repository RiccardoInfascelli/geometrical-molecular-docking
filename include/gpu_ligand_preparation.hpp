#ifndef GPU_LIGAND_PREPARATION_HPP
#define GPU_LIGAND_PREPARATION_HPP

#include "cuda_triplet_index.hpp"
#include "ligand_preparation.hpp"
#include "run_config.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

// Owns one immutable search input after host preparation. Moving the packet
// transfers the molecule, its feature snapshot and the matching query order
// together; no CUDA resource or reference to a producer's temporary is retained.
struct GpuPreparedLigand {
    std::unique_ptr<RDKit::RWMol> molecule;
    PreparedLigand ligand;
    std::vector<GpuTripletKey> queries;
    std::vector<GpuLigandTripletOrders> orders;
    GpuQueryPreparationInfo query_info;
    bool queries_enabled = false;
    bool trace_enabled = false;
    double bin_size = 0.0;
    double preparation_seconds = 0.0;
    double query_preparation_seconds = 0.0;
    std::chrono::steady_clock::time_point query_start{}, query_end{};

    GpuPreparedLigand() = default;
    GpuPreparedLigand(GpuPreparedLigand&&) = default;
    GpuPreparedLigand& operator=(GpuPreparedLigand&&) = default;
    GpuPreparedLigand(const GpuPreparedLigand&) = delete;
    GpuPreparedLigand& operator=(const GpuPreparedLigand&) = delete;
};

// Query packing failures were fatal inside the blocking CUDA search. Moving
// packing to a producer must not turn them into recoverable molecular errors.
class GpuQueryPreparationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Host-only preparation. The caller freezes selectors before starting a worker
// and supplies exclusive ownership of a sanitized molecule with coordinates.
// Disable queries for an empty shared index; empty ligand triplets also skip
// packing, as in the blocking search. No RDKit object is shared or modified later.
GpuPreparedLigand prepare_gpu_ligand(
    std::unique_ptr<RDKit::RWMol> molecule, const RunConfig& config,
    bool queries_enabled, bool collect_timings = true, bool trace = false);

// Optional host ranges for a separate Nsight run. These add no CUDA calls or
// waits and become no-ops when the build has no NVTX library.
bool gpu_overlap_trace_available() noexcept;
class GpuOverlapTraceRange {
public:
    GpuOverlapTraceRange(bool enabled, const char* name) noexcept;
    ~GpuOverlapTraceRange() noexcept;
    GpuOverlapTraceRange(const GpuOverlapTraceRange&) = delete;
    GpuOverlapTraceRange& operator=(const GpuOverlapTraceRange&) = delete;
private:
    bool active_ = false;
};

#endif
