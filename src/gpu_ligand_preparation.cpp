#include "gpu_ligand_preparation.hpp"

#include <cmath>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef DOCKING_HAVE_NVTX
#include <nvToolsExt.h>
#endif

bool gpu_overlap_trace_available() noexcept {
#ifdef DOCKING_HAVE_NVTX
    return true;
#else
    return false;
#endif
}

GpuOverlapTraceRange::GpuOverlapTraceRange(bool enabled, const char* name) noexcept {
#ifdef DOCKING_HAVE_NVTX
    if (enabled) {
        nvtxRangePushA(name);
        active_ = true;
    }
#else
    (void)enabled;
    (void)name;
#endif
}

GpuOverlapTraceRange::~GpuOverlapTraceRange() noexcept {
#ifdef DOCKING_HAVE_NVTX
    if (active_) nvtxRangePop();
#endif
}

GpuPreparedLigand prepare_gpu_ligand(
    std::unique_ptr<RDKit::RWMol> molecule, const RunConfig& config,
    bool queries_enabled, bool collect_timings, bool trace) {
    if (!molecule) throw GpuQueryPreparationError("Cannot prepare an empty molecule owner");
    if (!std::isfinite(config.triplet_bin_size) || config.triplet_bin_size <= 0.0)
        throw GpuQueryPreparationError("Triplet bin size must be finite and positive");
    using Clock = std::chrono::steady_clock;
    GpuPreparedLigand result;
    result.molecule = std::move(molecule);
    result.queries_enabled = queries_enabled;
    result.trace_enabled = trace;
    result.bin_size = config.triplet_bin_size;
    const auto preparation_start = collect_timings ? Clock::now() : Clock::time_point{};
    prepare_ligand(*result.molecule, result.ligand);
    if (collect_timings)
        result.preparation_seconds = std::chrono::duration<double>(Clock::now() - preparation_start).count();
    result.query_info = {gpu_query_requested_threads(), 1};
    if (!queries_enabled || result.ligand.triplets.empty()) return result;

    GpuOverlapTraceRange range(trace, "cpu_query_preparation");
    // A common monotonic clock makes CPU intervals comparable across threads.
    // These host timestamps alone do not establish overlap with device kernels.
    if (collect_timings || trace) result.query_start = Clock::now();
    try {
        make_gpu_ligand_triplet_queries(result.ligand.triplets, config.triplet_bin_size,
                                       result.queries, result.orders, &result.query_info);
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        throw GpuQueryPreparationError(error.what());
    }
    if (collect_timings || trace) {
        result.query_end = Clock::now();
        result.query_preparation_seconds =
            std::chrono::duration<double>(result.query_end - result.query_start).count();
    }
    if (result.queries.size() != result.orders.size() || result.queries.size() != result.ligand.triplets.size())
        throw GpuQueryPreparationError("Prepared GPU query/order count mismatch");
    return result;
}
