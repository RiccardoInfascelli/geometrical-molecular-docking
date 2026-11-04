#ifndef GPU_LIGAND_PREFETCH_HPP
#define GPU_LIGAND_PREFETCH_HPP

#include "gpu_ligand_preparation.hpp"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

// One record owns its molecule, features and queries. Preparation failures keep
// the original record identity and are delivered in input order to the caller.
struct GpuPreparedRecord {
    std::size_t index = 0;
    std::string name;
    GpuPreparedLigand prepared;
    double parse_seconds = 0.0;
    std::size_t estimated_bytes = 0;
    std::exception_ptr error;
    const char* error_stage = "parsing/sanitization";
};

struct GpuPrefetchStatistics {
    std::size_t capacity = 1, max_ready_bytes = 0;
    std::size_t ready_records = 0, ready_bytes = 0;
    std::size_t high_water_records = 0, high_water_bytes = 0;
    std::size_t prepared_records = 0;
};

struct GpuLigandInput {
    std::size_t index = 0;
    std::string name;
    std::unique_ptr<RDKit::RWMol> molecule;
};
using GpuLigandSource = std::function<std::unique_ptr<GpuLigandInput>()>;

// A single host producer with bounded FIFO storage. The default reserves one
// preparation/ready slot, as in single-ligand overlap. Molecules and feature
// snapshots transfer ownership to the caller; the producer makes no CUDA calls.
// The byte bound covers estimated queued payload, not RSS or RDKit internals.
// One in-flight preparation may additionally wait for byte capacity. With
// overlap disabled, next() performs the same preparation synchronously.
class GpuLigandPrefetch {
public:
    GpuLigandPrefetch(const std::filesystem::path& input, const RunConfig& config,
                     std::size_t limit, bool queries_enabled, bool overlap,
                     bool collect_timings, bool trace,
                     std::size_t capacity = 1, std::size_t max_ready_bytes = 0);
    // Called only by the preparation thread (or next() when synchronous).
    // Each input transfers exclusive molecule ownership, just like MOL2 parsing.
    GpuLigandPrefetch(GpuLigandSource source, const RunConfig& config,
                     std::size_t limit, bool queries_enabled, bool overlap,
                     bool collect_timings, bool trace,
                     std::size_t capacity = 1, std::size_t max_ready_bytes = 0);
    ~GpuLigandPrefetch();
    GpuLigandPrefetch(const GpuLigandPrefetch&) = delete;
    GpuLigandPrefetch& operator=(const GpuLigandPrefetch&) = delete;

    // Returns null at EOF/limit. CPU parse/preparation exceptions are carried in
    // the packet; file-reading/resource failures are rethrown after queued
    // packets drain. A packet larger than the byte bound is a fatal error.
    // consumer_wait_s measures only waiting for a pending packet, not its CPU
    // work in serial mode. Timing collection adds no CUDA synchronization.
    std::unique_ptr<GpuPreparedRecord> next(double* consumer_wait_s = nullptr);

    // Cancels queued work, wakes waiters and joins any active preparation before
    // releasing its buffers. It cannot interrupt a running RDKit/OpenMP call.
    void stop() noexcept;
    double producer_queue_wait_seconds() const;
    GpuPrefetchStatistics statistics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
