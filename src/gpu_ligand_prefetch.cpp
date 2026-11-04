#include "gpu_ligand_prefetch.hpp"

#include "molecule_io.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::size_t add_bytes(std::size_t current, std::size_t count, std::size_t width) {
    if (width && count > (std::numeric_limits<std::size_t>::max() - current) / width)
        throw std::overflow_error("Prefetch payload size exceeds addressable capacity");
    return current + count * width;
}
std::size_t estimated_bytes(const GpuPreparedRecord& packet) {
    const auto& prepared = packet.prepared;
    std::size_t bytes = add_bytes(sizeof(packet), packet.name.capacity(), sizeof(char));
    bytes = add_bytes(bytes, prepared.queries.capacity(), sizeof(GpuTripletKey));
    bytes = add_bytes(bytes, prepared.orders.capacity(), sizeof(GpuLigandTripletOrders));
    bytes = add_bytes(bytes, prepared.ligand.areas.capacity(), sizeof(interaction::interaction_area));
    bytes = add_bytes(bytes, prepared.ligand.matching_areas.capacity(), sizeof(MatchingArea));
    bytes = add_bytes(bytes, prepared.ligand.triplets.capacity(), sizeof(Triplet));
    if (prepared.molecule) {
        const auto& molecule = *prepared.molecule;
        bytes = add_bytes(bytes, 1, sizeof(RDKit::RWMol));
        bytes = add_bytes(bytes, molecule.getNumAtoms(), sizeof(RDKit::Atom));
        bytes = add_bytes(bytes, molecule.getNumBonds(), sizeof(RDKit::Bond));
        bytes = add_bytes(bytes, molecule.getNumConformers(), sizeof(RDKit::Conformer));
        const auto coordinates = add_bytes(0, molecule.getNumAtoms(), sizeof(RDGeom::Point3D));
        bytes = add_bytes(bytes, molecule.getNumConformers(), coordinates);
    }
    return bytes;
}
}

struct GpuLigandPrefetch::Impl {
    const std::filesystem::path input;
    const RunConfig config;
    const std::size_t limit;
    const std::size_t capacity, max_ready_bytes;
    const bool queries_enabled, overlap, timed, trace;
    std::unique_ptr<Mol2Reader> reader;
    GpuLigandSource source;
    std::size_t read_count = 0;
    mutable std::mutex mutex;
    std::condition_variable available, empty;
    std::deque<std::unique_ptr<GpuPreparedRecord>> ready;
    GpuPrefetchStatistics stats;
    std::exception_ptr fatal;
    bool cancelled = false, done = false;
    double producer_wait_s = 0.0;
    std::thread producer;

    Impl(const std::filesystem::path& path, const RunConfig& run_config,
         std::size_t record_limit, bool make_queries, bool async,
         bool collect_timings, bool collect_trace, std::size_t queue_capacity,
         std::size_t byte_limit)
        : input(path), config(run_config), limit(record_limit),
          capacity(queue_capacity), max_ready_bytes(byte_limit),
          queries_enabled(make_queries), overlap(async), timed(collect_timings),
          trace(collect_trace) {
        if (limit == 0) throw std::invalid_argument("Prefetch record limit must be positive");
        if (capacity == 0 || capacity > 128)
            throw std::invalid_argument("Prefetch capacity must be in [1, 128]");
        stats.capacity = capacity;
        stats.max_ready_bytes = max_ready_bytes;
    }

    // Called by exactly one thread. Reader and each mutable molecule stay local
    // to that thread until the finished packet transfers ownership.
    std::unique_ptr<GpuPreparedRecord> prepare_next() {
        if (read_count == limit) return nullptr;
        auto packet = std::make_unique<GpuPreparedRecord>();
        std::string block;
        std::unique_ptr<RDKit::RWMol> molecule;
        const auto parse_start = timed ? Clock::now() : Clock::time_point{};
        if (source) {
            auto next = source();
            if (!next) return nullptr;
            packet->index = next->index;
            packet->name = std::move(next->name);
            molecule = std::move(next->molecule);
        } else {
            if (!reader) reader = std::make_unique<Mol2Reader>(input);
            if (!reader->next(block, packet->name)) return nullptr;
            packet->index = reader->index();
        }
        ++read_count;
        try {
            if (!source) molecule = parse_ligand_mol2_block(block);
            if (timed) packet->parse_seconds = elapsed(parse_start);
            packet->error_stage = "preparation/search/ranking";
            packet->prepared = prepare_gpu_ligand(std::move(molecule), config,
                                                 queries_enabled, timed, trace);
        } catch (const std::bad_alloc&) {
            throw;
        } catch (...) {
            packet->error = std::current_exception();
        }
        if (capacity > 1 || max_ready_bytes)
            packet->estimated_bytes = estimated_bytes(*packet);
        return packet;
    }

    void check_packet_size(const GpuPreparedRecord& packet) const {
        if (max_ready_bytes && packet.estimated_bytes > max_ready_bytes)
            throw std::runtime_error("Prefetch record " + std::to_string(packet.index) +
                " estimated payload " + std::to_string(packet.estimated_bytes) +
                " bytes exceeds queue budget " + std::to_string(max_ready_bytes) + " bytes");
    }

    void run() noexcept {
        try {
            for (;;) {
                {
                    std::unique_lock lock(mutex);
                    // Reserve count capacity before preparing. In particular,
                    // the legacy one-slot queue cannot prepare a second packet.
                    if (ready.size() == capacity && !cancelled) {
                        const auto start = timed ? Clock::now() : Clock::time_point{};
                        empty.wait(lock, [&] { return cancelled || ready.size() < capacity; });
                        if (timed) producer_wait_s += elapsed(start);
                    }
                    if (cancelled) break;
                }
                auto packet = prepare_next();
                {
                    std::unique_lock lock(mutex);
                    if (cancelled) break;
                    if (!packet) {
                        done = true;
                        available.notify_all();
                        return;
                    }
                    ++stats.prepared_records;
                    check_packet_size(*packet);
                    if (max_ready_bytes && packet->estimated_bytes > max_ready_bytes - stats.ready_bytes) {
                        const auto start = timed ? Clock::now() : Clock::time_point{};
                        empty.wait(lock, [&] { return cancelled ||
                            packet->estimated_bytes <= max_ready_bytes - stats.ready_bytes; });
                        if (timed) producer_wait_s += elapsed(start);
                    }
                    if (cancelled) break;
                    const auto bytes = add_bytes(stats.ready_bytes, packet->estimated_bytes, 1);
                    ready.push_back(std::move(packet));
                    stats.ready_records = ready.size();
                    stats.ready_bytes = bytes;
                    stats.high_water_records = std::max(stats.high_water_records, ready.size());
                    stats.high_water_bytes = std::max(stats.high_water_bytes, bytes);
                }
                available.notify_one();
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            fatal = std::current_exception();
        }
        {
            std::lock_guard lock(mutex);
            done = true;
        }
        available.notify_all();
    }
};

GpuLigandPrefetch::GpuLigandPrefetch(
    const std::filesystem::path& input, const RunConfig& config,
    std::size_t limit, bool queries_enabled, bool overlap,
    bool collect_timings, bool trace, std::size_t capacity, std::size_t max_ready_bytes)
    : impl_(std::make_unique<Impl>(input, config, limit, queries_enabled,
                                   overlap, collect_timings, trace, capacity, max_ready_bytes)) {
    if (overlap) impl_->producer = std::thread([state = impl_.get()] { state->run(); });
}

GpuLigandPrefetch::~GpuLigandPrefetch() { stop(); }

GpuLigandPrefetch::GpuLigandPrefetch(
    GpuLigandSource source, const RunConfig& config, std::size_t limit,
    bool queries_enabled, bool overlap, bool collect_timings, bool trace,
    std::size_t capacity, std::size_t max_ready_bytes)
    : impl_(std::make_unique<Impl>(std::filesystem::path{}, config, limit, queries_enabled,
                                   overlap, collect_timings, trace, capacity, max_ready_bytes)) {
    if (!source) throw std::invalid_argument("Prefetch molecule source must be callable");
    impl_->source = std::move(source);
    if (overlap) impl_->producer = std::thread([state = impl_.get()] { state->run(); });
}

std::unique_ptr<GpuPreparedRecord> GpuLigandPrefetch::next(double* consumer_wait_s) {
    if (consumer_wait_s) *consumer_wait_s = 0.0;
    if (!impl_->overlap) {
        if (impl_->cancelled || impl_->done) return nullptr;
        auto packet = impl_->prepare_next();
        if (!packet) impl_->done = true;
        else {
            std::lock_guard lock(impl_->mutex);
            ++impl_->stats.prepared_records;
            impl_->check_packet_size(*packet);
        }
        return packet;
    }
    std::unique_lock lock(impl_->mutex);
    if (impl_->ready.empty() && !impl_->done && !impl_->cancelled) {
        const auto start = impl_->timed ? Clock::now() : Clock::time_point{};
        impl_->available.wait(lock, [&] { return impl_->cancelled || !impl_->ready.empty() || impl_->done; });
        if (consumer_wait_s && impl_->timed) *consumer_wait_s = elapsed(start);
    }
    if (impl_->cancelled) return nullptr;
    if (impl_->ready.empty()) {
        if (impl_->fatal) std::rethrow_exception(impl_->fatal);
        return nullptr;
    }
    auto packet = std::move(impl_->ready.front());
    impl_->ready.pop_front();
    impl_->stats.ready_records = impl_->ready.size();
    impl_->stats.ready_bytes -= packet->estimated_bytes;
    lock.unlock();
    impl_->empty.notify_one();
    return packet;
}

void GpuLigandPrefetch::stop() noexcept {
    if (!impl_) return;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->cancelled = true;
    }
    impl_->available.notify_all();
    impl_->empty.notify_all();
    if (impl_->producer.joinable()) impl_->producer.join();
    // Join happens before destroying a packet or its reader. No producer can
    // still refer to a discarded molecule or to this queue's synchronization.
    {
        std::lock_guard lock(impl_->mutex);
        impl_->ready.clear();
        impl_->stats.ready_records = 0;
        impl_->stats.ready_bytes = 0;
    }
    impl_->reader.reset();
}

double GpuLigandPrefetch::producer_queue_wait_seconds() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->producer_wait_s;
}

GpuPrefetchStatistics GpuLigandPrefetch::statistics() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}
