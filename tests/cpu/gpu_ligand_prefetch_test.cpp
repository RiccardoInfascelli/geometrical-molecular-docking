#include "gpu_ligand_prefetch.hpp"
#include "molecule_io.hpp"

#include <GraphMol/Conformer.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

template<class Predicate>
void wait_until(Predicate ready, const char* reason) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error(reason);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() / ("gpu-prefetch-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { require(fs::create_directory(path), "Cannot create fresh test directory"); }
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

std::string glycerol(const char* name) {
    return std::string("@<TRIPOS>MOLECULE\n") + name +
        "\n6 5 0 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0\n"
        "2 C2 1.5 0.0 0.0 C.3 1 LIG 0.0\n"
        "3 C3 2.8 0.6 0.0 C.3 1 LIG 0.0\n"
        "4 O1 -0.4 1.3 0.2 O.3 1 LIG 0.0\n"
        "5 O2 1.5 -1.3 0.2 O.3 1 LIG 0.0\n"
        "6 O3 3.5 1.7 -0.2 O.3 1 LIG 0.0\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 2 3 1\n3 1 4 1\n4 2 5 1\n5 3 6 1\n";
}

void write_input(const fs::path& path, const std::string& input) {
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << input;
    output.close();
}

void compare(const GpuPreparedRecord& a, const GpuPreparedRecord& b) {
    require(a.index == b.index && a.name == b.name, "Prefetch changed record identity or order");
    require(bool(a.error) == bool(b.error), "Prefetch changed the preparation outcome");
    if (a.error) {
        const auto message = [](std::exception_ptr error) {
            try { std::rethrow_exception(error); }
            catch (const std::exception& exception) { return std::string(exception.what()); }
        };
        require(message(a.error) == message(b.error) && std::string(a.error_stage) == b.error_stage,
                "Prefetch changed exception text or stage");
        return;
    }
    const auto& x = a.prepared;
    const auto& y = b.prepared;
    require(x.molecule && y.molecule && x.molecule.get() != y.molecule.get(),
            "Packets did not retain independent molecule ownership");
    require(x.molecule->getNumAtoms() == y.molecule->getNumAtoms() &&
            x.ligand.diameter == y.ligand.diameter && x.queries.size() == y.queries.size() &&
            x.orders.size() == y.orders.size() && x.ligand.areas.size() == y.ligand.areas.size() &&
            x.ligand.matching_areas.size() == y.ligand.matching_areas.size() &&
            x.ligand.triplets.size() == y.ligand.triplets.size(),
            "Prefetch changed scientific preparation");
    for (std::size_t i = 0; i < x.ligand.matching_areas.size(); ++i) {
        const auto& p = x.ligand.matching_areas[i];
        const auto& q = y.ligand.matching_areas[i];
        require(p.source_area_index == q.source_area_index && p.key == q.key &&
                p.centroid.x == q.centroid.x && p.centroid.y == q.centroid.y &&
                p.centroid.z == q.centroid.z && p.interaction_radius == q.interaction_radius,
                "Prefetch changed matching areas or their order");
    }
    for (std::size_t i = 0; i < x.ligand.triplets.size(); ++i) {
        const auto& p = x.ligand.triplets[i];
        const auto& q = y.ligand.triplets[i];
        require(p.matching_area_indices == q.matching_area_indices && p.keys == q.keys &&
                p.distances == q.distances, "Prefetch changed triplets or their order");
    }
    require(x.query_info.requested_threads == y.query_info.requested_threads &&
            x.query_info.used_threads == y.query_info.used_threads,
            "Prefetch changed query worker selection");
    for (unsigned int i = 0; i < x.molecule->getNumAtoms(); ++i) {
        const auto& p = x.molecule->getConformer().getAtomPos(i);
        const auto& q = y.molecule->getConformer().getAtomPos(i);
        require(p.x == q.x && p.y == q.y && p.z == q.z, "Prefetch changed ligand coordinates");
    }
    for (std::size_t i = 0; i < x.queries.size(); ++i) {
        require(x.queries[i].padding == y.queries[i].padding &&
                x.orders[i].equivalent_order_count == y.orders[i].equivalent_order_count,
                "Prefetch changed query padding or equivalent-order multiplicity");
        for (int vertex = 0; vertex < 3; ++vertex) {
            require(x.queries[i].keys[vertex] == y.queries[i].keys[vertex] &&
                    x.queries[i].distance_bins[vertex] == y.queries[i].distance_bins[vertex],
                    "Prefetch changed packed query keys/bins/order");
            for (int order = 0; order < 6; ++order)
                require(x.orders[i].equivalent_orders[order][vertex] ==
                        y.orders[i].equivalent_orders[order][vertex],
                        "Prefetch changed equivalent-order content");
        }
    }
}

void test_order_errors_and_owned_queries(const fs::path& directory) {
    const auto input = directory / "ordered.mol2";
    write_input(input, glycerol("first") +
        "@<TRIPOS>MOLECULE\nbroken\nnot-a-valid-molecule\n" + glycerol("third"));
    const RunConfig config;
    GpuLigandPrefetch serial(input, config, 3, true, false, true, false);
    std::vector<std::unique_ptr<GpuPreparedRecord>> reference;
    for (std::size_t index = 1; index <= 3; ++index) {
        double wait = -1;
        reference.push_back(serial.next(&wait));
        require(wait == 0.0, "Serial preparation was mislabeled as consumer wait");
    }
    require(!serial.next(), "Serial input limit changed");
    GpuLigandPrefetch overlap(input, config, 3, true, true, true, false);
    std::unique_ptr<GpuPreparedRecord> previous;
    for (std::size_t index = 1; index <= 3; ++index) {
        const auto& a = reference.at(index - 1);
        double wait = -1;
        auto b = overlap.next(&wait);
        require(a && b && a->index == index && wait >= 0.0, "Record was skipped or duplicated");
        compare(*a, *b);
        require(bool(a->error) == (index == 2), "Synthetic valid/invalid fixture outcome differs");
        if (index != 2) {
            require(!a->prepared.queries.empty(), "Fixture did not exercise actual query preparation");
            require(a->prepared.query_end >= a->prepared.query_start,
                    "Query interval is inverted");
            require(a->prepared.query_preparation_seconds == std::chrono::duration<double>(
                        a->prepared.query_end - a->prepared.query_start).count() &&
                    b->prepared.query_preparation_seconds == std::chrono::duration<double>(
                        b->prepared.query_end - b->prepared.query_start).count(),
                    "Query time does not match the packet's completed host interval");
        }
        if (previous && previous->prepared.molecule)
            require(previous->prepared.molecule->getNumAtoms() == 6,
                    "Producer invalidated a molecule retained by the consumer");
        previous = std::move(b);
    }
    require(!overlap.next() && !overlap.next(), "EOF/limit is not stable");
    overlap.stop();
    overlap.stop();
    require(overlap.producer_queue_wait_seconds() >= 0.0, "Producer wait is invalid");
}

void test_empty_index_limit_and_cancel(const fs::path& directory) {
    const auto input = directory / "bounded.mol2";
    write_input(input, glycerol("one") + glycerol("two") + glycerol("three"));
    const RunConfig config;
    const auto empty = directory / "empty.mol2";
    write_input(empty, "");
    for (bool async : {false, true}) {
        GpuLigandPrefetch empty_input(empty, config, 3, true, async, true, false);
        require(!empty_input.next() && !empty_input.next(), "Empty input fabricated a packet or did not terminate");
        empty_input.stop();
        GpuLigandPrefetch limited(input, config, 1, false, async, false, false);
        auto packet = limited.next();
        require(packet && !packet->error && packet->prepared.molecule,
                "Empty-index preparation lost the molecule");
        require(packet->prepared.queries.empty() && packet->prepared.orders.empty(),
                "Empty index unnecessarily prepared queries");
        require(packet->prepared.query_preparation_seconds == 0.0 &&
                packet->prepared.preparation_seconds == 0.0,
                "Disabled timing collection changed");
        require(!limited.next(), "Prefetch exceeded its input-record limit");
    }
    {
        GpuLigandPrefetch pending(input, config, 3, true, true, true, false);
        // A ready record may remain unconsumed when the caller aborts. Destruction
        // must wake the producer's capacity wait instead of waiting for a pop.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        pending.stop();
        require(!pending.next(), "Cancelled prefetch delivered another packet");
    }
    bool missing = false;
    try {
        GpuLigandPrefetch invalid(directory / "missing.mol2", config, 1, true, true, true, false);
        invalid.next();
    } catch (const std::runtime_error&) { missing = true; }
    require(missing, "Producer file-open failure was not delivered to the consumer");
}

void test_failure_categories_and_disabled_timing(const fs::path& directory) {
    const auto input = directory / "errors.mol2";
    write_input(input, glycerol("one"));
    const RunConfig config;
    bool null_owner_rejected = false;
    try { (void)prepare_gpu_ligand(nullptr, config, true); }
    catch (const GpuQueryPreparationError&) { null_owner_rejected = true; }
    require(null_owner_rejected, "Missing molecule ownership lost its fatal preparation category");

    for (bool async : {false, true}) {
        bool zero_limit_rejected = false;
        try { GpuLigandPrefetch invalid(input, config, 0, true, async, true, false); }
        catch (const std::invalid_argument&) { zero_limit_rejected = true; }
        require(zero_limit_rejected, "Zero record limit was accepted");

        auto invalid_config = config;
        invalid_config.triplet_bin_size = std::numeric_limits<double>::quiet_NaN();
        GpuLigandPrefetch failed(input, invalid_config, 1, true, async, true, false);
        auto packet = failed.next();
        require(packet && packet->index == 1 && packet->error,
                "Preparation failure lost its input record");
        bool query_error_retained = false;
        try { std::rethrow_exception(packet->error); }
        catch (const GpuQueryPreparationError&) { query_error_retained = true; }
        require(query_error_retained,
                "Query preparation failure became a recoverable molecular error");
        failed.stop();

        GpuLigandPrefetch untimed(input, config, 1, true, async, false, false);
        double wait = -1.0;
        packet = untimed.next(&wait);
        require(packet && !packet->error && !packet->prepared.queries.empty(),
                "Untimed preparation did not generate queries");
        require(wait == 0.0 && packet->parse_seconds == 0.0 &&
                packet->prepared.preparation_seconds == 0.0 &&
                packet->prepared.query_preparation_seconds == 0.0 &&
                packet->prepared.query_start == std::chrono::steady_clock::time_point{} &&
                packet->prepared.query_end == std::chrono::steady_clock::time_point{},
                "Disabled timing collected host intervals");
        require(!untimed.next(), "Untimed preparation did not terminate");
    }
}

void test_bounded_fifo_and_cancel(const fs::path& directory) {
    const auto input = directory / "fifo.mol2";
    write_input(input, glycerol("one") + glycerol("two") +
        "@<TRIPOS>MOLECULE\nbroken\nnot-a-valid-molecule\n" +
        glycerol("four") + glycerol("five") + glycerol("six"));
    const RunConfig config;
    GpuLigandPrefetch serial(input, config, 6, true, false, true, false, 3);
    std::vector<std::unique_ptr<GpuPreparedRecord>> reference;
    while (auto packet = serial.next()) reference.push_back(std::move(packet));
    require(reference.size() == 6, "FIFO fixture did not contain six records");

    GpuLigandPrefetch queued(input, config, 6, true, true, true, false, 3);
    wait_until([&] { return queued.statistics().ready_records == 3; }, "Queue did not reach its count bound");
    const auto full = queued.statistics();
    require(full.capacity == 3 && full.max_ready_bytes == 0 && full.prepared_records == 3 &&
            full.high_water_records == 3, "Producer prepared beyond its reserved count capacity");
    std::vector<std::unique_ptr<GpuPreparedRecord>> retained;
    for (const auto& expected : reference) {
        auto packet = queued.next();
        require(packet && packet->estimated_bytes == expected->estimated_bytes,
                "Queued payload estimate or record count changed");
        compare(*packet, *expected);
        retained.push_back(std::move(packet));
        require(queued.statistics().ready_records <= 3, "Producer exceeded the FIFO count bound");
    }
    require(!queued.next() && !queued.next(), "FIFO drain did not finish");
    const auto drained = queued.statistics();
    require(drained.ready_records == 0 && drained.ready_bytes == 0 && drained.prepared_records == 6 &&
            retained.front()->prepared.molecule->getNumAtoms() == 6,
            "FIFO drain lost ownership or retained queue bytes");
    queued.stop();

    GpuLigandPrefetch cancelled(input, config, 6, true, true, true, false, 2);
    wait_until([&] { return cancelled.statistics().ready_records == 2; }, "Cancellation fixture did not fill");
    cancelled.stop();
    require(!cancelled.next() && cancelled.statistics().ready_records == 0 &&
            cancelled.statistics().ready_bytes == 0, "Full-queue cancellation did not release packets");
    for (std::size_t capacity : {std::size_t{0}, std::size_t{129}}) {
        bool rejected = false;
        try { GpuLigandPrefetch invalid(input, config, 1, true, true, false, false, capacity); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid FIFO capacity was accepted");
    }
}

void test_byte_backpressure_and_fatal_drain(const fs::path& directory) {
    const auto input = directory / "bytes.mol2";
    write_input(input, glycerol("same") + glycerol("same") + glycerol("same") + glycerol("same"));
    const RunConfig config;
    GpuLigandPrefetch serial(input, config, 1, true, false, false, false, 1,
                             std::numeric_limits<std::size_t>::max());
    auto reference = serial.next();
    require(reference && !reference->error && reference->estimated_bytes > sizeof(GpuPreparedRecord),
            "Payload estimate omitted prepared molecule/query storage");
    const auto budget = reference->estimated_bytes;
    GpuLigandPrefetch queued(input, config, 4, true, true, true, false, 4, budget);
    wait_until([&] { const auto stats = queued.statistics();
        return stats.ready_records == 1 && stats.prepared_records == 2; },
        "Producer did not wait for byte capacity with one in-flight packet");
    require(queued.statistics().ready_bytes == budget && queued.statistics().high_water_records == 1,
            "Byte bound admitted more than one packet");
    for (std::size_t index = 1; index <= 4; ++index) {
        auto packet = queued.next();
        require(packet && !packet->error && packet->index == index && packet->estimated_bytes == budget,
                "Byte backpressure lost or duplicated a packet");
        const auto stats = queued.statistics();
        require(stats.ready_bytes <= budget && stats.high_water_bytes <= budget && stats.ready_records <= 1,
                "Estimated queued payload exceeded the byte bound");
    }
    require(!queued.next(), "Byte-bounded FIFO did not finish");
    queued.stop();
    require(queued.producer_queue_wait_seconds() > 0.0, "Byte backpressure did not record producer waiting");

    GpuLigandPrefetch cancelled(input, config, 4, true, true, true, false, 4, budget);
    wait_until([&] { return cancelled.statistics().prepared_records == 2; }, "Byte cancellation did not reach waiting state");
    cancelled.stop();
    require(!cancelled.next(), "Byte-wait cancellation delivered another packet");

    const auto fatal_input = directory / "oversized.mol2";
    write_input(fatal_input, "@<TRIPOS>MOLECULE\nsmall-error\ninvalid\n" + glycerol("same"));
    GpuLigandPrefetch failed(fatal_input, config, 2, true, true, false, false, 3, budget - 1);
    wait_until([&] { return failed.statistics().prepared_records == 2; }, "Oversized fixture did not prepare its second record");
    auto first = failed.next();
    require(first && first->index == 1 && first->error, "Fatal producer error overtook the queued FIFO prefix");
    bool oversized = false;
    try { (void)failed.next(); }
    catch (const std::runtime_error& error) {
        const std::string message = error.what();
        oversized = message.find("record 2") != std::string::npos &&
            message.find("exceeds queue budget") != std::string::npos;
    }
    require(oversized, "Oversized packet was not reported as an explicit fatal resource error");
    failed.stop();
}

void test_in_memory_source() {
    const auto original = parse_ligand_mol2_block(glycerol("source"));
    const auto first_position = original->getConformer().getAtomPos(0);
    const auto make_source = [&]() -> GpuLigandSource {
        return [&, next = std::size_t{0}]() mutable -> std::unique_ptr<GpuLigandInput> {
            if (next == 3) return nullptr;
            auto input = std::make_unique<GpuLigandInput>();
            input->index = ++next;
            input->name = "conformer-" + std::to_string(next);
            input->molecule = std::make_unique<RDKit::RWMol>(*original);
            input->molecule->getConformer().getAtomPos(0).x += next;
            return input;
        };
    };
    const RunConfig config;
    GpuLigandPrefetch serial(make_source(), config, 3, true, false, true, false, 3);
    GpuLigandPrefetch queued(make_source(), config, 3, true, true, true, false, 3);
    std::vector<std::unique_ptr<GpuPreparedRecord>> retained;
    for (std::size_t index = 1; index <= 3; ++index) {
        auto a = serial.next();
        auto b = queued.next();
        require(a && b && !a->error && !b->error, "In-memory source did not prepare every conformer");
        compare(*a, *b);
        retained.push_back(std::move(b));
    }
    require(!serial.next() && !queued.next(), "In-memory source exceeded its input limit");
    require(original->getConformer().getAtomPos(0).x == first_position.x &&
            retained.front()->prepared.molecule->getConformer().getAtomPos(0).x == first_position.x + 1,
            "In-memory preparation modified a source or an earlier packet");
    GpuLigandSource fails = [&, index = std::size_t{0}]() mutable -> std::unique_ptr<GpuLigandInput> {
        if (index++) throw std::runtime_error("source failed");
        auto input = std::make_unique<GpuLigandInput>();
        input->index = 1;
        input->molecule = std::make_unique<RDKit::RWMol>(*original);
        return input;
    };
    GpuLigandPrefetch failing(std::move(fails), config, 3, true, true, false, false, 3);
    require(bool(failing.next()), "Source failure discarded its completed prefix");
    bool failure = false;
    try { (void)failing.next(); }
    catch (const std::runtime_error& error) { failure = std::string(error.what()) == "source failed"; }
    require(failure, "Source exception was lost or mistaken for EOF");
    bool rejected = false;
    try { GpuLigandPrefetch empty(GpuLigandSource{}, config, 1, true, false, false, false); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Empty molecule source was accepted");
}

void test_openmp_queries_from_producer_thread() {
    if (!gpu_query_openmp_available()) return;
    const auto key = InteractionKey::Hydrophobic;
    const std::size_t count = 8 * gpu_query_min_per_thread;
    std::vector<Triplet> triplets(count);
    for (std::size_t i = 0; i < count; ++i)
        triplets[i] = {{3 * i, 3 * i + 1, 3 * i + 2}, {key, key, key}, {2, 2, 3}};
    std::vector<GpuTripletKey> reference;
    std::vector<GpuLigandTripletOrders> reference_orders;
    make_gpu_ligand_triplet_queries(triplets, 0.5, reference, reference_orders);
    for (const int threads : {4, 8}) {
        const auto selector = std::to_string(threads);
        require(::setenv("DOCKING_QUERY_THREADS", selector.c_str(), 1) == 0,
                "Cannot freeze producer query threads");
        std::vector<GpuTripletKey> parallel;
        std::vector<GpuLigandTripletOrders> parallel_orders;
        GpuQueryPreparationInfo info;
        std::exception_ptr failure;
        std::thread producer([&] {
            try { make_gpu_ligand_triplet_queries(triplets, 0.5, parallel, parallel_orders, &info); }
            catch (...) { failure = std::current_exception(); }
        });
        producer.join();
        require(::setenv("DOCKING_QUERY_THREADS", "1", 1) == 0, "Cannot restore query selector");
        if (failure) std::rethrow_exception(failure);
        require(parallel.size() == reference.size() && parallel_orders.size() == reference_orders.size() &&
                info.requested_threads == threads && info.used_threads >= 1 && info.used_threads <= threads,
                "OpenMP preparation from a non-main thread failed");
        for (std::size_t i = 0; i < count; ++i) {
            require(parallel[i].padding == reference[i].padding &&
                    parallel_orders[i].equivalent_order_count == reference_orders[i].equivalent_order_count,
                    "Producer OpenMP query multiplicity differs");
            for (int vertex = 0; vertex < 3; ++vertex) {
                require(parallel[i].keys[vertex] == reference[i].keys[vertex] &&
                        parallel[i].distance_bins[vertex] == reference[i].distance_bins[vertex],
                        "Producer OpenMP query key/bin differs");
                for (int order = 0; order < 6; ++order)
                    require(parallel_orders[i].equivalent_orders[order][vertex] ==
                            reference_orders[i].equivalent_orders[order][vertex],
                            "Producer OpenMP equivalent order differs");
            }
        }
        std::cout << "OpenMP producer query team: " << info.used_threads
                  << " (requested " << threads << ")\n";
    }
}
}

int main() {
    try {
        // Freeze all process selectors before starting a worker, as in the CLI.
        if (::setenv("DOCKING_QUERY_THREADS", "1", 1) ||
            ::setenv("DOCKING_QUERY_METHOD", "reference", 1))
            throw std::runtime_error("Cannot set synthetic query selectors");
        const TemporaryDirectory directory;
        test_order_errors_and_owned_queries(directory.path);
        test_empty_index_limit_and_cancel(directory.path);
        test_failure_categories_and_disabled_timing(directory.path);
        test_bounded_fifo_and_cancel(directory.path);
        test_byte_backpressure_and_fatal_drain(directory.path);
        test_in_memory_source();
        test_openmp_queries_from_producer_thread();
        std::cout << "GPU ligand prefetch tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
