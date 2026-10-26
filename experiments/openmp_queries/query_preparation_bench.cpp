#include "cuda_triplet_index.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::size_t positive_integer(const char* value, const char* name) {
    const std::string_view text(value);
    std::size_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || result == 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer fitting in size_t");
    return result;
}

struct ScopedEnvironment {
    const char* name;
    bool had_previous;
    std::string previous;

    ScopedEnvironment(const char* variable, const char* value)
        : name(variable), had_previous(std::getenv(variable) != nullptr),
          previous(had_previous ? std::getenv(variable) : "") {
        if (::setenv(name, value, 1) != 0)
            throw std::runtime_error(std::string("Cannot set ") + name);
    }

    ~ScopedEnvironment() {
        if (had_previous) ::setenv(name, previous.c_str(), 1);
        else ::unsetenv(name);
    }
};

std::vector<Triplet> make_triplets(std::size_t count) {
    // Give every row distinct, packable area indices to expose any reordering.
    if (count > std::numeric_limits<std::uint32_t>::max() / 3)
        throw std::invalid_argument("triplet_count is too large for unique uint32_t matching-area indices");
    const auto hydrophobic = InteractionKey::Hydrophobic;
    const double edge = 1.5;
    const std::vector<Triplet> seeds{
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2, 2, 2}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2, 2, 3}},
        {{19, 2, 47}, {InteractionKey::Aromatic, hydrophobic,
                      InteractionKey::HydrogenDonor}, {2, 3, 4}},
        {{19, 2, 47}, {hydrophobic, InteractionKey::Aromatic, hydrophobic}, {1.25, 1.75, 2}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2.01, 2.02, 2.03}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {0.01, 0.02, 0.03}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic},
                     {std::nextafter(edge, 0.0), edge,
                      std::nextafter(edge, std::numeric_limits<double>::infinity())}}
    };
    std::vector<Triplet> triplets;
    triplets.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Triplet triplet = seeds[i % seeds.size()];
        triplet.matching_area_indices = {3 * i + 2, 3 * i, 3 * i + 1};
        triplets.push_back(triplet);
    }
    return triplets;
}

void verify_outputs(const std::vector<GpuTripletKey>& expected_queries,
                    const std::vector<GpuLigandTripletOrders>& expected_orders,
                    const std::vector<GpuTripletKey>& queries,
                    const std::vector<GpuLigandTripletOrders>& orders) {
    if (queries.size() != expected_queries.size() || orders.size() != expected_orders.size())
        throw std::runtime_error("Query output sizes differ from the serial reference");
    for (std::size_t i = 0; i < queries.size(); ++i) {
        if (queries[i].padding != expected_queries[i].padding)
            throw std::runtime_error("Packed query padding differs from the serial reference");
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            if (queries[i].keys[vertex] != expected_queries[i].keys[vertex] ||
                queries[i].distance_bins[vertex] != expected_queries[i].distance_bins[vertex])
                throw std::runtime_error("Packed query key, bin or source order differs from the serial reference");
        }
        if (orders[i].equivalent_order_count != expected_orders[i].equivalent_order_count)
            throw std::runtime_error("Equivalent-order count differs from the serial reference");
        for (std::size_t order = 0; order < 6; ++order) {
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                if (orders[i].equivalent_orders[order][vertex] !=
                    expected_orders[i].equivalent_orders[order][vertex])
                    throw std::runtime_error("Equivalent-order content, order or unused slots differ from the serial reference");
            }
        }
    }
}

std::uint64_t checksum(const std::vector<GpuTripletKey>& queries,
                       const std::vector<GpuLigandTripletOrders>& orders) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    for (std::size_t i = 0; i < queries.size(); ++i) {
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            mix(queries[i].keys[vertex]);
            mix(queries[i].distance_bins[vertex]);
        }
        mix(queries[i].padding);
        mix(orders[i].equivalent_order_count);
        for (std::size_t order = 0; order < 6; ++order)
            for (std::size_t vertex = 0; vertex < 3; ++vertex)
                mix(orders[i].equivalent_orders[order][vertex]);
    }
    return hash;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 3)
            throw std::invalid_argument("Usage: query_preparation_bench [triplet_count=16384] [repetitions=20]");
        const std::size_t triplet_count = argc > 1 ? positive_integer(argv[1], "triplet_count") : 16384;
        const std::size_t repetitions = argc > 2 ? positive_integer(argv[2], "repetitions") : 20;
        const char* selected_method = std::getenv("DOCKING_QUERY_METHOD");
        const std::string method = selected_method ? selected_method : "reference";
        if (method != "reference" && method != "optimized")
            throw std::invalid_argument("DOCKING_QUERY_METHOD must be reference or optimized");
        const int requested_threads = gpu_query_requested_threads();
        const auto triplets = make_triplets(triplet_count);
        constexpr double bin_size = 0.5;

        std::vector<GpuTripletKey> reference_queries;
        std::vector<GpuLigandTripletOrders> reference_orders;
        {
            const ScopedEnvironment reference_method("DOCKING_QUERY_METHOD", "reference");
            const ScopedEnvironment serial_threads("DOCKING_QUERY_THREADS", "1");
            make_gpu_ligand_triplet_queries(triplets, bin_size, reference_queries, reference_orders);
        }

        std::vector<GpuTripletKey> queries;
        std::vector<GpuLigandTripletOrders> orders;
        GpuQueryPreparationInfo info;
        for (int warmup = 0; warmup < 2; ++warmup)
            make_gpu_ligand_triplet_queries(triplets, bin_size, queries, orders, &info);
        verify_outputs(reference_queries, reference_orders, queries, orders);

        double total_ms = 0;
        double min_ms = std::numeric_limits<double>::infinity();
        double max_ms = 0;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            make_gpu_ligand_triplet_queries(triplets, bin_size, queries, orders, &info);
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
            total_ms += elapsed_ms;
            min_ms = std::min(min_ms, elapsed_ms);
            max_ms = std::max(max_ms, elapsed_ms);
        }
        verify_outputs(reference_queries, reference_orders, queries, orders);
        if (info.requested_threads != requested_threads || info.used_threads < 1 ||
            info.used_threads > requested_threads)
            throw std::runtime_error("Query preparation reported an invalid thread count");

        std::cout << "method,triplets,repetitions,requested_threads,used_threads,openmp_available,"
                     "parallel_min_triplets,min_triplets_per_thread,mean_ms,min_ms,max_ms,checksum\n"
                  << method << ',' << triplet_count << ',' << repetitions << ','
                  << info.requested_threads << ',' << info.used_threads << ','
                  << (gpu_query_openmp_available() ? 1 : 0) << ',' << gpu_query_parallel_min_triplets << ','
                  << gpu_query_min_per_thread << ','
                  << std::fixed << std::setprecision(6) << total_ms / static_cast<double>(repetitions) << ','
                  << min_ms << ',' << max_ms << ',' << checksum(queries, orders) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Query preparation benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
