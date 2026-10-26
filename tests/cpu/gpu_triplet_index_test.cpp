#include "cuda_triplet_index.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct ScopedQueryMethod
{
    bool had_previous;
    std::string previous;

    explicit ScopedQueryMethod(const char* method)
        : had_previous(std::getenv("DOCKING_QUERY_METHOD") != nullptr),
          previous(had_previous ? std::getenv("DOCKING_QUERY_METHOD") : "")
    {
        require((method ? setenv("DOCKING_QUERY_METHOD", method, 1)
                        : unsetenv("DOCKING_QUERY_METHOD")) == 0,
                "Cannot set query preparation method");
    }

    ~ScopedQueryMethod()
    {
        if (had_previous) setenv("DOCKING_QUERY_METHOD", previous.c_str(), 1);
        else unsetenv("DOCKING_QUERY_METHOD");
    }
};

struct ScopedQueryThreads
{
    bool had_previous;
    std::string previous;

    explicit ScopedQueryThreads(const char* threads)
        : had_previous(std::getenv("DOCKING_QUERY_THREADS") != nullptr),
          previous(had_previous ? std::getenv("DOCKING_QUERY_THREADS") : "")
    {
        require((threads ? setenv("DOCKING_QUERY_THREADS", threads, 1)
                         : unsetenv("DOCKING_QUERY_THREADS")) == 0,
                "Cannot set query preparation threads");
    }

    ~ScopedQueryThreads()
    {
        if (had_previous) setenv("DOCKING_QUERY_THREADS", previous.c_str(), 1);
        else unsetenv("DOCKING_QUERY_THREADS");
    }
};

void check_query_outputs_equal(const std::vector<GpuTripletKey>& expected_queries,
                               const std::vector<GpuLigandTripletOrders>& expected_orders,
                               const std::vector<GpuTripletKey>& queries,
                               const std::vector<GpuLigandTripletOrders>& orders)
{
    require(queries.size() == expected_queries.size() && orders.size() == expected_orders.size(),
            "Query preparation changed output sizes or exception prefixes");
    for (std::size_t i = 0; i < queries.size(); ++i) {
        require(queries[i].padding == expected_queries[i].padding,
                "Query preparation changed packed padding");
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            require(queries[i].keys[vertex] == expected_queries[i].keys[vertex] &&
                    queries[i].distance_bins[vertex] == expected_queries[i].distance_bins[vertex],
                    "Query preparation changed a key, bin or source order");
        }
    }
    for (std::size_t i = 0; i < orders.size(); ++i) {
        require(orders[i].equivalent_order_count == expected_orders[i].equivalent_order_count,
                "Query preparation changed equivalent-order counts");
        for (std::size_t order = 0; order < 6; ++order) {
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                require(orders[i].equivalent_orders[order][vertex] ==
                            expected_orders[i].equivalent_orders[order][vertex],
                        "Query preparation changed equivalent-order content, order or unused slots");
            }
        }
    }
}

void check_packed_queries(const std::vector<Triplet>& triplets, double bin_size,
                         const std::vector<GpuTripletKey>& queries,
                         const std::vector<GpuLigandTripletOrders>& orders)
{
    require(queries.size() == triplets.size() && orders.size() == triplets.size(),
            "Query preparation changed the number or multiplicity of triplets");
    for (std::size_t i = 0; i < triplets.size(); ++i) {
        // The unchanged CPU canonicalizer is the oracle, independent of the
        // optimized path that permutes bins already computed for the three edges.
        const auto canonical = canonicalize_triplet(triplets[i], bin_size);
        require(queries[i].padding == 0, "Nonzero packed query padding");
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            require(queries[i].keys[vertex] ==
                        static_cast<std::uint8_t>(canonical.keys[vertex]) &&
                    queries[i].distance_bins[vertex] == canonical.distance_bins[vertex],
                    "Query signature or query order differs from CPU canonicalization");
        }
        require(orders[i].equivalent_order_count == canonical.equivalent_order_count,
                "Equivalent-order count differs from CPU canonicalization");
        for (std::size_t order = 0; order < 6; ++order) {
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                require(orders[i].equivalent_orders[order][vertex] ==
                            canonical.equivalent_orders[order][vertex],
                        "Equivalent-order content or unused zero slot differs");
            }
        }
    }
}

void compare_query_methods(const std::vector<Triplet>& triplets, double bin_size)
{
    std::vector<GpuTripletKey> reference_queries;
    std::vector<GpuLigandTripletOrders> reference_orders;
    for (const char* method : {static_cast<const char*>(nullptr), "reference", "optimized"}) {
        const ScopedQueryMethod selection(method);
        std::vector<GpuTripletKey> queries(2);
        std::vector<GpuLigandTripletOrders> orders(3);
        queries[0].padding = 99;
        orders[0].equivalent_order_count = 99;

        // Each call replaces both outputs, including when they are nonempty.
        for (int repeat = 0; repeat < 2; ++repeat) {
            make_gpu_ligand_triplet_queries(triplets, bin_size, queries, orders);
            check_packed_queries(triplets, bin_size, queries, orders);
        }
        if (!method) {
            reference_queries = queries;
            reference_orders = orders;
        } else {
            require(queries.size() == reference_queries.size() &&
                    orders.size() == reference_orders.size(),
                    "Query methods produced different vector sizes");
            for (std::size_t i = 0; i < queries.size(); ++i) {
                require(queries[i].padding == reference_queries[i].padding,
                        "Query methods produced different padding");
                for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                    require(queries[i].keys[vertex] == reference_queries[i].keys[vertex] &&
                            queries[i].distance_bins[vertex] ==
                                reference_queries[i].distance_bins[vertex],
                            "Query methods changed a key, bin or vector order");
                }
                require(orders[i].equivalent_order_count ==
                            reference_orders[i].equivalent_order_count,
                        "Query methods produced different equivalent-order counts");
                for (std::size_t order = 0; order < 6; ++order) {
                    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                        require(orders[i].equivalent_orders[order][vertex] ==
                                    reference_orders[i].equivalent_orders[order][vertex],
                                "Query methods changed equivalent-order content or order");
                    }
                }
            }
        }
        make_gpu_ligand_triplet_queries({}, bin_size, queries, orders);
        require(queries.empty() && orders.empty(), "Empty input did not clear query outputs");
    }
}

void check_known_orders(const Triplet& triplet,
                        const std::vector<std::array<std::uint32_t, 3>>& expected)
{
    for (const char* method : {"reference", "optimized"}) {
        const ScopedQueryMethod selection(method);
        std::vector<GpuTripletKey> queries;
        std::vector<GpuLigandTripletOrders> orders;
        make_gpu_ligand_triplet_queries({triplet}, 0.5, queries, orders);
        require(orders.size() == 1 && orders[0].equivalent_order_count == expected.size(),
                "Known symmetric triplet has an unexpected number of orders");
        for (std::size_t order = 0; order < expected.size(); ++order) {
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                require(orders[0].equivalent_orders[order][vertex] == expected[order][vertex],
                        "Known symmetric triplet changed permutation order");
            }
        }
    }
}

void test_query_preparation()
{
    const auto hydrophobic = InteractionKey::Hydrophobic;
    const Triplet equilateral{{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2, 2, 2}};
    Triplet isosceles = equilateral;
    isosceles.distances = {2, 2, 3};
    const Triplet distinct{{19, 2, 47},
        {InteractionKey::Aromatic, hydrophobic, InteractionKey::HydrogenDonor}, {2, 3, 4}};

    check_known_orders(equilateral,
        {{19, 2, 47}, {19, 47, 2}, {2, 19, 47}, {2, 47, 19}, {47, 19, 2}, {47, 2, 19}});
    check_known_orders(isosceles, {{19, 2, 47}, {19, 47, 2}});
    check_known_orders(distinct, {{2, 47, 19}});
    compare_query_methods({}, 0.5);
    compare_query_methods({equilateral}, 0.5);
    compare_query_methods({distinct, isosceles, distinct, equilateral}, 0.5);

    Triplet boundary = equilateral;
    boundary.distances = {0.01, 0.02, 0.03};
    compare_query_methods({boundary}, 0.5); // All distances lie in bin zero.
    boundary.distances = {2.01, 2.02, 2.03};
    compare_query_methods({boundary}, 0.5); // Different lengths, one binned signature.
    for (const double bin_size : {0.5, 0.1, 0.3}) {
        const double edge = 3.0 * bin_size;
        boundary.distances = {std::nextafter(edge, 0.0), edge,
                             std::nextafter(edge, std::numeric_limits<double>::infinity())};
        compare_query_methods({boundary, distinct, boundary}, bin_size);
    }

    constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}}, {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}
    }};
    const std::array<InteractionKey, 3> key_types{
        InteractionKey::Hydrophobic, InteractionKey::HydrogenDonor, InteractionKey::Aromatic
    };
    // Exercise repeated chemical keys and every input vertex ordering. These are
    // valid packing inputs; geometric triangle rejection belongs to generation.
    for (const auto first : key_types) {
        for (const auto second : key_types) {
            for (const auto third : key_types) {
                const Triplet source{{19, 2, 47}, {first, second, third}, {1.25, 1.75, 2.0}};
                std::vector<Triplet> triplets;
                const auto distance = [&source](std::size_t a, std::size_t b) {
                    if ((a == 0 && b == 1) || (a == 1 && b == 0)) return source.distances[0];
                    if ((a == 0 && b == 2) || (a == 2 && b == 0)) return source.distances[1];
                    return source.distances[2];
                };
                for (const auto& order : permutations) {
                    triplets.push_back({
                        {source.matching_area_indices[order[0]], source.matching_area_indices[order[1]],
                         source.matching_area_indices[order[2]]},
                        {source.keys[order[0]], source.keys[order[1]], source.keys[order[2]]},
                        {distance(order[0], order[1]), distance(order[0], order[2]),
                         distance(order[1], order[2])}
                    });
                }
                compare_query_methods(triplets, 0.5);
            }
        }
    }

    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    Triplet packed_limit = distinct;
    packed_limit.matching_area_indices = {maximum, maximum - 1, maximum - 2};
    packed_limit.distances = {static_cast<double>(maximum), 2, 3};
    compare_query_methods({packed_limit}, 1.0);

    for (const char* method : {"reference", "optimized"}) {
        const ScopedQueryMethod selection(method);
        const auto expect_overflow = [](const Triplet& triplet) {
            std::vector<GpuTripletKey> queries;
            std::vector<GpuLigandTripletOrders> orders;
            bool threw = false;
            try {
                make_gpu_ligand_triplet_queries({triplet}, 1.0, queries, orders);
            } catch (const std::overflow_error&) {
                threw = true;
            }
            require(threw, "Packed query overflow was not rejected");
        };
        Triplet invalid = distinct;
        invalid.keys[0] = static_cast<InteractionKey>(256);
        expect_overflow(invalid);
        if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
            // Keep double-to-size_t conversion defined; only uint32_t packing overflows.
            invalid = distinct;
            invalid.distances[0] = static_cast<double>(maximum) + 1.0;
            expect_overflow(invalid);
            invalid = distinct;
            invalid.matching_area_indices[0] = static_cast<std::size_t>(maximum) + 1;
            expect_overflow(invalid);
        }
    }
    const ScopedQueryMethod invalid_method("unknown");
    std::vector<GpuTripletKey> queries(1);
    std::vector<GpuLigandTripletOrders> orders(1);
    queries[0].padding = 99;
    queries[0].distance_bins[0] = 123;
    orders[0].equivalent_order_count = 99;
    orders[0].equivalent_orders[0][0] = 456;
    bool threw = false;
    try {
        make_gpu_ligand_triplet_queries({distinct}, 0.5, queries, orders);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Unknown query preparation method was not rejected");
    require(queries.size() == 1 && orders.size() == 1 &&
            queries[0].padding == 99 && queries[0].distance_bins[0] == 123 &&
            orders[0].equivalent_order_count == 99 &&
            orders[0].equivalent_orders[0][0] == 456,
            "Invalid method changed query outputs before throwing");
}

std::vector<Triplet> make_parallel_query_fixture()
{
    const auto hydrophobic = InteractionKey::Hydrophobic;
    const double edge = 1.5;
    const std::vector<Triplet> seeds{
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2, 2, 2}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2, 2, 3}},
        {{19, 2, 47}, {InteractionKey::Aromatic, hydrophobic,
                      InteractionKey::HydrogenDonor}, {2, 3, 4}},
        {{19, 2, 47}, {hydrophobic, InteractionKey::Aromatic, hydrophobic},
                     {1.25, 1.75, 2}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {2.01, 2.02, 2.03}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic}, {0.01, 0.02, 0.03}},
        {{19, 2, 47}, {hydrophobic, hydrophobic, hydrophobic},
                     {std::nextafter(edge, 0.0), edge,
                      std::nextafter(edge, std::numeric_limits<double>::infinity())}}
    };
    std::vector<Triplet> triplets;
    triplets.reserve(8 * gpu_query_parallel_min_triplets + 37);
    for (std::size_t i = 0; i < 8 * gpu_query_parallel_min_triplets + 37; ++i) {
        Triplet triplet = seeds[i % seeds.size()];
        // Unique indices detect reordered source rows even for equal signatures.
        triplet.matching_area_indices = {3 * i + 2, 3 * i, 3 * i + 1};
        triplets.push_back(triplet);
    }
    return triplets;
}

int available_query_team_size(int requested_threads)
{
    int used_threads = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads) shared(used_threads)
    {
#pragma omp single
        used_threads = omp_get_num_threads();
    }
#else
    (void)requested_threads;
#endif
    return used_threads;
}

void test_query_thread_selectors(const Triplet& triplet)
{
    const ScopedQueryMethod method("reference");
    {
        const ScopedQueryThreads threads(nullptr);
        require(gpu_query_requested_threads() == 1,
                "Unset query thread selector did not default to one");
        GpuQueryPreparationInfo info;
        require(info.requested_threads == 1 && info.used_threads == 1,
                "Query preparation information did not default to serial");
        info = {-1, -1};
        std::vector<GpuTripletKey> queries;
        std::vector<GpuLigandTripletOrders> orders;
        make_gpu_ligand_triplet_queries({triplet}, 0.5, queries, orders, &info);
        require(info.requested_threads == 1 && info.used_threads == 1,
                "Default query preparation did not report serial execution");
    }

    const auto expect_rejected = [&triplet](const char* value, bool unsupported) {
        const ScopedQueryThreads threads(value);
        const auto check_message = [unsupported](const std::invalid_argument& error) {
            const std::string message(error.what());
            require(message.find("DOCKING_QUERY_THREADS") != std::string::npos,
                    "Query thread error did not identify the selector");
            require(!unsupported || message.find("OpenMP") != std::string::npos,
                    "Unsupported query thread request did not explain the OpenMP requirement");
        };
        bool threw = false;
        try {
            (void)gpu_query_requested_threads();
        } catch (const std::invalid_argument& error) {
            check_message(error);
            threw = true;
        }
        require(threw, "Invalid query thread selector was accepted by the parser");

        for (const char* choice : {static_cast<const char*>(nullptr), "reference", "optimized"}) {
            const ScopedQueryMethod method(choice);
            for (const auto& input : {std::vector<Triplet>{}, std::vector<Triplet>{triplet}}) {
                std::vector<GpuTripletKey> queries(2);
                std::vector<GpuLigandTripletOrders> orders(3);
                queries[0].padding = 99;
                queries[1].distance_bins[2] = 123;
                orders[0].equivalent_order_count = 99;
                orders[2].equivalent_orders[5][2] = 456;
                const auto previous_queries = queries;
                const auto previous_orders = orders;
                threw = false;
                try {
                    make_gpu_ligand_triplet_queries(input, 0.5, queries, orders);
                } catch (const std::invalid_argument& error) {
                    check_message(error);
                    threw = true;
                }
                require(threw, "Invalid query thread selector was accepted by preparation");
                check_query_outputs_equal(previous_queries, previous_orders, queries, orders);
            }
        }
    };

    for (const char* value : {"", "0", "-1", "+1", " 2", "2 ", "abc", "2x", "1.5",
                              "999999999999999999999999999999999999"}) {
        expect_rejected(value, false);
    }
    if (!gpu_query_openmp_available()) {
        for (const char* value : {"2", "4", "8"}) {
            expect_rejected(value, true);
        }
    }
}

void test_parallel_query_preparation()
{
#ifdef _OPENMP
    struct ScopedDynamicTeams
    {
        int previous = omp_get_dynamic();
        ScopedDynamicTeams() { omp_set_dynamic(0); }
        ~ScopedDynamicTeams() { omp_set_dynamic(previous); }
    } dynamic_teams;
    require(gpu_query_openmp_available(), "OpenMP build did not report query OpenMP support");
#else
    require(!gpu_query_openmp_available(), "Serial build unexpectedly reported query OpenMP support");
#endif
    const auto triplets = make_parallel_query_fixture();
    test_query_thread_selectors(triplets.front());

    std::vector<GpuTripletKey> reference_queries;
    std::vector<GpuLigandTripletOrders> reference_orders;
    {
        const ScopedQueryThreads threads("1");
        const ScopedQueryMethod method("reference");
        make_gpu_ligand_triplet_queries(triplets, 0.5, reference_queries, reference_orders);
        check_packed_queries(triplets, 0.5, reference_queries, reference_orders);
    }

    for (const char* value : {"1", "2", "4", "8"}) {
        const int requested_threads = std::stoi(value);
        if (!gpu_query_openmp_available() && requested_threads > 1) continue;
        const ScopedQueryThreads threads(value);
        require(gpu_query_requested_threads() == requested_threads,
                "Query thread selector was cached or parsed incorrectly");
        const int available_threads = available_query_team_size(requested_threads);
        for (const char* choice : {static_cast<const char*>(nullptr), "reference", "optimized"}) {
            const ScopedQueryMethod method(choice);
            std::vector<GpuTripletKey> queries(2);
            std::vector<GpuLigandTripletOrders> orders(3);
            for (int repeat = 0; repeat < 2; ++repeat) {
                GpuQueryPreparationInfo info{-1, -1};
                make_gpu_ligand_triplet_queries(triplets, 0.5, queries, orders, &info);
                require(info.requested_threads == requested_threads &&
                        info.used_threads >= 1 && info.used_threads <= requested_threads,
                        "Query preparation reported an invalid request or actual team size");
                require(available_threads <= 1 || info.used_threads > 1,
                        "Large query preparation stayed serial despite an available OpenMP team");
                check_query_outputs_equal(reference_queries, reference_orders, queries, orders);
            }
            // Test empty, single-row and just-below-threshold serial fallbacks.
            for (const auto size : {std::size_t{0}, std::size_t{1},
                                    gpu_query_parallel_min_triplets - 1}) {
                const std::vector<Triplet> small(triplets.begin(), triplets.begin() + size);
                GpuQueryPreparationInfo info{-1, -1};
                make_gpu_ligand_triplet_queries(small, 0.5, queries, orders, &info);
                require(info.requested_threads == requested_threads && info.used_threads == 1,
                        "Small query preparation did not report its serial fallback");
                check_packed_queries(small, 0.5, queries, orders);
            }
            if (requested_threads > 2) {
                const std::vector<Triplet> light(triplets.begin(),
                    triplets.begin() + gpu_query_parallel_min_triplets);
                GpuQueryPreparationInfo info;
                make_gpu_ligand_triplet_queries(light, 0.5, queries, orders, &info);
                require(info.used_threads == 1, "Insufficient work per thread must stay serial");
                check_packed_queries(light, 0.5, queries, orders);
            }
        }
    }
#ifdef _OPENMP
    // Both active and serialized enclosing regions must avoid nested teams.
    const ScopedQueryThreads nested_threads("4");
    for (const int enclosing_threads : {1, 2}) {
        for (const char* choice : {"reference", "optimized"}) {
            const ScopedQueryMethod method(choice);
            std::vector<GpuTripletKey> queries;
            std::vector<GpuLigandTripletOrders> orders;
            GpuQueryPreparationInfo info{-1, -1};
            std::exception_ptr failure;
#pragma omp parallel num_threads(enclosing_threads) shared(queries, orders, info, failure)
            {
#pragma omp single
                {
                    try {
                        make_gpu_ligand_triplet_queries(triplets, 0.5, queries, orders, &info);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                }
            }
            if (failure) std::rethrow_exception(failure);
            require(info.requested_threads == 4 && info.used_threads == 1,
                    "Query preparation did not use its serial fallback inside an OpenMP region");
            check_query_outputs_equal(reference_queries, reference_orders, queries, orders);
        }
    }
#endif
}

void test_parallel_query_overflow_prefixes()
{
    const auto triplets = make_parallel_query_fixture();
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    std::vector<Triplet> failures;
    Triplet invalid = triplets.front();
    invalid.keys[0] = static_cast<InteractionKey>(256);
    failures.push_back(invalid);
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        invalid = triplets.front();
        invalid.distances[0] = (static_cast<double>(maximum) + 1.0) * 0.5;
        failures.push_back(invalid);
        invalid = triplets.front();
        invalid.matching_area_indices[0] = static_cast<std::size_t>(maximum) + 1;
        failures.push_back(invalid);
    }

    struct OverflowResult
    {
        std::vector<GpuTripletKey> queries;
        std::vector<GpuLigandTripletOrders> orders;
        std::string message;
    };
    const auto capture_overflow = [](const std::vector<Triplet>& input) {
        OverflowResult result;
        result.queries.resize(2);
        result.orders.resize(3);
        result.queries[0].padding = 99;
        result.orders[0].equivalent_order_count = 99;
        try {
            make_gpu_ligand_triplet_queries(input, 0.5, result.queries, result.orders);
        } catch (const std::overflow_error& error) {
            result.message = error.what();
        }
        require(!result.message.empty(), "Invalid packed query did not throw overflow_error");
        return result;
    };

    for (std::size_t failure = 0; failure < failures.size(); ++failure) {
        for (const auto first_bad : {std::size_t{0}, gpu_query_parallel_min_triplets + 31}) {
            auto input = triplets;
            input[first_bad] = failures[failure];
            // Different workers can fail with different messages. Source order
            // must choose the first exception, independent of completion order.
            input[input.size() - 2] = failures[(failure + 1) % failures.size()];
            OverflowResult expected;
            {
                const ScopedQueryThreads threads("1");
                const ScopedQueryMethod method("reference");
                expected = capture_overflow(input);
            }
            const bool index_overflow = failure == 2;
            require(expected.orders.size() == first_bad &&
                    expected.queries.size() == first_bad + (index_overflow ? 1 : 0),
                    "Serial overflow did not retain the expected output prefix");
            for (const char* value : {"1", "2", "4", "8"}) {
                if (!gpu_query_openmp_available() && std::stoi(value) > 1) continue;
                const ScopedQueryThreads threads(value);
                for (const char* choice : {"reference", "optimized"}) {
                    const ScopedQueryMethod method(choice);
                    const auto actual = capture_overflow(input);
                    require(actual.message == expected.message,
                            "Query preparation did not rethrow the first source-index overflow");
                    check_query_outputs_equal(expected.queries, expected.orders,
                                              actual.queries, actual.orders);
                }
            }
        }
    }
}

bool test_cpu_lookup()
{
    const std::array<InteractionKey, 3> chemical_keys{
        InteractionKey::Hydrophobic,
        InteractionKey::HydrogenDonor,
        InteractionKey::Aromatic
    };
    ProteinTripletHashTable table;

    // Insert in reverse bin order; flattening must produce sorted keys.
    for (int i = 3; i >= 0; --i) {
        for (int j = 3; j >= 0; --j) {
            for (int k = 3; k >= 0; --k) {
                const TripletKey key{
                    chemical_keys,
                    {static_cast<std::size_t>(i), static_cast<std::size_t>(j),
                     static_cast<std::size_t>(k)}
                };
                const auto id = static_cast<std::uint32_t>(100 * i + 10 * j + k);
                table[key].push_back({{id, id + 1, id + 2}});
            }
        }
    }

    // Preserve record order within a bucket, and exclude different chemistry.
    table[TripletKey{chemical_keys, {2, 2, 2}}].push_back({{1000, 1001, 1002}});
    const std::array<InteractionKey, 3> other_keys{
        InteractionKey::Hydrophobic,
        InteractionKey::HydrogenAcceptor,
        InteractionKey::Aromatic
    };
    table[TripletKey{other_keys, {0, 0, 0}}].push_back({{2000, 2001, 2002}});

    const auto flat = flatten_protein_triplet_index(table);
    if (!validate_flat_triplet_index(table, flat) || flat.keys.size() != 65 ||
        flat.keys.front() != TripletKey{chemical_keys, {0, 0, 0}} ||
        flat.keys[1] != TripletKey{chemical_keys, {0, 0, 1}} ||
        flat.keys[4] != TripletKey{chemical_keys, {0, 1, 0}} ||
        flat.keys[16] != TripletKey{chemical_keys, {1, 0, 0}} ||
        flat.keys[63] != TripletKey{chemical_keys, {3, 3, 3}} ||
        flat.keys.back() != TripletKey{other_keys, {0, 0, 0}}) {
        std::cerr << "Unexpected CPU triplet index ordering\n";
        return false;
    }

    const auto check_lookup = [&table, &flat](
        const CanonicalTriplet& query,
        const std::vector<std::uint32_t>& expected)
    {
        std::vector<std::uint32_t> hash_records;
        for (const auto* bucket : find_neighbor_triplet_buckets(table, query)) {
            for (const auto& record : *bucket) {
                hash_records.push_back(record.indices[0]);
            }
        }
        std::vector<std::uint32_t> flat_records;
        for (const auto& bucket : find_neighbor_triplet_buckets_flat(flat, query)) {
            for (std::size_t i = bucket.begin; i < bucket.end; ++i) {
                flat_records.push_back(flat.records[i].indices[0]);
            }
        }
        return hash_records == expected && flat_records == expected;
    };

    CanonicalTriplet query;
    query.keys = chemical_keys;
    query.distance_bins = {2, 2, 2};
    if (!check_lookup(query, {
            111, 112, 113, 121, 122, 123, 131, 132, 133,
            211, 212, 213, 221, 222, 1000, 223, 231, 232, 233,
            311, 312, 313, 321, 322, 323, 331, 332, 333})) {
        std::cerr << "CPU 27-neighbor record order mismatch\n";
        return false;
    }
    query.distance_bins = {0, 0, 0};
    if (!check_lookup(query, {0, 1, 10, 11, 100, 101, 110, 111})) {
        std::cerr << "CPU zero-bin neighbor lookup mismatch\n";
        return false;
    }
    query.distance_bins = {9, 9, 9};
    if (!check_lookup(query, {})) {
        std::cerr << "CPU lookup returned records outside the neighborhood\n";
        return false;
    }
    query.keys = {InteractionKey::Metal, InteractionKey::Metal, InteractionKey::Metal};
    query.distance_bins = {2, 2, 2};
    if (!check_lookup(query, {})) {
        std::cerr << "CPU lookup returned incompatible chemical keys\n";
        return false;
    }

    const ProteinTripletHashTable empty_table;
    const auto empty_flat = flatten_protein_triplet_index(empty_table);
    if (!validate_flat_triplet_index(empty_table, empty_flat) ||
        !find_neighbor_triplet_buckets(empty_table, query).empty() ||
        !find_neighbor_triplet_buckets_flat(empty_flat, query).empty()) {
        std::cerr << "CPU empty triplet index lookup failed\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    const ScopedQueryThreads serial_threads("1");
    try {
        test_query_preparation();
        test_parallel_query_preparation();
        test_parallel_query_overflow_prefixes();
    } catch (const std::exception& error) {
        std::cerr << "Ligand query preparation failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Reference and optimized ligand query preparation passed\n";
    std::cout << "Query thread selection, parallel equivalence and overflow prefixes passed\n";

    if (!test_cpu_lookup()) {
        return 1;
    }
    std::cout << "CPU triplet index and neighbor lookup passed\n";

    FlatTripletIndex cpu_index;

    TripletKey key{};

    key.keys = {
        InteractionKey::Hydrophobic,
        InteractionKey::HydrogenDonor,
        InteractionKey::Aromatic
    };

    key.distance_bins = {
        4,
        7,
        10
    };

    cpu_index.keys.push_back(key);

    cpu_index.offsets = {
        0,
        2
    };

    cpu_index.records.push_back({
        {1, 2, 3}
    });

    cpu_index.records.push_back({
        {4, 5, 6}
    });

    const auto gpu_index =
        make_gpu_flat_triplet_index(
            cpu_index
        );

    if (
        !validate_gpu_flat_triplet_index(
            cpu_index,
            gpu_index
        )
    ) {
        std::cerr
            << "GPU triplet index conversion failed\n";

        return 1;
    }

    std::cout
        << "GPU triplet index conversion passed\n"
        << "keys: " << gpu_index.keys.size() << '\n'
        << "offsets: " << gpu_index.offsets.size() << '\n'
        << "records: " << gpu_index.records.size() << '\n';

    return 0;
}
