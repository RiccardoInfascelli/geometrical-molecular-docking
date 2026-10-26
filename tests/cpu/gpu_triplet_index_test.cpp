#include "cuda_triplet_index.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    try {
        test_query_preparation();
    } catch (const std::exception& error) {
        std::cerr << "Ligand query preparation failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Reference and optimized ligand query preparation passed\n";

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
