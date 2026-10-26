#include "cuda_triplet_index.hpp"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#ifdef DOCKING_ENABLE_OPENMP
#include <omp.h>
#endif


namespace
{

GpuTripletKey make_gpu_triplet_key(
    const TripletKey& key
)
{
    GpuTripletKey gpu_key{};


    for (
        std::size_t i = 0;
        i < 3;
        ++i
    ) {
        const auto interaction_value =
            static_cast<unsigned int>(
                key.keys[i]
            );


        if (
            interaction_value >
            std::numeric_limits<
                std::uint8_t
            >::max()
        ) {
            throw std::overflow_error(
                "InteractionKey does not fit in uint8_t"
            );
        }


        if (
            key.distance_bins[i] >
            std::numeric_limits<
                std::uint32_t
            >::max()
        ) {
            throw std::overflow_error(
                "Triplet distance bin does not fit in uint32_t"
            );
        }


        gpu_key.keys[i] =
            static_cast<std::uint8_t>(
                interaction_value
            );

        gpu_key.distance_bins[i] =
            static_cast<std::uint32_t>(
                key.distance_bins[i]
            );
    }


    return gpu_key;
}

} // namespace


GpuFlatTripletIndexHost
make_gpu_flat_triplet_index(
    const FlatTripletIndex& index
)
{
    GpuFlatTripletIndexHost result;


    result.keys.reserve(
        index.keys.size()
    );

    result.offsets.reserve(
        index.offsets.size()
    );

    result.records.reserve(
        index.records.size()
    );


    for (const auto& key :
         index.keys) {

        result.keys.push_back(
            make_gpu_triplet_key(
                key
            )
        );
    }


    for (const auto offset :
         index.offsets) {

        if (
            offset >
            std::numeric_limits<
                std::uint32_t
            >::max()
        ) {
            throw std::overflow_error(
                "Triplet index offset does not fit in uint32_t"
            );
        }


        result.offsets.push_back(
            static_cast<std::uint32_t>(
                offset
            )
        );
    }


    for (const auto& record :
         index.records) {

        GpuProteinTripletRecord gpu_record{};


        for (
            std::size_t i = 0;
            i < 3;
            ++i
        ) {
            gpu_record.indices[i] =
                record.indices[i];
        }


        result.records.push_back(
            gpu_record
        );
    }


    return result;
}


bool validate_gpu_flat_triplet_index(
    const FlatTripletIndex& cpu_index,
    const GpuFlatTripletIndexHost& gpu_index
)
{
    if (
        cpu_index.keys.size() !=
            gpu_index.keys.size()
        ||
        cpu_index.offsets.size() !=
            gpu_index.offsets.size()
        ||
        cpu_index.records.size() !=
            gpu_index.records.size()
    ) {
        return false;
    }


    for (
        std::size_t i = 0;
        i < cpu_index.keys.size();
        ++i
    ) {
        for (
            std::size_t j = 0;
            j < 3;
            ++j
        ) {
            if (
                gpu_index.keys[i].keys[j] !=
                static_cast<std::uint8_t>(
                    cpu_index.keys[i].keys[j]
                )
            ) {
                return false;
            }


            if (
                gpu_index.keys[i]
                    .distance_bins[j] !=
                cpu_index.keys[i]
                    .distance_bins[j]
            ) {
                return false;
            }
        }
    }


    for (
        std::size_t i = 0;
        i < cpu_index.offsets.size();
        ++i
    ) {
        if (
            gpu_index.offsets[i] !=
            cpu_index.offsets[i]
        ) {
            return false;
        }
    }


    for (
        std::size_t i = 0;
        i < cpu_index.records.size();
        ++i
    ) {
        for (
            std::size_t j = 0;
            j < 3;
            ++j
        ) {
            if (
                gpu_index.records[i]
                    .indices[j] !=
                cpu_index.records[i]
                    .indices[j]
            ) {
                return false;
            }
        }
    }


    return true;
}


namespace
{

template <typename StoreKey>
GpuLigandTripletOrders make_gpu_ligand_triplet_query_reference(
    const Triplet& triplet,
    double bin_size,
    const StoreKey& store_key
)
{
    const auto canonical =
        canonicalize_triplet(
            triplet,
            bin_size
        );


    store_key(
        make_gpu_triplet_key(
            make_triplet_key(
                canonical
            )
        )
    );


    if (
        canonical.equivalent_order_count >
        6
    ) {
        throw std::runtime_error(
            "Too many equivalent ligand triplet orders"
        );
    }


    GpuLigandTripletOrders gpu_orders{};

    gpu_orders.equivalent_order_count =
        static_cast<std::uint32_t>(
            canonical.equivalent_order_count
        );


    for (
        std::size_t order_index = 0;
        order_index <
            canonical.equivalent_order_count;
        ++order_index
    ) {
        for (
            std::size_t vertex = 0;
            vertex < 3;
            ++vertex
        ) {
            const auto area_index =
                canonical
                    .equivalent_orders[
                        order_index
                    ][vertex];


            if (
                area_index >
                std::numeric_limits<
                    std::uint32_t
                >::max()
            ) {
                throw std::overflow_error(
                    "Ligand matching-area index does not fit in uint32_t"
                );
            }


            gpu_orders
                .equivalent_orders[
                    order_index
                ][vertex] =
                    static_cast<std::uint32_t>(
                        area_index
                    );
        }
    }

    return gpu_orders;
}


template <typename StoreKey>
GpuLigandTripletOrders make_gpu_ligand_triplet_query_optimized(
    const Triplet& triplet,
    double bin_size,
    const StoreKey& store_key
)
{
    constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}
    }};

    struct Signature
    {
        std::array<InteractionKey, 3> keys{};
        std::array<std::size_t, 3> bins{};
    };

    const auto edge_index = [](std::size_t a, std::size_t b) -> std::size_t {
        if (a > b)
            std::swap(a, b);
        if (a == 0 && b == 1)
            return 0;
        if (a == 0 && b == 2)
            return 1;
        return 2;
    };

    // Permuting vertices only reorders these edges; their bins are unchanged.
    const std::array<std::size_t, 3> base_bins{{
        static_cast<std::size_t>(triplet.distances[0] / bin_size),
        static_cast<std::size_t>(triplet.distances[1] / bin_size),
        static_cast<std::size_t>(triplet.distances[2] / bin_size)
    }};

    std::array<Signature, 6> signatures{};
    std::size_t best_index = 0;

    for (std::size_t permutation_index = 0;
         permutation_index < permutations.size(); ++permutation_index) {
        const auto& order = permutations[permutation_index];
        auto& signature = signatures[permutation_index];
        signature.keys = {{
            triplet.keys[order[0]],
            triplet.keys[order[1]],
            triplet.keys[order[2]]
        }};
        signature.bins = {{
            base_bins[edge_index(order[0], order[1])],
            base_bins[edge_index(order[0], order[2])],
            base_bins[edge_index(order[1], order[2])]
        }};

        const auto& best = signatures[best_index];
        if (permutation_index != 0 &&
            (signature.keys < best.keys ||
             (signature.keys == best.keys && signature.bins < best.bins))) {
            best_index = permutation_index;
        }
    }

    const auto& best = signatures[best_index];
    store_key(make_gpu_triplet_key({best.keys, best.bins}));

    GpuLigandTripletOrders gpu_orders{};
    // Use canonicalize_triplet's permutation order, including every tie.
    for (std::size_t permutation_index = 0;
         permutation_index < permutations.size(); ++permutation_index) {
        const auto& signature = signatures[permutation_index];
        if (signature.keys != best.keys || signature.bins != best.bins)
            continue;

        const std::size_t order_index = gpu_orders.equivalent_order_count;
        if (order_index >= 6) {
            throw std::runtime_error(
                "Too many equivalent ligand triplet orders"
            );
        }

        const auto& order = permutations[permutation_index];
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            const auto area_index = triplet.matching_area_indices[order[vertex]];
            if (area_index > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Ligand matching-area index does not fit in uint32_t"
                );
            }
            gpu_orders.equivalent_orders[order_index][vertex] =
                static_cast<std::uint32_t>(area_index);
        }
        ++gpu_orders.equivalent_order_count;
    }

    return gpu_orders;
}

template <typename PrepareTriplet>
void make_gpu_ligand_triplet_queries_impl(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders,
    int requested_threads,
    GpuQueryPreparationInfo* info,
    const PrepareTriplet& prepare_triplet
)
{
    queries.clear();
    orders.clear();
    queries.reserve(triplets.size());
    orders.reserve(triplets.size());

#ifdef DOCKING_ENABLE_OPENMP
    if (requested_threads > 1 &&
        triplets.size() >= gpu_query_parallel_min_triplets &&
        triplets.size() / static_cast<std::size_t>(requested_threads) >= gpu_query_min_per_thread &&
        omp_get_level() == 0) {
        std::exception_ptr failure;
        std::size_t failed_index = triplets.size();
        bool failed_query_written = false;
        queries.resize(triplets.size());
        orders.resize(triplets.size());
        int used_threads = 1;

        // Each iteration owns one output slot. Only failures enter the critical
        // section; no per-query error buffer or serial scan is needed.
#pragma omp parallel num_threads(requested_threads)
        {
#pragma omp single
            used_threads = omp_get_num_threads();

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < triplets.size(); ++i) {
                bool query_written = false;
                try {
                    orders[i] = prepare_triplet(
                        triplets[i], bin_size,
                        [&](const GpuTripletKey& key) {
                            queries[i] = key;
                            query_written = true;
                        }
                    );
                } catch (...) {
#pragma omp critical(docking_query_failure)
                    {
                        if (i < failed_index) {
                            failed_index = i;
                            failed_query_written = query_written;
                            failure = std::current_exception();
                        }
                    }
                }
            }
        }

        if (info)
            info->used_threads = used_threads;

        if (failure) {
            queries.resize(failed_index + (failed_query_written ? 1 : 0));
            orders.resize(failed_index);
            std::rethrow_exception(failure);
        }
        return;
    }
#else
    (void)requested_threads;
    (void)info;
#endif

    // Keep the serial append order: a key may be present when its order packing
    // throws. The same per-triplet builders supply the indexed parallel path.
    for (const auto& triplet : triplets) {
        orders.push_back(prepare_triplet(
            triplet, bin_size,
            [&](const GpuTripletKey& key) { queries.push_back(key); }
        ));
    }
}

} // namespace


bool gpu_query_openmp_available()
{
#ifdef DOCKING_ENABLE_OPENMP
    return true;
#else
    return false;
#endif
}


int gpu_query_requested_threads()
{
    const char* value = std::getenv("DOCKING_QUERY_THREADS");
    if (!value)
        return 1;

    const std::string_view text(value);
    int requested_threads = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), requested_threads
    );
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        requested_threads < 1) {
        throw std::invalid_argument(
            "DOCKING_QUERY_THREADS must be a positive integer fitting in int"
        );
    }
    if (requested_threads > 1 && !gpu_query_openmp_available()) {
        throw std::invalid_argument(
            "DOCKING_QUERY_THREADS > 1 requires an OpenMP build "
            "(configure with ENABLE_OPENMP=ON)"
        );
    }
    return requested_threads;
}


void make_gpu_ligand_triplet_queries(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders,
    GpuQueryPreparationInfo* info
)
{
    const char* choice = std::getenv("DOCKING_QUERY_METHOD");
    const bool reference = !choice || std::string_view(choice) == "reference";
    if (!reference && std::string_view(choice) != "optimized") {
        throw std::invalid_argument(
            "DOCKING_QUERY_METHOD must be reference or optimized"
        );
    }

    const int requested_threads = gpu_query_requested_threads();
    if (info)
        *info = {requested_threads, 1};

    if (reference) {
        make_gpu_ligand_triplet_queries_impl(
            triplets, bin_size, queries, orders, requested_threads, info,
            [](const Triplet& triplet, double size, const auto& store_key) {
                return make_gpu_ligand_triplet_query_reference(
                    triplet, size, store_key
                );
            }
        );
    } else {
        make_gpu_ligand_triplet_queries_impl(
            triplets, bin_size, queries, orders, requested_threads, info,
            [](const Triplet& triplet, double size, const auto& store_key) {
                return make_gpu_ligand_triplet_query_optimized(
                    triplet, size, store_key
                );
            }
        );
    }
}
