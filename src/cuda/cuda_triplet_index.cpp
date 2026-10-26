#include "cuda_triplet_index.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>


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

void make_gpu_ligand_triplet_queries_reference(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
)
{
    queries.clear();
    orders.clear();


    queries.reserve(
        triplets.size()
    );

    orders.reserve(
        triplets.size()
    );


    for (const auto& triplet :
         triplets) {

        const auto canonical =
            canonicalize_triplet(
                triplet,
                bin_size
            );


        queries.push_back(
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


        orders.push_back(
            gpu_orders
        );
    }
}


void make_gpu_ligand_triplet_queries_optimized(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
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

    queries.clear();
    orders.clear();
    queries.reserve(triplets.size());
    orders.reserve(triplets.size());

    for (const auto& triplet : triplets) {
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
        queries.push_back(make_gpu_triplet_key({best.keys, best.bins}));

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

        orders.push_back(gpu_orders);
    }
}

} // namespace


void make_gpu_ligand_triplet_queries(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
)
{
    const char* choice = std::getenv("DOCKING_QUERY_METHOD");
    if (!choice || std::string_view(choice) == "reference") {
        make_gpu_ligand_triplet_queries_reference(triplets, bin_size, queries, orders);
    } else if (std::string_view(choice) == "optimized") {
        make_gpu_ligand_triplet_queries_optimized(triplets, bin_size, queries, orders);
    } else {
        throw std::invalid_argument(
            "DOCKING_QUERY_METHOD must be reference or optimized"
        );
    }
}
