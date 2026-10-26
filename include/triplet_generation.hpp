#ifndef TRIPLET_GENERATION_HPP
#define TRIPLET_GENERATION_HPP

#include "matching_areas.hpp"
#include "matching_grid.hpp"
#include "pose_targets.hpp"

#include <array>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <unordered_map>


struct Triplet
{
    std::array<std::size_t, 3> matching_area_indices;
    std::array<InteractionKey, 3> keys;
    // Edge order: (0, 1), (0, 2), (1, 2).
    std::array<double, 3> distances;
};

// Emit triplets in input index order (i < j < k), with distinct source areas.
// max_distance applies to all three edges; equality is accepted.
// Reject edges shorter than 0.001 A and squared cross-product norm below 1e-6 A^4.
std::vector<Triplet> generate_triplets(
    const std::vector<MatchingArea>& areas,
    double max_distance
);

// Separate experimental counter, not a replacement for generate_triplets().
// The grid must describe these same areas; its fixed 27-cell neighborhood
// requires grid.cell_size >= max_distance.
std::size_t count_triplets_spatial(
    const std::vector<MatchingArea>& areas,
    const MatchingGrid& grid,
    double max_distance
);

struct CanonicalTriplet
{
    std::array<std::size_t, 3> matching_area_indices{};

    std::array<InteractionKey, 3> keys{};

    std::array<double, 3> distances{};

    std::array<std::size_t, 3> distance_bins{};

    // All vertex orders with the same canonical signature, up to six.
    std::array<
        std::array<std::size_t, 3>,
        6
    > equivalent_orders{};

    std::uint8_t equivalent_order_count = 0;
};


// Choose the lexicographically smallest chemical-key/bin signature among six
// vertex orders; retain every tied order in permutation traversal order.
// Exact distances do not break signature ties. Bins truncate distance/bin_size
// to size_t; callers supply positive bin_size and valid nonnegative distances.
CanonicalTriplet canonicalize_triplet(
    const Triplet& triplet,
    double bin_size
);

struct TripletKey
{
    std::array<InteractionKey, 3> keys{};
    std::array<std::size_t, 3> distance_bins{};

    bool operator==(
        const TripletKey& other
    ) const = default;
};


struct TripletKeyHash
{
    std::size_t operator()(
        const TripletKey& key
    ) const;
};


TripletKey make_triplet_key(
    const CanonicalTriplet& triplet
);

struct ProteinTripletRecord
{
    // Canonical vertex order; indices refer to the original PoseTarget vector.
    std::array<std::uint32_t, 3> indices{};
};


using ProteinTripletHashTable =
    std::unordered_map<
        TripletKey,
        std::vector<ProteinTripletRecord>,
        TripletKeyHash
    >;

ProteinTripletRecord make_protein_triplet_record(
    const CanonicalTriplet& triplet
);

// Applies the real max_distance cutoff before binning, with the same geometric
// and physical-area exclusions as generate_triplets(). Target indices must fit uint32_t.
ProteinTripletHashTable build_protein_triplet_hash_table(
    const std::vector<PoseTarget>& targets,
    double max_distance,
    double bin_size
);

struct FlatTripletIndex
{
    // Lexicographic order: chemical keys first, then distance bins.
    std::vector<TripletKey> keys;

    // offsets.size() == keys.size() + 1; bucket i is [offsets[i], offsets[i+1]).
    // Records within each bucket retain the hash table bucket's insertion order.
    std::vector<std::size_t> offsets;

    std::vector<ProteinTripletRecord> records;
};

struct FlatTripletBucket
{
    std::size_t begin{};
    std::size_t end{};
};


// Both lookups visit +/-1 distance bins with d0 outermost and d2 innermost.
// Decrements from zero are skipped; chemical keys are not changed or reordered.
// Bin neighbors are not recanonicalized or filtered by exact edge distances.
std::vector<FlatTripletBucket>
find_neighbor_triplet_buckets_flat(
    const FlatTripletIndex& index,
    const CanonicalTriplet& ligand_triplet
);


// Sort keys, preserving each bucket's record order and candidate multiplicity.
FlatTripletIndex flatten_protein_triplet_index(
    const ProteinTripletHashTable& table
);

// Same neighborhood/order as the flat lookup; returned buckets borrow table data.
std::vector<const std::vector<ProteinTripletRecord>*>
find_neighbor_triplet_buckets(
    const ProteinTripletHashTable& table,
    const CanonicalTriplet& ligand_triplet
);

bool validate_flat_triplet_index(
    const ProteinTripletHashTable& table,
    const FlatTripletIndex& flat_index
);

#endif
