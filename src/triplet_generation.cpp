#include "triplet_generation.hpp"

#include <interaction/geometric_features.hpp>

#include <cmath>
#include <algorithm>


namespace
{

// Sorting and binary lookup must use the same lexicographic order.
bool triplet_key_less(const TripletKey& a, const TripletKey& b)
{
    if (a.keys != b.keys) {
        return a.keys < b.keys;
    }
    return a.distance_bins < b.distance_bins;
}

// Each offset is -1, 0 or 1. Chemical keys remain unchanged.
bool offset_distance_bins(TripletKey& key, const std::array<int, 3>& offsets)
{
    for (std::size_t i = 0; i < 3; ++i) {
        if (offsets[i] < 0) {
            if (key.distance_bins[i] == 0) {
                return false;
            }
            --key.distance_bins[i];
        }
        else if (offsets[i] > 0) {
            ++key.distance_bins[i];
        }
    }
    return true;
}

double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    return std::sqrt(
        interaction::distance2(a, b)
    );
}

bool points_are_aligned(
    const interaction::point3D& a,
    const interaction::point3D& b,
    const interaction::point3D& c)
{
    const double abx = b.x - a.x;
    const double aby = b.y - a.y;
    const double abz = b.z - a.z;

    const double acx = c.x - a.x;
    const double acy = c.y - a.y;
    const double acz = c.z - a.z;

    const double cross_x = aby * acz - abz * acy;
    const double cross_y = abz * acx - abx * acz;
    const double cross_z = abx * acy - aby * acx;

    // |AB x AC| is twice the triangle area. The fixed test below is in A^4,
    // rather than an angle or a distance normalized by edge length.
    const double cross_norm2 =
        cross_x * cross_x +
        cross_y * cross_y +
        cross_z * cross_z;

    return cross_norm2 < 1e-6;
}

} // namespace


std::vector<Triplet> generate_triplets(
    const std::vector<MatchingArea>& areas,
    double max_distance)
{
    std::vector<Triplet> triplets;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        for (std::size_t j = i + 1; j < areas.size(); ++j) {

            // Logical representations of one physical area cannot share a triplet.
            if (areas[i].source_area_index ==
                areas[j].source_area_index)
                continue;

            const double dij =
                point_distance(
                    areas[i].centroid,
                    areas[j].centroid
                );

            if (dij > max_distance || dij < 0.001)
                continue;

            for (std::size_t k = j + 1; k < areas.size(); ++k) {

                if (areas[i].source_area_index ==
                        areas[k].source_area_index ||
                    areas[j].source_area_index ==
                        areas[k].source_area_index)
                    continue;

                const double dik =
                    point_distance(
                        areas[i].centroid,
                        areas[k].centroid
                    );

                const double djk =
                    point_distance(
                        areas[j].centroid,
                        areas[k].centroid
                    );

                if (dik > max_distance ||
                    djk > max_distance)
                    continue;

                if (dik < 0.001 ||
                    djk < 0.001)
                    continue;

                if (points_are_aligned(
                        areas[i].centroid,
                        areas[j].centroid,
                        areas[k].centroid))
                    continue;

                triplets.push_back({
                    {i, j, k},
                    {
                        areas[i].key,
                        areas[j].key,
                        areas[k].key
                    },
                    {
                        dij,
                        dik,
                        djk
                    }
                });
            }
        }
    }

    return triplets;
}

std::size_t count_triplets_spatial(
    const std::vector<MatchingArea>& areas,
    const MatchingGrid& grid,
    double max_distance)
{
    std::size_t count = 0;

    const double max_distance2 =
        max_distance * max_distance;

    for (std::size_t i = 0;
         i < areas.size();
         ++i) {

        const auto& a = areas[i];

        const int gx =
            grid.to_grid(a.centroid.x, grid.xmin);

        const int gy =
            grid.to_grid(a.centroid.y, grid.ymin);

        const int gz =
            grid.to_grid(a.centroid.z, grid.zmin);

        std::vector<std::size_t> neighbors;

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {

                    const int x = gx + dx;
                    const int y = gy + dy;
                    const int z = gz + dz;

                    if (!grid.in_bounds(x, y, z))
                        continue;

                    const std::size_t cell =
                        static_cast<std::size_t>(
                            grid.index(x, y, z)
                        );

                    for (std::size_t p =
                             grid.cell_offsets[cell];
                         p < grid.cell_offsets[cell + 1];
                         ++p) {

                        const std::size_t j =
                            grid.area_indices[p];

                        if (j <= i)
                            continue;

                        if (areas[j].source_area_index ==
                            a.source_area_index)
                            continue;

                        const double d2 =
                            interaction::distance2(
                                a.centroid,
                                areas[j].centroid
                            );

                        if (d2 < 1e-6 ||
                            d2 > max_distance2)
                            continue;

                        neighbors.push_back(j);
                    }
                }
            }
        }

        std::sort(
            neighbors.begin(),
            neighbors.end()
        );

        for (std::size_t u = 0;
             u < neighbors.size();
             ++u) {

            const std::size_t j =
                neighbors[u];

            for (std::size_t v = u + 1;
                 v < neighbors.size();
                 ++v) {

                const std::size_t k =
                    neighbors[v];

                if (areas[j].source_area_index ==
                    areas[k].source_area_index)
                    continue;

                const double d2 =
                    interaction::distance2(
                        areas[j].centroid,
                        areas[k].centroid
                    );

                if (d2 < 1e-6 ||
                    d2 > max_distance2)
                    continue;

                if (points_are_aligned(
                        a.centroid,
                        areas[j].centroid,
                        areas[k].centroid))
                    continue;

                ++count;
            }
        }
    }

    return count;
}

std::size_t discretize_distance(
    double distance,
    double bin_size)
{
    return static_cast<std::size_t>(
        distance / bin_size
    );
}

CanonicalTriplet canonicalize_triplet(
    const Triplet& triplet,
    double bin_size)
{
    struct Candidate
    {
        std::array<std::size_t, 3> order{};
        std::array<InteractionKey, 3> keys{};
        std::array<double, 3> distances{};
        std::array<std::size_t, 3> bins{};
    };


    constexpr std::array<
        std::array<std::size_t, 3>,
        6
    > permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}}
    }};


    const auto distance_between =
        [&](std::size_t a,
            std::size_t b)
        {
            if (a > b)
                std::swap(a, b);

            if (a == 0 && b == 1)
                return triplet.distances[0];

            if (a == 0 && b == 2)
                return triplet.distances[1];

            return triplet.distances[2];
        };


    const auto make_candidate =
        [&](const std::array<std::size_t, 3>& order)
        {
            Candidate candidate;

            candidate.order = order;

            candidate.keys = {
                triplet.keys[order[0]],
                triplet.keys[order[1]],
                triplet.keys[order[2]]
            };

            candidate.distances = {
                distance_between(
                    order[0],
                    order[1]
                ),

                distance_between(
                    order[0],
                    order[2]
                ),

                distance_between(
                    order[1],
                    order[2]
                )
            };

            candidate.bins = {
                discretize_distance(
                    candidate.distances[0],
                    bin_size
                ),

                discretize_distance(
                    candidate.distances[1],
                    bin_size
                ),

                discretize_distance(
                    candidate.distances[2],
                    bin_size
                )
            };

            return candidate;
        };


    const auto signature_less =
        [](const Candidate& a,
           const Candidate& b)
        {
            if (a.keys != b.keys)
                return a.keys < b.keys;

            return a.bins < b.bins;
        };


    const auto same_signature =
        [](const Candidate& a,
           const Candidate& b)
        {
            return
                a.keys == b.keys &&
                a.bins == b.bins;
        };


    std::array<Candidate, 6> candidates;

    for (std::size_t i = 0;
         i < permutations.size();
         ++i) {

        candidates[i] =
            make_candidate(
                permutations[i]
            );
    }


    std::size_t best_index = 0;

    for (std::size_t i = 1;
         i < candidates.size();
         ++i) {

        if (signature_less(
                candidates[i],
                candidates[best_index])) {

            best_index = i;
        }
    }


    const Candidate& best =
        candidates[best_index];


    CanonicalTriplet result;

    result.keys =
        best.keys;

    result.distances =
        best.distances;

    result.distance_bins =
        best.bins;


    result.matching_area_indices = {
        triplet.matching_area_indices[
            best.order[0]
        ],
        triplet.matching_area_indices[
            best.order[1]
        ],
        triplet.matching_area_indices[
            best.order[2]
        ]
    };


    // Keep equivalent ligand orders: each gives a separate point correspondence.
    for (const auto& candidate :
         candidates) {

        if (!same_signature(
                candidate,
                best)) {

            continue;
        }

        const std::size_t n =
            result.equivalent_order_count;

        result.equivalent_orders[n] = {
            triplet.matching_area_indices[
                candidate.order[0]
            ],
            triplet.matching_area_indices[
                candidate.order[1]
            ],
            triplet.matching_area_indices[
                candidate.order[2]
            ]
        };

        ++result.equivalent_order_count;
    }


    return result;
}

TripletKey make_triplet_key(
    const CanonicalTriplet& triplet)
{
    return {
        triplet.keys,
        triplet.distance_bins
    };
}


std::size_t TripletKeyHash::operator()(
    const TripletKey& key) const
{
    std::size_t seed = 0;

    for (const auto interaction_key :
         key.keys) {

        seed ^=
            std::hash<int>{}(
                static_cast<int>(
                    interaction_key
                )
            )
            + 0x9e3779b9
            + (seed << 6)
            + (seed >> 2);
    }

    for (const auto bin :
         key.distance_bins) {

        seed ^=
            std::hash<std::size_t>{}(
                bin
            )
            + 0x9e3779b9
            + (seed << 6)
            + (seed >> 2);
    }

    return seed;
}

ProteinTripletRecord make_protein_triplet_record(
    const CanonicalTriplet& triplet)
{
    return {
        {
            static_cast<std::uint32_t>(
                triplet.matching_area_indices[0]
            ),
            static_cast<std::uint32_t>(
                triplet.matching_area_indices[1]
            ),
            static_cast<std::uint32_t>(
                triplet.matching_area_indices[2]
            )
        }
    };
}

ProteinTripletHashTable build_protein_triplet_hash_table(
    const std::vector<PoseTarget>& targets,
    double max_distance,
    double bin_size)
{
    ProteinTripletHashTable table;

    for (std::size_t i = 0;
         i < targets.size();
         ++i) {

        for (std::size_t j = i + 1;
             j < targets.size();
             ++j) {

            // Targets from one physical area cannot share a triplet.
            if (
                targets[i].source_area_index ==
                targets[j].source_area_index
            ) {
                continue;
            }

            const double dij =
                point_distance(
                    targets[i].position,
                    targets[j].position
                );

            if (
                dij < 0.001 ||
                dij > max_distance
            ) {
                continue;
            }

            for (std::size_t k = j + 1;
                 k < targets.size();
                 ++k) {

                if (
                    targets[k].source_area_index ==
                        targets[i].source_area_index
                    ||
                    targets[k].source_area_index ==
                        targets[j].source_area_index
                ) {
                    continue;
                }

                const double dik =
                    point_distance(
                        targets[i].position,
                        targets[k].position
                    );

                const double djk =
                    point_distance(
                        targets[j].position,
                        targets[k].position
                    );

                if (
                    dik < 0.001 ||
                    djk < 0.001
                ) {
                    continue;
                }

                if (
                    dik > max_distance ||
                    djk > max_distance
                ) {
                    continue;
                }

                if (
                    points_are_aligned(
                        targets[i].position,
                        targets[j].position,
                        targets[k].position
                    )
                ) {
                    continue;
                }

                const Triplet triplet{
                    {i, j, k},
                    {
                        targets[i].key,
                        targets[j].key,
                        targets[k].key
                    },
                    {
                        dij,
                        dik,
                        djk
                    }
                };

                const auto canonical =
                    canonicalize_triplet(
                        triplet,
                        bin_size
                    );

                const auto key =
                    make_triplet_key(
                        canonical
                    );

                table[key].push_back(
                    make_protein_triplet_record(
                        canonical
                    )
                );
            }
        }
    }

    return table;
}


std::vector<const std::vector<ProteinTripletRecord>*>
find_neighbor_triplet_buckets(
    const ProteinTripletHashTable& table,
    const CanonicalTriplet& ligand_triplet)
{
    std::vector<
        const std::vector<ProteinTripletRecord>*
    > result;

    const auto base_key =
        make_triplet_key(ligand_triplet);

    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                TripletKey key = base_key;

                if (!offset_distance_bins(key, {d0, d1, d2})) {
                    continue;
                }

                const auto it =
                    table.find(key);

                if (it != table.end()) {
                    result.push_back(
                        &it->second
                    );
                }
            }
        }
    }

    return result;
}

FlatTripletIndex flatten_protein_triplet_index(
    const ProteinTripletHashTable& table
)
{
    FlatTripletIndex result;


    result.keys.reserve(
        table.size()
    );


    for (const auto& [key, bucket] :
         table) {

        (void)bucket;

        result.keys.push_back(
            key
        );
    }


    std::sort(
        result.keys.begin(),
        result.keys.end(),
        triplet_key_less
    );


    result.offsets.reserve(
        result.keys.size() + 1
    );


    result.offsets.push_back(0);


    for (const auto& key :
         result.keys) {

        const auto it =
            table.find(
                key
            );


        if (it == table.end()) {
            continue;
        }


        result.records.insert(
            result.records.end(),
            it->second.begin(),
            it->second.end()
        );


        result.offsets.push_back(
            result.records.size()
        );
    }


    return result;
}

std::vector<FlatTripletBucket>
find_neighbor_triplet_buckets_flat(
    const FlatTripletIndex& index,
    const CanonicalTriplet& ligand_triplet
)
{
    std::vector<FlatTripletBucket> result;


    const TripletKey base_key =
        make_triplet_key(
            ligand_triplet
        );


    for (int d0 = -1; d0 <= 1; ++d0) {

        for (int d1 = -1; d1 <= 1; ++d1) {

            for (int d2 = -1; d2 <= 1; ++d2) {

                TripletKey key =
                    base_key;


                if (!offset_distance_bins(key, {d0, d1, d2})) {
                    continue;
                }


                const auto it =
                    std::lower_bound(
                        index.keys.begin(),
                        index.keys.end(),
                        key,
                        triplet_key_less
                    );


                if (
                    it == index.keys.end() ||
                    !(*it == key)
                ) {
                    continue;
                }


                const std::size_t key_index =
                    static_cast<std::size_t>(
                        std::distance(
                            index.keys.begin(),
                            it
                        )
                    );


                result.push_back({
                    index.offsets[key_index],
                    index.offsets[key_index + 1]
                });
            }
        }
    }


    return result;
}

bool validate_flat_triplet_index(
    const ProteinTripletHashTable& table,
    const FlatTripletIndex& flat_index
)
{
    if (
        flat_index.keys.size() != table.size() ||
        flat_index.offsets.size() !=
            flat_index.keys.size() + 1
    ) {
        return false;
    }


    std::size_t expected_records = 0;

    for (const auto& [key, bucket] : table) {
        (void)key;

        expected_records +=
            bucket.size();
    }


    if (
        flat_index.records.size() !=
        expected_records
    ) {
        return false;
    }


    for (std::size_t key_index = 0;
         key_index < flat_index.keys.size();
         ++key_index) {

        const auto& key =
            flat_index.keys[key_index];


        const auto table_it =
            table.find(key);


        if (table_it == table.end()) {
            return false;
        }


        const auto& original_bucket =
            table_it->second;


        const std::size_t begin =
            flat_index.offsets[key_index];

        const std::size_t end =
            flat_index.offsets[key_index + 1];


        if (
            end - begin !=
            original_bucket.size()
        ) {
            return false;
        }


        for (std::size_t i = 0;
             i < original_bucket.size();
             ++i) {

            if (
                flat_index.records[begin + i].indices !=
                original_bucket[i].indices
            ) {
                return false;
            }
        }
    }


    return true;
}
