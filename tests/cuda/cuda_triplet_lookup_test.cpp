#include <cstring>
#include <cuda_runtime_api.h>
#include "cuda_backend.hpp"

#include <cstdint>
#include <iostream>
#include <vector>


namespace
{

GpuTripletKey make_key(
    std::uint8_t k0,
    std::uint8_t k1,
    std::uint8_t k2,
    std::uint32_t d0,
    std::uint32_t d1,
    std::uint32_t d2
)
{
    GpuTripletKey key{};

    key.keys[0] = k0;
    key.keys[1] = k1;
    key.keys[2] = k2;

    key.distance_bins[0] = d0;
    key.distance_bins[1] = d1;
    key.distance_bins[2] = d2;

    return key;
}

} // namespace


int main()
{
    GpuFlatTripletIndexHost host_index;


    // ========================================================
    // Test receptor index
    // ========================================================

    host_index.keys = {
        make_key(
            0, 1, 2,
            1, 2, 3
        ),

        make_key(
            0, 1, 2,
            4, 5, 6
        ),

        make_key(
            1, 2, 3,
            7, 8, 9
        )
    };


    /*
     * key 0 -> 2 records
     * key 1 -> 1 record
     * key 2 -> 1 record
     */
    host_index.offsets = {
        0,
        2,
        3,
        4
    };


    host_index.records = {
        {{0, 1, 2}},
        {{10, 11, 12}},

        {{3, 4, 5}},

        {{6, 7, 8}}
    };


    GpuFlatTripletIndexDevice device_index;


    // ========================================================
    // Upload
    // ========================================================

    if (
        !cuda_upload_triplet_index(
            host_index,
            device_index
        )
    ) {
        std::cerr
            << "Triplet index upload failed\n";

        return 1;
    }


    // ========================================================
    // Exact lookup
    // ========================================================

    const std::vector<GpuTripletKey>
        exact_queries = {

            host_index.keys[0],
            host_index.keys[2],

            make_key(
                0, 1, 2,
                100, 100, 100
            )
        };


    std::vector<std::uint32_t>
        exact_results;


    if (
        !cuda_lookup_triplet_keys_exact(
            device_index,
            exact_queries,
            exact_results
        )
    ) {
        std::cerr
            << "Exact CUDA lookup failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        exact_results.size() != 3 ||
        exact_results[0] != 0 ||
        exact_results[1] != 2 ||
        exact_results[2] !=
            GPU_TRIPLET_KEY_NOT_FOUND
    ) {
        std::cerr
            << "Unexpected exact lookup results\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA exact triplet lookup passed\n";


    // ========================================================
    // 27-neighbor lookup
    // ========================================================

    const std::vector<GpuTripletKey>
        neighbor_queries = {

            make_key(
                0, 1, 2,
                4, 5, 6
            )
        };


    std::vector<std::uint32_t>
        neighbor_results;


    if (
        !cuda_lookup_triplet_key_neighbors(
            device_index,
            neighbor_queries,
            neighbor_results
        )
    ) {
        std::cerr
            << "CUDA neighbor lookup failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        neighbor_results.size() != 27
    ) {
        std::cerr
            << "Unexpected neighbor result count\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    bool found_exact_key = false;


    for (const auto key_index :
         neighbor_results) {

        if (key_index == 1) {
            found_exact_key = true;
            break;
        }
    }


    if (!found_exact_key) {

        std::cerr
            << "Exact key missing from "
               "27-neighbor lookup\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA 27-neighbor triplet lookup passed\n";


    // ========================================================
    // Record counting
    // ========================================================

    const std::vector<GpuTripletKey>
        count_queries = {

            make_key(
                0, 1, 2,
                2, 3, 4
            ),

            make_key(
                0, 1, 2,
                4, 5, 6
            ),

            make_key(
                9, 9, 9,
                4, 5, 6
            )
        };


    std::vector<std::uint32_t>
        record_counts;


    if (
        !cuda_count_triplet_neighbor_records(
            device_index,
            count_queries,
            record_counts
        )
    ) {
        std::cerr
            << "CUDA neighbor record counting failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        record_counts.size() != 3 ||
        record_counts[0] != 2 ||
        record_counts[1] != 1 ||
        record_counts[2] != 0
    ) {
        std::cerr
            << "Unexpected neighbor record counts\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA neighbor record counting passed\n";



    // ========================================================
    // Device-native triplet neighbor counting equivalence
    // ========================================================

    {
        GpuTripletNeighborCountArrayDevice device_counts{};
        GpuTripletNeighborCountArrayDevice empty_counts{};

        const auto release = [&]() {
            cuda_free_triplet_neighbor_counts(empty_counts);
            cuda_free_triplet_neighbor_counts(device_counts);
        };

        const auto fail = [&](const char* message) {
            release();
            cuda_free_triplet_index(device_index);
            std::cerr << message << '\n';
            return 1;
        };

        if (
            !cuda_count_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts
            )
        ) {
            return fail("CUDA device neighbor counting failed");
        }

        std::vector<std::uint32_t> downloaded_counts;

        if (
            !cuda_download_triplet_neighbor_counts(
                device_counts,
                downloaded_counts
            )
            ||
            device_counts.count != record_counts.size()
            ||
            downloaded_counts != record_counts
        ) {
            return fail("CUDA device neighbor count mismatch");
        }

        // A non-empty output must not be overwritten.
        if (
            cuda_count_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts
            )
        ) {
            return fail("Occupied device count output accepted");
        }

        // An empty query list produces an empty output.
        const std::vector<GpuTripletKey> empty_queries;

        if (
            !cuda_count_triplet_neighbor_records_device(
                device_index,
                empty_queries,
                empty_counts
            )
        ) {
            return fail("Empty device neighbor counting failed");
        }

        std::vector<std::uint32_t> downloaded_empty;

        if (
            !cuda_download_triplet_neighbor_counts(
                empty_counts,
                downloaded_empty
            )
            ||
            !downloaded_empty.empty()
            ||
            empty_counts.count != 0
            ||
            empty_counts.counts != nullptr
        ) {
            return fail("Empty device neighbor counting mismatch");
        }

        // A missing GPU triplet index must be rejected
        // when queries are non-empty.
        GpuFlatTripletIndexDevice invalid_index{};
        GpuTripletNeighborCountArrayDevice invalid_counts{};

        if (
            cuda_count_triplet_neighbor_records_device(
                invalid_index,
                count_queries,
                invalid_counts
            )
            ||
            invalid_counts.counts != nullptr
            ||
            invalid_counts.count != 0
        ) {
            cuda_free_triplet_neighbor_counts(invalid_counts);
            return fail("Invalid device triplet index accepted");
        }

        release();

        if (
            device_counts.counts != nullptr ||
            device_counts.count != 0 ||
            empty_counts.counts != nullptr ||
            empty_counts.count != 0
        ) {
            cuda_free_triplet_index(device_index);
            std::cerr << "Device neighbor count cleanup failed\n";
            return 1;
        }

        std::cout
            << "CUDA device neighbor counting equivalence passed\n";
    }


    // ========================================================
    // Device-native triplet neighbor prefix scan equivalence
    // ========================================================

    {
        GpuTripletNeighborCountArrayDevice device_counts{};
        GpuTripletNeighborOffsetArrayDevice device_offsets{};

        GpuTripletNeighborCountArrayDevice empty_counts{};
        GpuTripletNeighborOffsetArrayDevice empty_offsets{};

        GpuTripletNeighborCountArrayDevice overflow_counts{};
        GpuTripletNeighborOffsetArrayDevice overflow_offsets{};

        const auto release = [&]() {
            cuda_free_triplet_neighbor_offsets(overflow_offsets);
            cuda_free_triplet_neighbor_counts(overflow_counts);
            cuda_free_triplet_neighbor_offsets(empty_offsets);
            cuda_free_triplet_neighbor_offsets(device_offsets);
            cuda_free_triplet_neighbor_counts(device_counts);
        };

        const auto fail = [&](const char* message) {
            release();
            cuda_free_triplet_index(device_index);
            std::cerr << message << '\n';
            return 1;
        };

        // Count neighbors directly on the GPU.
        if (
            !cuda_count_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts
            )
        ) {
            return fail("Device neighbor counting failed");
        }

        // Scan counts without downloading or uploading them.
        if (
            !cuda_exclusive_triplet_neighbor_prefix_sum_device(
                device_counts,
                device_offsets
            )
        ) {
            return fail("Device neighbor prefix scan failed");
        }

        // Independent reference from the legacy API.
        std::vector<std::uint32_t> reference_offsets;

        if (
            !cuda_exclusive_prefix_sum(
                record_counts,
                reference_offsets
            )
        ) {
            return fail("Legacy neighbor prefix scan failed");
        }

        std::vector<std::uint32_t> downloaded_offsets;

        if (
            !cuda_download_triplet_neighbor_offsets(
                device_offsets,
                downloaded_offsets
            )
            ||
            device_offsets.count != reference_offsets.size()
            ||
            downloaded_offsets != reference_offsets
            ||
            downloaded_offsets !=
                std::vector<std::uint32_t>{0, 2, 3, 3}
        ) {
            return fail("Device neighbor prefix offset mismatch");
        }

        // Reusing an occupied output must fail.
        if (
            cuda_exclusive_triplet_neighbor_prefix_sum_device(
                device_counts,
                device_offsets
            )
        ) {
            return fail("Occupied neighbor offset output accepted");
        }

        // Empty counts must produce exactly [0].
        if (
            !cuda_exclusive_triplet_neighbor_prefix_sum_device(
                empty_counts,
                empty_offsets
            )
        ) {
            return fail("Empty neighbor prefix scan failed");
        }

        std::vector<std::uint32_t> downloaded_empty;

        if (
            !cuda_download_triplet_neighbor_offsets(
                empty_offsets,
                downloaded_empty
            )
            ||
            empty_offsets.count != 1
            ||
            downloaded_empty !=
                std::vector<std::uint32_t>{0}
        ) {
            return fail("Empty neighbor prefix scan mismatch");
        }

        // UINT32_MAX + 1 must be rejected.
        const std::uint32_t overflow_input[2] = {
            ~std::uint32_t{0},
            1
        };

        if (
            cudaMalloc(
                reinterpret_cast<void**>(
                    &overflow_counts.counts
                ),
                sizeof(overflow_input)
            ) != cudaSuccess
        ) {
            return fail("Neighbor overflow allocation failed");
        }

        overflow_counts.count = 2;

        if (
            cudaMemcpy(
                overflow_counts.counts,
                overflow_input,
                sizeof(overflow_input),
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            return fail("Neighbor overflow upload failed");
        }

        if (
            cuda_exclusive_triplet_neighbor_prefix_sum_device(
                overflow_counts,
                overflow_offsets
            )
            ||
            overflow_offsets.offsets != nullptr
            ||
            overflow_offsets.count != 0
        ) {
            return fail("Neighbor prefix scan overflow not rejected");
        }

        release();

        if (
            device_counts.counts != nullptr ||
            device_counts.count != 0 ||
            device_offsets.offsets != nullptr ||
            device_offsets.count != 0 ||
            empty_offsets.offsets != nullptr ||
            empty_offsets.count != 0 ||
            overflow_counts.counts != nullptr ||
            overflow_counts.count != 0 ||
            overflow_offsets.offsets != nullptr ||
            overflow_offsets.count != 0
        ) {
            cuda_free_triplet_index(device_index);
            std::cerr << "Device neighbor scan cleanup failed\n";
            return 1;
        }

        std::cout
            << "CUDA device neighbor prefix scan equivalence passed\n";
    }


    std::vector<std::uint32_t>
        candidate_offsets;


    if (
        !cuda_exclusive_prefix_sum(
            record_counts,
            candidate_offsets
        )
    ) {
        std::cerr
            << "CUDA prefix sum failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_offsets = {
            0,
            2,
            3,
            3
        };


    if (
        candidate_offsets !=
        expected_offsets
    ) {
        std::cerr
            << "Unexpected CUDA prefix-sum result\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA prefix sum passed\n";


    // ========================================================
    // Candidate expansion
    // ========================================================

    std::vector<GpuTripletCandidate>
        candidates;


    if (
        !cuda_expand_triplet_neighbor_records(
            device_index,
            count_queries,
            candidate_offsets,
            candidates
        )
    ) {
        std::cerr
            << "CUDA candidate expansion failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    // ========================================================
    // Device-native candidate expansion equivalence
    // ========================================================

    {
        GpuTripletCandidateArrayDevice
            device_candidates{};

        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                candidate_offsets,
                device_candidates
            )
        ) {
            cuda_free_triplet_candidates(device_candidates);
            cuda_free_triplet_index(device_index);

            std::cerr
                << "CUDA device candidate expansion failed\n";

            return 1;
        }

        std::vector<GpuTripletCandidate>
            downloaded_candidates;

        bool matches =
            cuda_download_triplet_candidates(
                device_candidates,
                downloaded_candidates
            )
            &&
            device_candidates.count == candidates.size()
            &&
            downloaded_candidates.size() == candidates.size();

        if (matches) {
            for (
                std::size_t i = 0;
                i < candidates.size();
                ++i
            ) {
                const auto& a = candidates[i];
                const auto& b = downloaded_candidates[i];

                if (
                    a.ligand_triplet_index !=
                        b.ligand_triplet_index
                    ||
                    a.protein_indices[0] != b.protein_indices[0]
                    ||
                    a.protein_indices[1] != b.protein_indices[1]
                    ||
                    a.protein_indices[2] != b.protein_indices[2]
                ) {
                    matches = false;
                    break;
                }
            }
        }

        cuda_free_triplet_candidates(
            device_candidates
        );

        if (
            device_candidates.candidates != nullptr ||
            device_candidates.count != 0
        ) {
            matches = false;
        }

        if (!matches) {
            std::cerr
                << "CUDA device candidate equivalence failed\n";

            cuda_free_triplet_index(device_index);

            return 1;
        }

        std::cout
            << "CUDA device triplet candidate equivalence passed\n";
    }


    if (candidates.size() != 3) {

        std::cerr
            << "Unexpected candidate count: "
            << candidates.size()
            << '\n';

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[0].ligand_triplet_index != 0 ||
        candidates[0].protein_indices[0] != 0 ||
        candidates[0].protein_indices[1] != 1 ||
        candidates[0].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected candidate 0\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[1].ligand_triplet_index != 0 ||
        candidates[1].protein_indices[0] != 10 ||
        candidates[1].protein_indices[1] != 11 ||
        candidates[1].protein_indices[2] != 12
    ) {
        std::cerr
            << "Unexpected candidate 1\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[2].ligand_triplet_index != 1 ||
        candidates[2].protein_indices[0] != 3 ||
        candidates[2].protein_indices[1] != 4 ||
        candidates[2].protein_indices[2] != 5
    ) {
        std::cerr
            << "Unexpected candidate 2\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA candidate expansion passed\n";


    // ========================================================
    // Fully device-native triplet candidate expansion
    // ========================================================

    {
        GpuTripletNeighborCountArrayDevice device_counts{};
        GpuTripletNeighborOffsetArrayDevice device_offsets{};
        GpuTripletCandidateArrayDevice device_candidates{};

        GpuTripletNeighborOffsetArrayDevice invalid_offsets{};

        GpuTripletNeighborCountArrayDevice empty_counts{};
        GpuTripletNeighborOffsetArrayDevice empty_offsets{};
        GpuTripletCandidateArrayDevice empty_candidates{};

        const auto release = [&]() {
            cuda_free_triplet_candidates(empty_candidates);
            cuda_free_triplet_neighbor_offsets(empty_offsets);

            cuda_free_triplet_neighbor_offsets(invalid_offsets);

            cuda_free_triplet_candidates(device_candidates);
            cuda_free_triplet_neighbor_offsets(device_offsets);
            cuda_free_triplet_neighbor_counts(device_counts);
        };

        const auto fail = [&](const char* message) {
            release();
            cuda_free_triplet_index(device_index);
            std::cerr << message << '\n';
            return 1;
        };

        // Counts remain on the GPU.
        if (
            !cuda_count_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts
            )
        ) {
            return fail("Device triplet counting failed");
        }

        // Prefix scan consumes the GPU counts directly.
        if (
            !cuda_exclusive_triplet_neighbor_prefix_sum_device(
                device_counts,
                device_offsets
            )
        ) {
            return fail("Device triplet prefix scan failed");
        }

        // Expansion consumes both GPU buffers directly.
        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts,
                device_offsets,
                device_candidates
            )
        ) {
            return fail("Device-native triplet expansion failed");
        }

        std::vector<GpuTripletCandidate>
            downloaded_candidates;

        if (
            !cuda_download_triplet_candidates(
                device_candidates,
                downloaded_candidates
            )
            ||
            device_candidates.count != candidates.size()
            ||
            downloaded_candidates.size() != candidates.size()
            ||
            downloaded_candidates.size() != 3
        ) {
            return fail("Device-native triplet count mismatch");
        }

        // Both paths use the same expansion kernel.
        // Compare the complete generated records.
        for (
            std::size_t i = 0;
            i < candidates.size();
            ++i
        ) {
            if (
                std::memcmp(
                    &candidates[i],
                    &downloaded_candidates[i],
                    sizeof(GpuTripletCandidate)
                ) != 0
            ) {
                return fail("Device-native triplet record mismatch");
            }
        }

        // Occupied output must be rejected.
        if (
            cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts,
                device_offsets,
                device_candidates
            )
        ) {
            return fail("Occupied candidate output accepted");
        }

        cuda_free_triplet_candidates(device_candidates);

        // Corrupt one intermediate offset.
        std::vector<std::uint32_t> bad_host_offsets =
            candidate_offsets;

        bad_host_offsets[1] += 1;

        const std::size_t offset_bytes =
            bad_host_offsets.size() * sizeof(std::uint32_t);

        if (
            cudaMalloc(
                reinterpret_cast<void**>(&invalid_offsets.offsets),
                offset_bytes
            ) != cudaSuccess
        ) {
            return fail("Invalid offset allocation failed");
        }

        invalid_offsets.count =
            static_cast<std::uint32_t>(
                bad_host_offsets.size()
            );

        if (
            cudaMemcpy(
                invalid_offsets.offsets,
                bad_host_offsets.data(),
                offset_bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            return fail("Invalid offset upload failed");
        }

        if (
            cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                device_counts,
                invalid_offsets,
                device_candidates
            )
            ||
            device_candidates.candidates != nullptr
            ||
            device_candidates.count != 0
        ) {
            return fail("Invalid device offsets not rejected");
        }

        // Empty queries must produce no candidates.
        if (
            !cuda_exclusive_triplet_neighbor_prefix_sum_device(
                empty_counts,
                empty_offsets
            )
        ) {
            return fail("Empty device triplet scan failed");
        }

        const std::vector<GpuTripletKey> empty_queries;

        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                empty_queries,
                empty_counts,
                empty_offsets,
                empty_candidates
            )
            ||
            empty_candidates.candidates != nullptr
            ||
            empty_candidates.count != 0
        ) {
            return fail("Empty device triplet expansion failed");
        }

        release();

        if (
            device_counts.counts != nullptr ||
            device_counts.count != 0 ||
            device_offsets.offsets != nullptr ||
            device_offsets.count != 0 ||
            device_candidates.candidates != nullptr ||
            device_candidates.count != 0 ||
            invalid_offsets.offsets != nullptr ||
            invalid_offsets.count != 0 ||
            empty_offsets.offsets != nullptr ||
            empty_offsets.count != 0 ||
            empty_candidates.candidates != nullptr ||
            empty_candidates.count != 0
        ) {
            cuda_free_triplet_index(device_index);
            std::cerr << "Device triplet expansion cleanup failed\n";
            return 1;
        }

        std::cout
            << "CUDA device-native triplet expansion equivalence passed\n";
    }


    std::vector<GpuLigandTripletOrders>
        ligand_orders(3);


    /*
    * Ligand triplet 0 has two equivalent orders.
    */
    ligand_orders[0].equivalent_order_count = 2;

    ligand_orders[0].equivalent_orders[0][0] = 100;
    ligand_orders[0].equivalent_orders[0][1] = 101;
    ligand_orders[0].equivalent_orders[0][2] = 102;

    ligand_orders[0].equivalent_orders[1][0] = 102;
    ligand_orders[0].equivalent_orders[1][1] = 101;
    ligand_orders[0].equivalent_orders[1][2] = 100;


    /*
    * Ligand triplet 1 has one equivalent order.
    */
    ligand_orders[1].equivalent_order_count = 1;

    ligand_orders[1].equivalent_orders[0][0] = 200;
    ligand_orders[1].equivalent_orders[0][1] = 201;
    ligand_orders[1].equivalent_orders[0][2] = 202;


    /*
    * Ligand triplet 2 has no receptor match,
    * but keep one valid order for completeness.
    */
    ligand_orders[2].equivalent_order_count = 1;

    ligand_orders[2].equivalent_orders[0][0] = 300;
    ligand_orders[2].equivalent_orders[0][1] = 301;
    ligand_orders[2].equivalent_orders[0][2] = 302;


    /*
    * candidates currently are:
    *
    * candidate 0 -> ligand triplet 0
    * candidate 1 -> ligand triplet 0
    * candidate 2 -> ligand triplet 1
    *
    * Therefore:
    *
    * counts = [2, 2, 1]
    */
    std::vector<std::uint32_t>
        alignment_counts;


    if (
        !cuda_count_alignment_candidates(
            candidates,
            ligand_orders,
            alignment_counts
        )
    ) {
        std::cerr
            << "CUDA alignment candidate counting failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_alignment_counts = {
            2,
            2,
            1
        };


    if (
        alignment_counts !=
        expected_alignment_counts
    ) {
        std::cerr
            << "Unexpected alignment candidate counts\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    // ========================================================
    // Device-native alignment counting equivalence
    // ========================================================

    {
        GpuTripletCandidateArrayDevice
            device_candidates{};

        GpuAlignmentCountArrayDevice
            device_counts{};

        const auto fail = [&](const char* message) {
            cuda_free_alignment_counts(device_counts);
            cuda_free_triplet_candidates(device_candidates);
            cuda_free_triplet_index(device_index);

            std::cerr << message << '\n';
            return 1;
        };

        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                candidate_offsets,
                device_candidates
            )
        ) {
            return fail(
                "CUDA device candidate expansion failed"
            );
        }

        if (
            !cuda_count_alignment_candidates_device(
                device_candidates,
                ligand_orders,
                device_counts
            )
        ) {
            return fail(
                "CUDA device alignment counting failed"
            );
        }

        std::vector<std::uint32_t>
            downloaded_counts;

        if (
            !cuda_download_alignment_counts(
                device_counts,
                downloaded_counts
            )
            ||
            device_counts.count != alignment_counts.size()
            ||
            downloaded_counts != alignment_counts
            ||
            downloaded_counts != expected_alignment_counts
        ) {
            return fail(
                "CUDA device alignment count mismatch"
            );
        }

        cuda_free_alignment_counts(device_counts);

        if (
            device_counts.counts != nullptr ||
            device_counts.count != 0
        ) {
            return fail(
                "CUDA alignment count cleanup failed"
            );
        }

        // Candidate 2 references ligand triplet 1.
        // With only triplet 0 available, counting must fail.
        const std::vector<GpuLigandTripletOrders>
            truncated_orders{ligand_orders[0]};

        if (
            cuda_count_alignment_candidates_device(
                device_candidates,
                truncated_orders,
                device_counts
            )
            ||
            device_counts.counts != nullptr
            ||
            device_counts.count != 0
        ) {
            return fail(
                "CUDA invalid ligand triplet index not rejected"
            );
        }

        cuda_free_triplet_candidates(device_candidates);

        if (
            device_candidates.candidates != nullptr ||
            device_candidates.count != 0
        ) {
            return fail(
                "CUDA triplet candidate cleanup failed"
            );
        }

        std::cout
            << "CUDA device alignment counting equivalence passed\n";
    }


    std::vector<std::uint32_t>
        alignment_offsets;


    if (
        !cuda_exclusive_prefix_sum(
            alignment_counts,
            alignment_offsets
        )
    ) {
        std::cerr
            << "CUDA alignment prefix sum failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_alignment_offsets = {
            0,
            2,
            4,
            5
        };


    if (
        alignment_offsets !=
        expected_alignment_offsets
    ) {
        std::cerr
            << "Unexpected alignment offsets\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    // ========================================================
    // Device-native alignment prefix scan equivalence
    // ========================================================

    {
        GpuTripletCandidateArrayDevice device_candidates{};
        GpuAlignmentCountArrayDevice device_counts{};
        GpuAlignmentOffsetArrayDevice device_offsets{};

        GpuAlignmentOffsetArrayDevice empty_offsets{};

        GpuAlignmentCountArrayDevice overflow_counts{};
        GpuAlignmentOffsetArrayDevice overflow_offsets{};

        const auto release = [&]() {
            cuda_free_alignment_offsets(overflow_offsets);
            cuda_free_alignment_counts(overflow_counts);
            cuda_free_alignment_offsets(empty_offsets);
            cuda_free_alignment_offsets(device_offsets);
            cuda_free_alignment_counts(device_counts);
            cuda_free_triplet_candidates(device_candidates);
        };

        const auto fail = [&](const char* message) {
            release();
            cuda_free_triplet_index(device_index);
            std::cerr << message << '\n';
            return 1;
        };

        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                candidate_offsets,
                device_candidates
            )
        ) {
            return fail("Device candidate expansion failed");
        }

        if (
            !cuda_count_alignment_candidates_device(
                device_candidates,
                ligand_orders,
                device_counts
            )
        ) {
            return fail("Device alignment counting failed");
        }

        if (
            !cuda_exclusive_prefix_sum_device(
                device_counts,
                device_offsets
            )
        ) {
            return fail("Device alignment prefix scan failed");
        }

        std::vector<std::uint32_t> downloaded_offsets;

        if (
            !cuda_download_alignment_offsets(
                device_offsets,
                downloaded_offsets
            )
            ||
            device_offsets.count !=
                expected_alignment_offsets.size()
            ||
            downloaded_offsets != alignment_offsets
            ||
            downloaded_offsets != expected_alignment_offsets
        ) {
            return fail("Device alignment offsets mismatch");
        }

        // Empty counts must produce offsets = [0].
        GpuAlignmentCountArrayDevice empty_counts{};

        if (
            !cuda_exclusive_prefix_sum_device(
                empty_counts,
                empty_offsets
            )
        ) {
            return fail("Empty device scan failed");
        }

        std::vector<std::uint32_t> downloaded_empty;

        if (
            !cuda_download_alignment_offsets(
                empty_offsets,
                downloaded_empty
            )
            ||
            downloaded_empty !=
                std::vector<std::uint32_t>{0}
        ) {
            return fail("Empty device scan mismatch");
        }

        // Overflow: UINT32_MAX + 1 must be rejected.
        const std::uint32_t overflow_input[2] = {
            ~std::uint32_t{0},
            1
        };

        if (
            cudaMalloc(
                reinterpret_cast<void**>(
                    &overflow_counts.counts
                ),
                sizeof(overflow_input)
            ) != cudaSuccess
        ) {
            return fail("Overflow test allocation failed");
        }

        overflow_counts.count = 2;

        if (
            cudaMemcpy(
                overflow_counts.counts,
                overflow_input,
                sizeof(overflow_input),
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            return fail("Overflow test upload failed");
        }

        if (
            cuda_exclusive_prefix_sum_device(
                overflow_counts,
                overflow_offsets
            )
            ||
            overflow_offsets.offsets != nullptr
            ||
            overflow_offsets.count != 0
        ) {
            return fail("Device scan overflow not rejected");
        }

        release();

        if (
            device_candidates.candidates != nullptr ||
            device_candidates.count != 0 ||
            device_counts.counts != nullptr ||
            device_counts.count != 0 ||
            device_offsets.offsets != nullptr ||
            device_offsets.count != 0 ||
            empty_offsets.offsets != nullptr ||
            empty_offsets.count != 0 ||
            overflow_counts.counts != nullptr ||
            overflow_counts.count != 0
        ) {
            cuda_free_triplet_index(device_index);
            std::cerr << "Device scan cleanup failed\n";
            return 1;
        }

        std::cout
            << "CUDA device alignment prefix scan equivalence passed\n";
    }


    // ========================================================
    // Device-native alignment expansion equivalence
    // ========================================================

    {
        GpuTripletCandidateArrayDevice device_candidates{};
        GpuAlignmentCountArrayDevice device_counts{};
        GpuAlignmentOffsetArrayDevice device_offsets{};
        GpuAlignmentCandidateArrayDevice device_alignments{};

        GpuAlignmentOffsetArrayDevice modified_offsets{};
        GpuAlignmentOffsetArrayDevice empty_offsets{};

        const auto release = [&]() {
            cuda_free_alignment_candidates(device_alignments);
            cuda_free_alignment_offsets(empty_offsets);
            cuda_free_alignment_offsets(modified_offsets);
            cuda_free_alignment_offsets(device_offsets);
            cuda_free_alignment_counts(device_counts);
            cuda_free_triplet_candidates(device_candidates);
        };

        const auto fail = [&](const char* message) {
            release();
            cuda_free_triplet_index(device_index);
            std::cerr << message << '\n';
            return 1;
        };

        // Baseline from the host-vector API; both paths use the same expansion kernel.
        std::vector<GpuAlignmentCandidate> reference_alignments;

        if (
            !cuda_expand_alignment_candidates(
                candidates,
                ligand_orders,
                alignment_offsets,
                reference_alignments
            )
        ) {
            return fail("Legacy alignment expansion failed");
        }

        if (
            !cuda_expand_triplet_neighbor_records_device(
                device_index,
                count_queries,
                candidate_offsets,
                device_candidates
            )
        ) {
            return fail("Device triplet expansion failed");
        }

        if (
            !cuda_count_alignment_candidates_device(
                device_candidates,
                ligand_orders,
                device_counts
            )
        ) {
            return fail("Device alignment counting failed");
        }

        if (
            !cuda_exclusive_prefix_sum_device(
                device_counts,
                device_offsets
            )
        ) {
            return fail("Device alignment scan failed");
        }

        if (
            !cuda_expand_alignment_candidates_device(
                device_candidates,
                ligand_orders,
                device_offsets,
                device_alignments
            )
        ) {
            return fail("Device-native alignment expansion failed");
        }

        std::vector<GpuAlignmentCandidate> downloaded_alignments;

        if (
            !cuda_download_alignment_candidates(
                device_alignments,
                downloaded_alignments
            )
            ||
            device_alignments.count != reference_alignments.size()
            ||
            downloaded_alignments.size() != reference_alignments.size()
            ||
            downloaded_alignments.size() != 5
        ) {
            return fail("Device-native alignment count mismatch");
        }

        // Both paths run the same expansion kernel. Compare
        // the complete GPU-generated output records bytewise.
        for (std::size_t i = 0; i < reference_alignments.size(); ++i) {
            if (
                std::memcmp(
                    &reference_alignments[i],
                    &downloaded_alignments[i],
                    sizeof(GpuAlignmentCandidate)
                ) != 0
            ) {
                return fail("Device-native alignment record mismatch");
            }
        }

        cuda_free_alignment_candidates(device_alignments);

        // Invalid ligand triplet index must be rejected.
        const std::vector<GpuLigandTripletOrders>
            truncated_orders{ligand_orders[0]};

        if (
            cuda_expand_alignment_candidates_device(
                device_candidates,
                truncated_orders,
                device_offsets,
                device_alignments
            )
            ||
            device_alignments.alignments != nullptr
            ||
            device_alignments.count != 0
        ) {
            return fail("Invalid ligand triplet index not rejected");
        }

        // Corrupt one intermediate offset.
        std::vector<std::uint32_t> invalid_host_offsets =
            alignment_offsets;

        invalid_host_offsets[1] = 1;

        const std::size_t offset_bytes =
            invalid_host_offsets.size() * sizeof(std::uint32_t);

        if (
            cudaMalloc(
                reinterpret_cast<void**>(&modified_offsets.offsets),
                offset_bytes
            ) != cudaSuccess
        ) {
            return fail("Invalid offset allocation failed");
        }

        modified_offsets.count =
            static_cast<std::uint32_t>(invalid_host_offsets.size());

        if (
            cudaMemcpy(
                modified_offsets.offsets,
                invalid_host_offsets.data(),
                offset_bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            return fail("Invalid offset upload failed");
        }

        if (
            cuda_expand_alignment_candidates_device(
                device_candidates,
                ligand_orders,
                modified_offsets,
                device_alignments
            )
            ||
            device_alignments.alignments != nullptr
            ||
            device_alignments.count != 0
        ) {
            return fail("Invalid alignment offsets not rejected");
        }

        // Empty candidates and offsets [0] must succeed
        // without allocating alignment output.
        GpuAlignmentCountArrayDevice empty_counts{};
        GpuTripletCandidateArrayDevice empty_candidates{};

        if (
            !cuda_exclusive_prefix_sum_device(
                empty_counts,
                empty_offsets
            )
            ||
            !cuda_expand_alignment_candidates_device(
                empty_candidates,
                ligand_orders,
                empty_offsets,
                device_alignments
            )
            ||
            device_alignments.alignments != nullptr
            ||
            device_alignments.count != 0
        ) {
            return fail("Empty device-native expansion failed");
        }

        release();

        if (
            device_candidates.candidates != nullptr ||
            device_candidates.count != 0 ||
            device_counts.counts != nullptr ||
            device_counts.count != 0 ||
            device_offsets.offsets != nullptr ||
            device_offsets.count != 0 ||
            device_alignments.alignments != nullptr ||
            device_alignments.count != 0 ||
            modified_offsets.offsets != nullptr ||
            modified_offsets.count != 0 ||
            empty_offsets.offsets != nullptr ||
            empty_offsets.count != 0
        ) {
            cuda_free_triplet_index(device_index);
            std::cerr << "Device-native expansion cleanup failed\n";
            return 1;
        }

        std::cout
            << "CUDA device-native alignment expansion equivalence passed\n";
    }


    std::vector<GpuAlignmentCandidate>
        alignments;


    if (
        !cuda_expand_alignment_candidates(
            candidates,
            ligand_orders,
            alignment_offsets,
            alignments
        )
    ) {
        std::cerr
            << "CUDA alignment expansion failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (alignments.size() != 5) {

        std::cerr
            << "Unexpected alignment count: "
            << alignments.size()
            << '\n';

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * First receptor candidate,
    * first ligand order.
    */
    if (
        alignments[0].ligand_indices[0] != 100 ||
        alignments[0].ligand_indices[1] != 101 ||
        alignments[0].ligand_indices[2] != 102 ||
        alignments[0].protein_indices[0] != 0 ||
        alignments[0].protein_indices[1] != 1 ||
        alignments[0].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected alignment 0\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * First receptor candidate,
    * second ligand order.
    */
    if (
        alignments[1].ligand_indices[0] != 102 ||
        alignments[1].ligand_indices[1] != 101 ||
        alignments[1].ligand_indices[2] != 100 ||
        alignments[1].protein_indices[0] != 0 ||
        alignments[1].protein_indices[1] != 1 ||
        alignments[1].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected alignment 1\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * Third receptor candidate belongs to
    * ligand triplet 1.
    */
    if (
        alignments[4].ligand_indices[0] != 200 ||
        alignments[4].ligand_indices[1] != 201 ||
        alignments[4].ligand_indices[2] != 202 ||
        alignments[4].protein_indices[0] != 3 ||
        alignments[4].protein_indices[1] != 4 ||
        alignments[4].protein_indices[2] != 5
    ) {
        std::cerr
            << "Unexpected alignment 4\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA equivalent-order expansion passed\n";


    // ========================================================
    // Cleanup
    // ========================================================

    cuda_free_triplet_index(
        device_index
    );


    std::cout
        << "All CUDA triplet lookup tests passed\n";


    return 0;
}