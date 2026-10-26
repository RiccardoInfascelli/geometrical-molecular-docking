#ifndef CUDA_TRIPLET_INDEX_HPP
#define CUDA_TRIPLET_INDEX_HPP

#include "cuda_triplet_types.hpp"
#include "triplet_generation.hpp"

#include <vector>


// CPU-side packing only: preserve flat index key, bucket and record order.
// Device allocation/upload and lookup are implemented separately in CUDA.
GpuFlatTripletIndexHost
make_gpu_flat_triplet_index(
    const FlatTripletIndex& index
);


bool validate_gpu_flat_triplet_index(
    const FlatTripletIndex& cpu_index,
    const GpuFlatTripletIndexHost& gpu_index
);


// Host-only preparation: clear outputs, then preserve input triplet order and the
// sequence of equivalent vertex orders. DOCKING_QUERY_METHOD selects reference
// (default) or optimized on each call;
// optimized bins each edge once, with the same canonical signatures and ties.
// As in canonicalize_triplet, require a finite positive bin_size and finite,
// nonnegative distances whose bins fit size_t. GPU keys/indices must fit their
// packed integer types; overflow throws and may leave partial outputs.
// An unknown method throws invalid_argument before changing either output.
void make_gpu_ligand_triplet_queries(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
);


#endif
