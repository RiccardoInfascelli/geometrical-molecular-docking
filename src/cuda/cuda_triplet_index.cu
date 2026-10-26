
#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>


namespace
{

template <typename T>
bool upload_vector(
    const std::vector<T>& host,
    T*& device
)
{
    device = nullptr;

    if (host.empty()) {
        return true;
    }


    const std::size_t bytes =
        host.size() * sizeof(T);


    const cudaError_t allocation_status =
        cudaMalloc(
            reinterpret_cast<void**>(
                &device
            ),
            bytes
        );


    if (allocation_status != cudaSuccess) {
        device = nullptr;
        return false;
    }


    const cudaError_t copy_status =
        cudaMemcpy(
            device,
            host.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


    if (copy_status != cudaSuccess) {

        cudaFree(device);

        device = nullptr;

        return false;
    }


    return true;
}

} // namespace


void cuda_free_triplet_index(
    GpuFlatTripletIndexDevice& device_index
)
{
    if (device_index.keys != nullptr) {
        cudaFree(device_index.keys);
    }


    if (device_index.offsets != nullptr) {
        cudaFree(device_index.offsets);
    }


    if (device_index.records != nullptr) {
        cudaFree(device_index.records);
    }


    device_index = {};
}


bool cuda_upload_triplet_index(
    const GpuFlatTripletIndexHost& host_index,
    GpuFlatTripletIndexDevice& device_index
)
{
    constexpr std::size_t max_u32 =
        std::numeric_limits<
            std::uint32_t
        >::max();


    if (
        host_index.keys.size() > max_u32 ||
        host_index.offsets.size() > max_u32 ||
        host_index.records.size() > max_u32
    ) {
        return false;
    }


    GpuFlatTripletIndexDevice uploaded{};


    if (
        !upload_vector(
            host_index.keys,
            uploaded.keys
        )
    ) {
        return false;
    }


    if (
        !upload_vector(
            host_index.offsets,
            uploaded.offsets
        )
    ) {
        cuda_free_triplet_index(uploaded);

        return false;
    }


    if (
        !upload_vector(
            host_index.records,
            uploaded.records
        )
    ) {
        cuda_free_triplet_index(uploaded);

        return false;
    }


    uploaded.key_count =
        static_cast<std::uint32_t>(
            host_index.keys.size()
        );

    uploaded.offset_count =
        static_cast<std::uint32_t>(
            host_index.offsets.size()
        );

    uploaded.record_count =
        static_cast<std::uint32_t>(
            host_index.records.size()
        );


    /*
     * Sostituiamo un eventuale indice
     * precedentemente residente solo dopo
     * che il nuovo upload è riuscito.
     */
    cuda_free_triplet_index(
        device_index
    );

    device_index = uploaded;


    return true;
}