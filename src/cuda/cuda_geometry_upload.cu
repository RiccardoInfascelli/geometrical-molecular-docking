#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>


void cuda_free_points(
    GpuPointArrayDevice& device_points
)
{
    if (device_points.points != nullptr) {

        cudaFree(
            device_points.points
        );
    }


    device_points =
        GpuPointArrayDevice{};
}


bool cuda_upload_points(
    const std::vector<GpuPoint3>& host_points,
    GpuPointArrayDevice& device_points
)
{
    if (
        host_points.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    /*
     * Empty input is valid.
     */
    if (host_points.empty()) {

        cuda_free_points(
            device_points
        );

        return true;
    }


    GpuPointArrayDevice replacement;


    const std::size_t bytes =
        host_points.size() *
        sizeof(GpuPoint3);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &replacement.points
            ),
            bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMemcpy(
            replacement.points,
            host_points.data(),
            bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_points(
            replacement
        );

        return false;
    }


    replacement.count =
        static_cast<std::uint32_t>(
            host_points.size()
        );


    /*
     * Replace the previous device array only
     * after the new upload succeeded.
     */
    cuda_free_points(
        device_points
    );


    device_points =
        replacement;


    return true;
}


bool cuda_download_points(
    const GpuPointArrayDevice& device_points,
    std::vector<GpuPoint3>& host_points
)
{
    host_points.clear();


    if (device_points.count == 0)
        return true;


    if (device_points.points == nullptr)
        return false;


    host_points.resize(
        device_points.count
    );


    const std::size_t bytes =
        static_cast<std::size_t>(
            device_points.count
        ) *
        sizeof(GpuPoint3);


    if (
        cudaMemcpy(
            host_points.data(),
            device_points.points,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_points.clear();
        return false;
    }


    return true;
}