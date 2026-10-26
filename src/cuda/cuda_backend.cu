#include "cuda_backend.hpp"

#include <cuda_runtime.h>

namespace
{

__global__ void smoke_test_kernel(int* value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *value = 42;
    }
}

} // namespace


bool cuda_smoke_test()
{
    int* device_value = nullptr;

    if (cudaMalloc(&device_value, sizeof(int)) != cudaSuccess) {
        return false;
    }

    smoke_test_kernel<<<1, 1>>>(device_value);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(device_value);
        return false;
    }

    int host_value = 0;

    const cudaError_t copy_status =
        cudaMemcpy(
            &host_value,
            device_value,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );

    cudaFree(device_value);

    if (copy_status != cudaSuccess) {
        return false;
    }

    return host_value == 42;
}