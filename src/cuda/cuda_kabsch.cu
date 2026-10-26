#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "cuda_backend.hpp"
#include "docking_device.cuh"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <climits>
#include <cstddef>
#include <cstdint>


namespace
{

__device__ double point_component(
    const GpuPoint3& point,
    int component
)
{
    if (component == 0)
        return point.x;

    if (component == 1)
        return point.y;

    return point.z;
}


__global__ void prepare_kabsch_kernel(
    const GpuPoint3* ligand_points,
    const GpuPoint3* protein_points,
    const GpuAlignmentCandidate* alignments,
    std::uint32_t alignment_count,
    double* covariance_matrices,
    GpuPoint3* ligand_centroids,
    GpuPoint3* protein_centroids
)
{
    const std::uint32_t index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (index >= alignment_count)
        return;


    const auto alignment =
        alignments[index];


    GpuPoint3 source[3];
    GpuPoint3 target[3];


    for (int i = 0; i < 3; ++i) {

        source[i] =
            ligand_points[
                alignment.ligand_indices[i]
            ];

        target[i] =
            protein_points[
                alignment.protein_indices[i]
            ];
    }


    GpuPoint3 source_centroid{};
    GpuPoint3 target_centroid{};


    for (int i = 0; i < 3; ++i) {

        source_centroid.x +=
            source[i].x;

        source_centroid.y +=
            source[i].y;

        source_centroid.z +=
            source[i].z;


        target_centroid.x +=
            target[i].x;

        target_centroid.y +=
            target[i].y;

        target_centroid.z +=
            target[i].z;
    }


    source_centroid.x /= 3.0;
    source_centroid.y /= 3.0;
    source_centroid.z /= 3.0;

    target_centroid.x /= 3.0;
    target_centroid.y /= 3.0;
    target_centroid.z /= 3.0;


    ligand_centroids[index] =
        source_centroid;

    protein_centroids[index] =
        target_centroid;


    /*
     * H = Pc * Qc^T
     *
     * cuSOLVER expects each matrix
     * in column-major storage.
     */
    double* H =
        covariance_matrices +
        static_cast<std::size_t>(index) * 9;


    for (int row = 0; row < 3; ++row) {

        for (int col = 0; col < 3; ++col) {

            double value = 0.0;


            for (int i = 0; i < 3; ++i) {

                const double p =
                    point_component(
                        source[i],
                        row
                    ) -
                    point_component(
                        source_centroid,
                        row
                    );


                const double q =
                    point_component(
                        target[i],
                        col
                    ) -
                    point_component(
                        target_centroid,
                        col
                    );


                value += p * q;
            }


            H[
                row +
                3 * col
            ] = value;
        }
    }
}


__device__ double matrix3_get(
    const double* matrix,
    int row,
    int col
)
{
    return matrix[
        row +
        3 * col
    ];
}


__global__ void finalize_kabsch_kernel(
    const double* U_matrices,
    const double* V_matrices,
    const int* info,
    const GpuPoint3* ligand_centroids,
    const GpuPoint3* protein_centroids,
    std::uint32_t alignment_count,
    GpuRigidTransform* transforms
)
{
    const std::uint32_t index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (index >= alignment_count)
        return;


    GpuRigidTransform output{};


    // cuSOLVER success is separate from geometric degeneracy: no rank or
    // finiteness test is performed here, and degenerate fits may not be unique.
    if (info[index] != 0) {

        output.valid = 0;

        transforms[index] =
            output;

        return;
    }


    const double* U =
        U_matrices +
        static_cast<std::size_t>(index) * 9;

    const double* V =
        V_matrices +
        static_cast<std::size_t>(index) * 9;


    /*
     * First compute
     *
     * M = V * U^T
     *
     * in order to determine whether
     * reflection correction is required.
     */

    double M[9];


    for (int row = 0; row < 3; ++row) {

        for (int col = 0; col < 3; ++col) {

            double value = 0.0;


            for (int k = 0; k < 3; ++k) {

                value +=
                    matrix3_get(
                        V,
                        row,
                        k
                    ) *
                    matrix3_get(
                        U,
                        col,
                        k
                    );
            }


            M[
                row * 3 +
                col
            ] = value;
        }
    }


    const double determinant =
          M[0] * (
              M[4] * M[8] -
              M[5] * M[7]
          )
        - M[1] * (
              M[3] * M[8] -
              M[5] * M[6]
          )
        + M[2] * (
              M[3] * M[7] -
              M[4] * M[6]
          );


    const double d2 =
        determinant < 0.0
        ? -1.0
        : 1.0;


    /*
     * R = V * D * U^T
     *
     * D = diag(1, 1, d2)
     */
    for (int row = 0; row < 3; ++row) {

        for (int col = 0; col < 3; ++col) {

            double value = 0.0;


            for (int k = 0; k < 3; ++k) {

                const double correction =
                    (k == 2)
                    ? d2
                    : 1.0;


                value +=
                    matrix3_get(
                        V,
                        row,
                        k
                    ) *
                    correction *
                    matrix3_get(
                        U,
                        col,
                        k
                    );
            }


            output.rotation[
                row * 3 +
                col
            ] = value;
        }
    }


    const GpuPoint3 p =
        ligand_centroids[index];

    const GpuPoint3 q =
        protein_centroids[index];


    output.translation[0] =
        q.x -
        (
            output.rotation[0] * p.x +
            output.rotation[1] * p.y +
            output.rotation[2] * p.z
        );


    output.translation[1] =
        q.y -
        (
            output.rotation[3] * p.x +
            output.rotation[4] * p.y +
            output.rotation[5] * p.z
        );


    output.translation[2] =
        q.z -
        (
            output.rotation[6] * p.x +
            output.rotation[7] * p.y +
            output.rotation[8] * p.z
        );


    output.valid = 1;


    transforms[index] =
        output;
}


// Horn/Davenport fit in double precision, one thread per alignment.
// The largest eigenvector of the symmetric 4x4 matrix gives a proper rotation.
// Keep the historical Jacobi limits and operation order for comparisons.
__global__ void quaternion_kabsch_kernel(
    const GpuPoint3* ligand_points,
    const GpuPoint3* protein_points,
    const GpuAlignmentCandidate* alignments,
    std::uint32_t count,
    GpuRigidTransform* transforms
)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    transforms[index] = docking_device::quaternion_fit(
        ligand_points, protein_points, alignments[index]);
}

} // namespace


bool cuda_compute_kabsch_transforms_device(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const GpuAlignmentCandidateArrayDevice& alignments,
    GpuRigidTransformArrayDevice& transforms
)
{
    cuda_free_rigid_transforms(
        transforms
    );


    if (alignments.count == 0)
        return true;


    if (
        alignments.alignments == nullptr ||
        ligand_points.points == nullptr ||
        protein_points.points == nullptr
    ) {
        return false;
    }


    if (
        alignments.count >
        static_cast<std::uint32_t>(
            INT_MAX
        )
    ) {
        return false;
    }


    const std::uint32_t alignment_count =
        alignments.count;


    // Read per call, as in the historical implementation. Only "quaternion"
    // selects the alternative; unset, "svd" and other values retain cuSOLVER.
    const char* kabsch_mode = std::getenv("DOCKING_KABSCH_METHOD");
    if (kabsch_mode != nullptr &&
        std::strcmp(kabsch_mode, "quaternion") == 0) {
        const auto start = std::chrono::steady_clock::now();
        GpuRigidTransform* result = nullptr;
        const std::size_t bytes = static_cast<std::size_t>(alignment_count)
                                * sizeof(GpuRigidTransform);
        if (cudaMalloc(reinterpret_cast<void**>(&result), bytes) != cudaSuccess)
            return false;
        constexpr std::uint32_t threads = 256;
        const std::uint32_t blocks = (alignment_count + threads - 1) / threads;
        quaternion_kabsch_kernel<<<blocks, threads>>>(
            ligand_points.points, protein_points.points,
            alignments.alignments, alignment_count, result);
        // Preserve the historical completion/error check, even without timings.
        // No host copy is needed, but this branch synchronizes the whole device.
        if (cudaGetLastError() != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(result);
            return false;
        }
        transforms.transforms = result;
        transforms.count = alignment_count;
        if (std::getenv("DOCKING_KABSCH_TIMINGS") != nullptr) {
            const auto end = std::chrono::steady_clock::now();
            std::cout << "CUDA_KABSCH_MS,quaternion_total,"
                      << std::chrono::duration<double, std::milli>(end-start).count()
                      << '\n';
        }
        return true;
    }


    const int batch_size =
        static_cast<int>(
            alignment_count
        );


    // Diagnostic only: enabling these checkpoints adds device synchronizations
    // and changes timing, so use separate runs for performance measurements.
    const bool enable_kabsch_timings =
        std::getenv("DOCKING_KABSCH_TIMINGS") != nullptr;

    using KabschClock = std::chrono::steady_clock;

    auto last_checkpoint = KabschClock::now();

    const auto kabsch_checkpoint =
        [&](const char* label) -> bool
        {
            if (!enable_kabsch_timings)
                return true;

            // Include completion of asynchronous CUDA work.
            if (cudaDeviceSynchronize() != cudaSuccess)
                return false;

            const auto now = KabschClock::now();

            const double milliseconds =
                std::chrono::duration<double, std::milli>(
                    now - last_checkpoint
                ).count();

            std::cout
                << "CUDA_KABSCH_MS,"
                << label
                << ","
                << milliseconds
                << '\n';

            last_checkpoint = now;

            return true;
        };


    double* device_covariance =
        nullptr;

    double* device_singular_values =
        nullptr;

    double* device_u =
        nullptr;

    double* device_v =
        nullptr;

    double* device_work =
        nullptr;

    int* device_info =
        nullptr;

    GpuPoint3* device_ligand_centroids =
        nullptr;

    GpuPoint3* device_protein_centroids =
        nullptr;

    GpuRigidTransform* device_transforms =
        nullptr;


    cusolverDnHandle_t solver =
        nullptr;

    gesvdjInfo_t solver_params =
        nullptr;


    auto cleanup =
        [&]()
        {
            cudaFree(
                device_transforms
            );

            cudaFree(
                device_protein_centroids
            );

            cudaFree(
                device_ligand_centroids
            );

            cudaFree(
                device_info
            );

            cudaFree(
                device_work
            );

            cudaFree(
                device_v
            );

            cudaFree(
                device_u
            );

            cudaFree(
                device_singular_values
            );

            cudaFree(
                device_covariance
            );


            if (
                solver_params != nullptr
            ) {
                cusolverDnDestroyGesvdjInfo(
                    solver_params
                );
            }


            if (
                solver != nullptr
            ) {
                cusolverDnDestroy(
                    solver
                );
            }
        };


    const std::size_t matrix_bytes =
        static_cast<std::size_t>(
            alignment_count
        ) *
        9 *
        sizeof(double);


    const std::size_t singular_bytes =
        static_cast<std::size_t>(
            alignment_count
        ) *
        3 *
        sizeof(double);


    const std::size_t centroid_bytes =
        static_cast<std::size_t>(
            alignment_count
        ) *
        sizeof(GpuPoint3);


    const std::size_t info_bytes =
        static_cast<std::size_t>(
            alignment_count
        ) *
        sizeof(int);


    const std::size_t transform_bytes =
        static_cast<std::size_t>(
            alignment_count
        ) *
        sizeof(GpuRigidTransform);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_covariance
            ),
            matrix_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_singular_values
            ),
            singular_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_u
            ),
            matrix_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_v
            ),
            matrix_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_info
            ),
            info_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_ligand_centroids
            ),
            centroid_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_protein_centroids
            ),
            centroid_bytes
        ) != cudaSuccess ||

        cudaMalloc(
            reinterpret_cast<void**>(
                &device_transforms
            ),
            transform_bytes
        ) != cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (!kabsch_checkpoint("allocations")) {
        cleanup();
        return false;
    }


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            alignment_count +
            block_size - 1
        ) /
        block_size;


    prepare_kabsch_kernel
        <<<block_count, block_size>>>(
            ligand_points.points,
            protein_points.points,
            alignments.alignments,
            alignment_count,
            device_covariance,
            device_ligand_centroids,
            device_protein_centroids
        );


    if (
        cudaGetLastError() !=
        cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (!kabsch_checkpoint("prepare_kernel")) {
        cleanup();
        return false;
    }


    if (
        cusolverDnCreate(
            &solver
        ) != CUSOLVER_STATUS_SUCCESS
    ) {
        cleanup();

        return false;
    }


    if (
        cusolverDnCreateGesvdjInfo(
            &solver_params
        ) != CUSOLVER_STATUS_SUCCESS
    ) {
        cleanup();

        return false;
    }


    constexpr int matrix_size =
        3;


    int work_size =
        0;


    if (
        cusolverDnDgesvdjBatched_bufferSize(
            solver,
            CUSOLVER_EIG_MODE_VECTOR,
            matrix_size,
            matrix_size,
            device_covariance,
            matrix_size,
            device_singular_values,
            device_u,
            matrix_size,
            device_v,
            matrix_size,
            &work_size,
            solver_params,
            batch_size
        ) != CUSOLVER_STATUS_SUCCESS
    ) {
        cleanup();

        return false;
    }


    if (work_size > 0) {

        if (
            cudaMalloc(
                reinterpret_cast<void**>(
                    &device_work
                ),
                static_cast<std::size_t>(
                    work_size
                ) *
                sizeof(double)
            ) != cudaSuccess
        ) {
            cleanup();

            return false;
        }
    }


    if (!kabsch_checkpoint("solver_setup")) {
        cleanup();
        return false;
    }


    if (
        cusolverDnDgesvdjBatched(
            solver,
            CUSOLVER_EIG_MODE_VECTOR,
            matrix_size,
            matrix_size,
            device_covariance,
            matrix_size,
            device_singular_values,
            device_u,
            matrix_size,
            device_v,
            matrix_size,
            device_work,
            work_size,
            device_info,
            solver_params,
            batch_size
        ) != CUSOLVER_STATUS_SUCCESS
    ) {
        cleanup();

        return false;
    }


    if (!kabsch_checkpoint("batched_svd")) {
        cleanup();
        return false;
    }


    finalize_kabsch_kernel
        <<<block_count, block_size>>>(
            device_u,
            device_v,
            device_info,
            device_ligand_centroids,
            device_protein_centroids,
            alignment_count,
            device_transforms
        );


    if (
        cudaGetLastError() !=
            cudaSuccess ||
        cudaDeviceSynchronize() !=
            cudaSuccess
    ) {
        cleanup();

        return false;
    }


    if (!kabsch_checkpoint("finalize_kernel")) {
        cleanup();
        return false;
    }


    /*
     * Transfer ownership of the transform buffer
     * to the caller.
     *
     * Alignment data remains owned by the caller.
     */
    transforms.transforms =
        device_transforms;

    transforms.count =
        alignment_count;


    /*
     * Prevent cleanup() from freeing the returned
     * transform buffer.
     */
    device_transforms =
        nullptr;


    cleanup();


    return true;
}

bool cuda_compute_kabsch_transforms_device(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    GpuRigidTransformArrayDevice& transforms
)
{
    if (
        alignments.size() >
        static_cast<std::size_t>(
            INT_MAX
        )
    ) {
        return false;
    }


    // Reject alignment indices outside the uploaded point arrays.
    for (const auto& alignment :
         alignments) {

        for (int i = 0; i < 3; ++i) {

            if (
                alignment.ligand_indices[i] >=
                    ligand_points.count ||
                alignment.protein_indices[i] >=
                    protein_points.count
            ) {
                return false;
            }
        }
    }


    if (alignments.empty()) {

        cuda_free_rigid_transforms(
            transforms
        );

        return true;
    }


    GpuAlignmentCandidateArrayDevice
        device_alignments{};


    const std::size_t alignment_bytes =
        alignments.size() *
        sizeof(GpuAlignmentCandidate);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_alignments.alignments
            ),
            alignment_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    device_alignments.count =
        static_cast<std::uint32_t>(
            alignments.size()
        );


    if (
        cudaMemcpy(
            device_alignments.alignments,
            alignments.data(),
            alignment_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_alignment_candidates(
            device_alignments
        );

        return false;
    }


    const bool ok =
        cuda_compute_kabsch_transforms_device(
            ligand_points,
            protein_points,
            device_alignments,
            transforms
        );


    cuda_free_alignment_candidates(
        device_alignments
    );


    return ok;
}


bool cuda_download_rigid_transforms(
    const GpuRigidTransformArrayDevice& device_transforms,
    std::vector<GpuRigidTransform>& host_transforms
)
{
    host_transforms.clear();


    if (device_transforms.count == 0)
        return true;


    if (device_transforms.transforms == nullptr)
        return false;


    host_transforms.resize(
        device_transforms.count
    );


    const std::size_t bytes =
        static_cast<std::size_t>(
            device_transforms.count
        ) *
        sizeof(GpuRigidTransform);


    if (
        cudaMemcpy(
            host_transforms.data(),
            device_transforms.transforms,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_transforms.clear();

        return false;
    }


    return true;
}


void cuda_free_rigid_transforms(
    GpuRigidTransformArrayDevice& device_transforms
)
{
    if (device_transforms.transforms != nullptr) {

        cudaFree(
            device_transforms.transforms
        );
    }


    device_transforms.transforms =
        nullptr;

    device_transforms.count =
        0;
}


bool cuda_compute_kabsch_transforms(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    std::vector<GpuRigidTransform>& transforms
)
{
    transforms.clear();


    GpuRigidTransformArrayDevice
        device_transforms;


    if (
        !cuda_compute_kabsch_transforms_device(
            ligand_points,
            protein_points,
            alignments,
            device_transforms
        )
    ) {
        return false;
    }


    const bool download_ok =
        cuda_download_rigid_transforms(
            device_transforms,
            transforms
        );


    cuda_free_rigid_transforms(
        device_transforms
    );


    return download_ok;
}
