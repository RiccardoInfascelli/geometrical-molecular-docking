#ifndef DOCKING_DEVICE_CUH
#define DOCKING_DEVICE_CUH

#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

// Shared scientific device operations for the single-ligand and batch kernels.
// Keep their operation order and clash float rounding consistent across paths.
namespace docking_device
{

// Lexicographic key order and lower-bound search match the baseline index.
inline __device__ int compare_triplet_keys(
    const GpuTripletKey& a,
    const GpuTripletKey& b
)
{
    for (int i = 0; i < 3; ++i) {

        if (a.keys[i] < b.keys[i])
            return -1;

        if (a.keys[i] > b.keys[i])
            return 1;
    }


    for (int i = 0; i < 3; ++i) {

        if (a.distance_bins[i] < b.distance_bins[i])
            return -1;

        if (a.distance_bins[i] > b.distance_bins[i])
            return 1;
    }


    return 0;
}


inline __device__ std::uint32_t find_triplet_key(
    const GpuTripletKey* keys,
    std::uint32_t key_count,
    const GpuTripletKey& query
)
{
    std::uint32_t begin = 0;
    std::uint32_t end = key_count;


    while (begin < end) {

        const std::uint32_t middle =
            begin + (end - begin) / 2;


        const int comparison =
            compare_triplet_keys(
                keys[middle],
                query
            );


        if (comparison < 0) {
            begin = middle + 1;
        }
        else {
            end = middle;
        }
    }


    if (
        begin < key_count &&
        compare_triplet_keys(
            keys[begin],
            query
        ) == 0
    ) {
        return begin;
    }


    return 0xFFFFFFFFu;
}


inline __device__ bool grid_in_bounds(
    const GpuProteinGridDevice& grid,
    int x,
    int y,
    int z
)
{
    return
        x >= 0 &&
        x < grid.nx &&
        y >= 0 &&
        y < grid.ny &&
        z >= 0 &&
        z < grid.nz;
}

inline __device__ std::size_t grid_index(
    const GpuProteinGridDevice& grid,
    int x,
    int y,
    int z
)
{
    return static_cast<std::size_t>(
        x +
        grid.nx *
        (
            y +
            grid.ny * z
        )
    );
}

inline __device__ int to_grid(
    double coordinate,
    double minimum,
    double voxel_size
)
{
    return static_cast<int>(
        floor(
            (
                coordinate -
                minimum
            ) /
            voxel_size
        )
    );
}

inline __device__ GpuPoint3 clash_transform_point(
    const GpuPoint3& point,
    const GpuRigidTransform& transform
)
{
    /*
     * Match the serial numerical semantics.
     *
     * interaction::coordinate_type is float:
     *
     * RDKit coordinates
     *   -> float input point
     *   -> double transform arithmetic
     *   -> float transformed point
     *   -> grid lookup
     *
     * GpuPoint3 remains double elsewhere so Kabsch itself
     * does not lose precision.
     */
    const double x =
        static_cast<double>(
            static_cast<float>(point.x)
        );

    const double y =
        static_cast<double>(
            static_cast<float>(point.y)
        );

    const double z =
        static_cast<double>(
            static_cast<float>(point.z)
        );


    const double transformed_x =
        transform.rotation[0] * x +
        transform.rotation[1] * y +
        transform.rotation[2] * z +
        transform.translation[0];

    const double transformed_y =
        transform.rotation[3] * x +
        transform.rotation[4] * y +
        transform.rotation[5] * z +
        transform.translation[1];

    const double transformed_z =
        transform.rotation[6] * x +
        transform.rotation[7] * y +
        transform.rotation[8] * z +
        transform.translation[2];


    GpuPoint3 result;

    result.x =
        static_cast<double>(
            static_cast<float>(
                transformed_x
            )
        );

    result.y =
        static_cast<double>(
            static_cast<float>(
                transformed_y
            )
        );

    result.z =
        static_cast<double>(
            static_cast<float>(
                transformed_z
            )
        );


    return result;
}

inline __device__ GpuPoint3 fast_transform_point(
    const GpuPoint3& point,
    const GpuRigidTransform& transform
)
{
    // Fast scoring keeps transformed centroids in double. The clash kernel
    // separately rounds atom coordinates to float to match CPU clash semantics.
    GpuPoint3 result;


    result.x =
        transform.rotation[0] * point.x +
        transform.rotation[1] * point.y +
        transform.rotation[2] * point.z +
        transform.translation[0];


    result.y =
        transform.rotation[3] * point.x +
        transform.rotation[4] * point.y +
        transform.rotation[5] * point.z +
        transform.translation[1];


    result.z =
        transform.rotation[6] * point.x +
        transform.rotation[7] * point.y +
        transform.rotation[8] * point.z +
        transform.translation[2];


    return result;
}

// Horn/Davenport fit in double precision, with the historical Jacobi limits.
// The caller supplies one already ordered correspondence of three points.
inline __device__ GpuRigidTransform quaternion_fit(
    const GpuPoint3* ligand_points,
    const GpuPoint3* protein_points,
    const GpuAlignmentCandidate& a
)
{
    GpuPoint3 p[3], q[3];
    double cp[3] = {0.0, 0.0, 0.0};
    double cq[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i) {
        p[i] = ligand_points[a.ligand_indices[i]];
        q[i] = protein_points[a.protein_indices[i]];
        cp[0] += p[i].x; cp[1] += p[i].y; cp[2] += p[i].z;
        cq[0] += q[i].x; cq[1] += q[i].y; cq[2] += q[i].z;
    }
    for (int k = 0; k < 3; ++k) {
        cp[k] /= 3.0;
        cq[k] /= 3.0;
    }

    // S = sum_i (p_i - cp) (q_i - cq)^T, row-major.
    double S[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        const double pp[3] = {p[i].x-cp[0], p[i].y-cp[1], p[i].z-cp[2]};
        const double qq[3] = {q[i].x-cq[0], q[i].y-cq[1], q[i].z-cq[2]};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                S[3*r+c] += pp[r]*qq[c];
    }

    const double xx=S[0], xy=S[1], xz=S[2];
    const double yx=S[3], yy=S[4], yz=S[5];
    const double zx=S[6], zy=S[7], zz=S[8];

    // Symmetric Davenport/Horn K: largest eigenvector is quaternion [w,x,y,z].
    // Signs reflect source-target covariance S=P^T Q.
    double K[4][4] = {
        {xx+yy+zz, yz-zy,     zx-xz,     xy-yx},
        {yz-zy,     xx-yy-zz,  xy+yx,     xz+zx},
        {zx-xz,     xy+yx,    -xx+yy-zz,  yz+zy},
        {xy-yx,     xz+zx,     yz+zy,    -xx-yy+zz}
    };
    double V[4][4] = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    // Jacobi stops at the relative off-diagonal tolerance or after 20 sweeps.
    // There is no rank or convergence status check: coincident or collinear
    // points are not rejected here, and their rotation need not be unique.
    for (int sweep = 0; sweep < 20; ++sweep) {
        double max_off = 0.0;
        double scale = 0.0;
        for (int r = 0; r < 4; ++r) {
            const double d = fabs(K[r][r]);
            if (d > scale) scale = d;
            for (int c = r + 1; c < 4; ++c) {
                const double x = fabs(K[r][c]);
                if (x > max_off) max_off = x;
            }
        }
        if (max_off <= 1.0e-13 * (scale + 1.0e-30)) break;

        for (int r = 0; r < 3; ++r) {
            for (int c = r + 1; c < 4; ++c) {
                const double off = K[r][c];
                if (fabs(off) <= 1.0e-15 * (scale + 1.0e-30)) continue;
                const double tau = (K[c][c] - K[r][r]) / (2.0 * off);
                const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                                 (fabs(tau) + sqrt(1.0 + tau*tau));
                const double cs = 1.0 / sqrt(1.0 + t*t);
                const double sn = t * cs;
                K[r][r] -= t * off;
                K[c][c] += t * off;
                K[r][c] = K[c][r] = 0.0;
                for (int k = 0; k < 4; ++k) {
                    if (k != r && k != c) {
                        const double kr = K[k][r], kc = K[k][c];
                        K[k][r] = K[r][k] = cs*kr - sn*kc;
                        K[k][c] = K[c][k] = sn*kr + cs*kc;
                    }
                    const double vr = V[k][r], vc = V[k][c];
                    V[k][r] = cs*vr - sn*vc;
                    V[k][c] = sn*vr + cs*vc;
                }
            }
        }
    }

    int best = 0;
    for (int k = 1; k < 4; ++k)
        if (K[k][k] > K[best][best]) best = k;

    double w=V[0][best], x=V[1][best], y=V[2][best], z=V[3][best];
    const double norm2 = w*w + x*x + y*y + z*z;
    GpuRigidTransform out{};
    // This norm guard is not a general finite-input or degeneracy check.
    if (!(norm2 > 1.0e-28)) {
        out.valid = 0;
        return out;
    }
    const double inv = 1.0 / sqrt(norm2);
    w*=inv; x*=inv; y*=inv; z*=inv;

    out.rotation[0] = 1.0-2.0*(y*y+z*z);
    out.rotation[1] = 2.0*(x*y-w*z);
    out.rotation[2] = 2.0*(x*z+w*y);
    out.rotation[3] = 2.0*(x*y+w*z);
    out.rotation[4] = 1.0-2.0*(x*x+z*z);
    out.rotation[5] = 2.0*(y*z-w*x);
    out.rotation[6] = 2.0*(x*z-w*y);
    out.rotation[7] = 2.0*(y*z+w*x);
    out.rotation[8] = 1.0-2.0*(x*x+y*y);

    for (int r=0; r<3; ++r) {
        out.translation[r] = cq[r]
            - out.rotation[3*r]*cp[0]
            - out.rotation[3*r+1]*cp[1]
            - out.rotation[3*r+2]*cp[2];
    }
    out.valid = 1;
    return out;
}

// Count every ligand atom, without stopping early at a clash threshold.
// Atoms outside the grid stay in the denominator but are not clashes.
inline __device__ GpuClashScore grid_clash(
    const GpuPoint3* ligand_atoms,
    std::uint32_t ligand_atom_count,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransform& transform
)
{
    GpuClashScore score;


    score.total_atoms =
        ligand_atom_count;


    if (transform.valid == 0) {

        score.valid = 0;

        return score;
    }


    score.valid = 1;


    for (
        std::uint32_t atom_index = 0;
        atom_index < ligand_atom_count;
        ++atom_index
    ) {
        const auto transformed =
            clash_transform_point(
                ligand_atoms[
                    atom_index
                ],
                transform
            );


        const int gx =
            to_grid(
                transformed.x,
                protein_grid.xmin,
                protein_grid.voxel_size
            );


        const int gy =
            to_grid(
                transformed.y,
                protein_grid.ymin,
                protein_grid.voxel_size
            );


        const int gz =
            to_grid(
                transformed.z,
                protein_grid.zmin,
                protein_grid.voxel_size
            );


        /*
         * Same semantics as the serial
         * score_fast_grid_clash:
         *
         * atoms outside the grid do not
         * contribute to the clash count.
         */
        if (
            !grid_in_bounds(
                protein_grid,
                gx,
                gy,
                gz
            )
        ) {
            continue;
        }


        const std::size_t voxel_index =
            grid_index(
                protein_grid,
                gx,
                gy,
                gz
            );


        if (
            protein_grid
                .voxels[
                    voxel_index
                ]
                .occupied != 0
        ) {
            ++score.occupied_atoms;
        }
    }


    if (score.total_atoms != 0) {

        score.fraction =
            static_cast<double>(
                score.occupied_atoms
            ) /
            static_cast<double>(
                score.total_atoms
            );
    }


    return score;
}

// Accumulate area weights in input order, using double centroids.
inline __device__ GpuFastScore fast_score(
    const GpuFastMatchingArea* ligand_areas,
    std::uint32_t ligand_area_count,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransform& transform
)
{
    GpuFastScore score;


    score.total_areas =
        ligand_area_count;


    if (transform.valid == 0) {

        score.valid = 0;

        return score;
    }


    score.valid = 1;


    for (
        std::uint32_t area_index = 0;
        area_index < ligand_area_count;
        ++area_index
    ) {
        const auto area =
            ligand_areas[
                area_index
            ];


        const auto transformed =
            fast_transform_point(
                area.point,
                transform
            );


        const int gx =
            to_grid(
                transformed.x,
                protein_grid.xmin,
                protein_grid.voxel_size
            );


        const int gy =
            to_grid(
                transformed.y,
                protein_grid.ymin,
                protein_grid.voxel_size
            );


        const int gz =
            to_grid(
                transformed.z,
                protein_grid.zmin,
                protein_grid.voxel_size
            );


        if (
            !grid_in_bounds(
                protein_grid,
                gx,
                gy,
                gz
            )
        ) {
            continue;
        }


        ++score.areas_in_grid;


        const std::size_t voxel_index =
            grid_index(
                protein_grid,
                gx,
                gy,
                gz
            );


        const auto voxel =
            protein_grid
                .voxels[
                    voxel_index
                ];


        if (voxel.occupied != 0) {

            ++score
                .areas_on_occupied_voxel;

            continue;
        }


        if (
            (
                voxel.interactions &
                area.interaction_bit
            ) == 0
        ) {
            continue;
        }


        ++score.compatible_areas;


        score.weighted_score +=
            area.weight;
    }


    return score;
}

} // namespace docking_device

#endif
