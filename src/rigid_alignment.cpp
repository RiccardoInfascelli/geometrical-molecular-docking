#include "rigid_alignment.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace
{

bool rigid_transform_from_points_svd(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform)
{
    // Each column holds one point, with corresponding columns in P and Q.
    Eigen::Matrix<double, 3, 3> P;
    Eigen::Matrix<double, 3, 3> Q;

    for (int i = 0; i < 3; ++i) {
        P.col(i) = Eigen::Vector3d(
            source[i].x,
            source[i].y,
            source[i].z
        );
        Q.col(i) = Eigen::Vector3d(
            target[i].x,
            target[i].y,
            target[i].z
        );
    }

    const Eigen::Vector3d p_centroid = P.rowwise().mean();
    const Eigen::Vector3d q_centroid = Q.rowwise().mean();
    const Eigen::Matrix3d Pc = P.colwise() - p_centroid;
    const Eigen::Matrix3d Qc = Q.colwise() - q_centroid;
    const Eigen::Matrix3d H = Pc * Qc.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        H,
        Eigen::ComputeFullU | Eigen::ComputeFullV
    );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();

    // With H = Pc * Qc^T, the source-to-target rotation is V * D * U^T.
    // Correct a negative determinant to exclude reflections.
    if ((V * U.transpose()).determinant() < 0.0) {
        D(2, 2) = -1.0;
    }

    const Eigen::Matrix3d R = V * D * U.transpose();
    const Eigen::Vector3d t = q_centroid - R * p_centroid;

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            transform.rotation[row * 3 + col] = R(row, col);
        }
    }
    transform.translation = {
        t.x(),
        t.y(),
        t.z()
    };

    return true;
}

// Horn/Davenport quaternion alignment for three pairs of points.
// Same least-squares proper-rotation objective as Eigen Kabsch/SVD.
// Jacobi diagonalization mirrors the experimental CUDA quaternion kernel.
bool rigid_transform_from_points_quaternion(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform)
{
    double cp[3] = {0.0, 0.0, 0.0};
    double cq[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        cp[0] += source[i].x; cp[1] += source[i].y; cp[2] += source[i].z;
        cq[0] += target[i].x; cq[1] += target[i].y; cq[2] += target[i].z;
    }
    for (int k = 0; k < 3; ++k) {
        cp[k] /= 3.0;
        cq[k] /= 3.0;
    }

    // Row-major source-target covariance, S = sum (p-cp)*(q-cq)^T.
    double S[9] = {};
    for (int i = 0; i < 3; ++i) {
        const double pp[3] = {
            source[i].x - cp[0], source[i].y - cp[1], source[i].z - cp[2]
        };
        const double qq[3] = {
            target[i].x - cq[0], target[i].y - cq[1], target[i].z - cq[2]
        };
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                S[3*r+c] += pp[r]*qq[c];
    }

    const double xx=S[0], xy=S[1], xz=S[2];
    const double yx=S[3], yy=S[4], yz=S[5];
    const double zx=S[6], zy=S[7], zz=S[8];
    double K[4][4] = {
        {xx+yy+zz, yz-zy,     zx-xz,     xy-yx},
        {yz-zy,     xx-yy-zz, xy+yx,     xz+zx},
        {zx-xz,     xy+yx,    -xx+yy-zz, yz+zy},
        {xy-yx,     xz+zx,     yz+zy,   -xx-yy+zz}
    };
    double V[4][4] = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    // Symmetric 4x4 Jacobi eigensolver: largest eigenvector is [w,x,y,z].
    for (int sweep = 0; sweep < 20; ++sweep) {
        double max_off = 0.0, scale = 0.0;
        for (int r = 0; r < 4; ++r) {
            const double d = std::fabs(K[r][r]);
            if (d > scale) scale = d;
            for (int c = r+1; c < 4; ++c) {
                const double off = std::fabs(K[r][c]);
                if (off > max_off) max_off = off;
            }
        }
        if (max_off <= 1.0e-13*(scale+1.0e-30)) break;
        for (int r = 0; r < 3; ++r) {
            for (int c = r+1; c < 4; ++c) {
                const double off = K[r][c];
                if (std::fabs(off) <= 1.0e-15*(scale+1.0e-30)) continue;
                const double tau = (K[c][c] - K[r][r])/(2.0*off);
                const double t = (tau >= 0.0 ? 1.0 : -1.0)/
                                 (std::fabs(tau) + std::sqrt(1.0 + tau*tau));
                const double cs = 1.0/std::sqrt(1.0 + t*t);
                const double sn = t*cs;
                K[r][r] -= t*off;
                K[c][c] += t*off;
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
    // Equal largest eigenvalues keep the first column; a degenerate triple
    // need not determine a unique optimal rotation.
    int best = 0;
    for (int k = 1; k < 4; ++k)
        if (K[k][k] > K[best][best]) best = k;

    double w=V[0][best], x=V[1][best], y=V[2][best], z=V[3][best];
    const double norm2 = w*w + x*x + y*y + z*z;
    if (!(norm2 > 1.0e-28)) return false;
    const double inv = 1.0/std::sqrt(norm2);
    w*=inv; x*=inv; y*=inv; z*=inv;

    const double R[9] = {
        1.0-2.0*(y*y+z*z), 2.0*(x*y-w*z),     2.0*(x*z+w*y),
        2.0*(x*y+w*z),    1.0-2.0*(x*x+z*z), 2.0*(y*z-w*x),
        2.0*(x*z-w*y),     2.0*(y*z+w*x),    1.0-2.0*(x*x+y*y)
    };
    for (int i = 0; i < 9; ++i) transform.rotation[i] = R[i];
    for (int r = 0; r < 3; ++r) {
        transform.translation[r] = cq[r] -
            R[3*r]*cp[0] - R[3*r+1]*cp[1] - R[3*r+2]*cp[2];
    }
    return true;
}

} // namespace

// Public CPU API: SVD by default, quaternions only when explicitly requested.
bool rigid_transform_from_points(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform)
{
    // Cached once: do not call getenv millions of times during matching.
    static const bool use_quaternion = [] {
        const char* mode = std::getenv("DOCKING_KABSCH_METHOD");
        return mode != nullptr && std::strcmp(mode, "quaternion") == 0;
    }();
    if (use_quaternion)
        return rigid_transform_from_points_quaternion(source, target, transform);
    return rigid_transform_from_points_svd(source, target, transform);
}
