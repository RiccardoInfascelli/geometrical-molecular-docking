#include "quaternion_rmsd.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace conformer_experiment {
namespace {

using Point = std::array<double, 3>;
using Cloud = std::vector<Point>;
using WidePoint = std::array<long double, 3>;

long double relative_coordinate(double value, double origin, long double coordinate_scale)
{
    const long double difference = static_cast<long double>(value) - static_cast<long double>(origin);
    if (std::isfinite(difference)) return difference / coordinate_scale;
    // Platforms whose long double is double cannot subtract opposite extreme
    // coordinates before scaling. Both normalized terms are bounded by one.
    return static_cast<long double>(value) / coordinate_scale -
           static_cast<long double>(origin) / coordinate_scale;
}

WidePoint centroid_offset(const Cloud& cloud, long double coordinate_scale)
{
    WidePoint result{};
    for (const auto& point : cloud)
        for (std::size_t axis = 0; axis < 3; ++axis)
            result[axis] += relative_coordinate(point[axis], cloud.front()[axis], coordinate_scale);
    for (auto& value : result) value /= static_cast<long double>(cloud.size());
    return result;
}

WidePoint centered(const Point& point, const Point& origin, const WidePoint& center_offset,
                   long double coordinate_scale, long double shape_scale = 1.0L)
{
    WidePoint result{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = (relative_coordinate(point[axis], origin[axis], coordinate_scale) -
                        center_offset[axis]) / shape_scale;
    return result;
}

} // namespace

double quaternion_aligned_rmsd(const Cloud& source, const Cloud& target)
{
    if (source.empty() || source.size() != target.size())
        throw std::invalid_argument("Quaternion RMSD requires equal nonempty coordinate arrays");

    long double coordinate_scale = 0.0L;
    for (const auto* cloud : {&source, &target}) {
        for (const auto& point : *cloud) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double value = point[axis];
                if (!std::isfinite(value))
                    throw std::invalid_argument("Quaternion RMSD coordinates must be finite");
                const double origin = cloud->front()[axis];
                const long double difference = static_cast<long double>(value) - static_cast<long double>(origin);
                const long double extent = std::isfinite(difference) ? std::abs(difference) :
                    std::max(std::abs(static_cast<long double>(value)), std::abs(static_cast<long double>(origin)));
                coordinate_scale = std::max(coordinate_scale, extent);
            }
        }
    }
    if (source.size() == 1 || coordinate_scale == 0.0L) return 0.0;

    // A common shape scale preserves the least-squares objective and prevents
    // large finite coordinate differences from overflowing subsequent work.
    // Absolute origins do not set this scale: a huge translation must not erase
    // a small shape, including when the other cloud is entirely coincident.
    // Center offsets from each cloud's first point, so translations cancel
    // before accumulation and a coincident cloud remains exactly centered.
    const auto source_center = centroid_offset(source, coordinate_scale);
    const auto target_center = centroid_offset(target, coordinate_scale);
    long double shape_scale = 0.0L;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const auto p = centered(source[i], source.front(), source_center, coordinate_scale);
        const auto q = centered(target[i], target.front(), target_center, coordinate_scale);
        for (std::size_t axis = 0; axis < 3; ++axis)
            shape_scale = std::max({shape_scale, std::abs(p[axis]), std::abs(q[axis])});
    }
    if (!std::isfinite(shape_scale))
        throw std::runtime_error("Quaternion RMSD centering produced nonfinite coordinates");
    if (shape_scale == 0.0L) return 0.0; // Both clouds consist of coincident points.

    // Match the engine convention S = sum (source-cs)*(target-ct)^T.
    // Divide by N before the double eigensolve; a positive scalar leaves its
    // optimal eigenvector unchanged. A second common scale protects tiny shapes.
    std::array<long double, 9> covariance{};
    for (std::size_t i = 0; i < source.size(); ++i) {
        const auto p = centered(source[i], source.front(), source_center, coordinate_scale, shape_scale);
        const auto q = centered(target[i], target.front(), target_center, coordinate_scale, shape_scale);
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t col = 0; col < 3; ++col)
                covariance[3 * row + col] += p[row] * q[col];
    }
    Eigen::Matrix3d s;
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t col = 0; col < 3; ++col)
            s(row, col) = static_cast<double>(covariance[3 * row + col] /
                                               static_cast<long double>(source.size()));
    const double xx = s(0, 0), xy = s(0, 1), xz = s(0, 2);
    const double yx = s(1, 0), yy = s(1, 1), yz = s(1, 2);
    const double zx = s(2, 0), zy = s(2, 1), zz = s(2, 2);
    Eigen::Matrix4d horn;
    horn << xx + yy + zz, yz - zy, zx - xz, xy - yx,
            yz - zy, xx - yy - zz, xy + yx, xz + zx,
            zx - xz, xy + yx, -xx + yy - zz, yz + zy,
            xy - yx, xz + zx, yz + zy, -xx - yy + zz;
    if (!horn.allFinite())
        throw std::runtime_error("Quaternion RMSD covariance is nonfinite");
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(horn);
    if (solver.info() != Eigen::Success || !solver.eigenvectors().allFinite())
        throw std::runtime_error("Quaternion RMSD eigendecomposition failed");

    // Eigen returns ascending eigenvalues; the last eigenvector is [w,x,y,z]
    // maximizing Horn's objective. Degeneracy permits any vector in that space.
    const Eigen::Vector4d eigenvector = solver.eigenvectors().col(3);
    Eigen::Quaterniond quaternion(eigenvector[0], eigenvector[1], eigenvector[2], eigenvector[3]);
    const double norm = quaternion.norm();
    if (!std::isfinite(norm) || norm <= 0.0)
        throw std::runtime_error("Quaternion RMSD produced an invalid quaternion");
    quaternion.normalize();
    const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();
    if (!rotation.allFinite())
        throw std::runtime_error("Quaternion RMSD produced a nonfinite rotation");

    // Accumulate actual residuals rather than subtracting two nearly equal
    // eigenvalue/norm expressions, which would lose accuracy for a perfect fit.
    long double squared_error = 0.0L;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const auto p = centered(source[i], source.front(), source_center, coordinate_scale, shape_scale);
        const auto q = centered(target[i], target.front(), target_center, coordinate_scale, shape_scale);
        for (std::size_t row = 0; row < 3; ++row) {
            long double residual = -q[row];
            for (std::size_t col = 0; col < 3; ++col)
                residual += static_cast<long double>(rotation(row, col)) * p[col];
            squared_error += residual * residual;
        }
    }
    const long double rmsd = std::sqrt(squared_error / static_cast<long double>(source.size())) *
                             shape_scale * coordinate_scale;
    if (!std::isfinite(rmsd) || rmsd > std::numeric_limits<double>::max())
        throw std::overflow_error("Quaternion aligned RMSD does not fit a finite double");
    return static_cast<double>(rmsd);
}

} // namespace conformer_experiment
