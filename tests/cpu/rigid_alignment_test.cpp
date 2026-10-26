#include "rigid_alignment.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

using Points = std::array<interaction::point3D, 3>;
using RowMajorMatrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

void require(bool condition, const char* name, const char* message)
{
    if (!condition) {
        throw std::runtime_error(std::string(name) + ": " + message);
    }
}

Eigen::Matrix3d point_matrix(const Points& points)
{
    Eigen::Matrix3d matrix;
    for (int i = 0; i < 3; ++i) {
        matrix.col(i) = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
    }
    return matrix;
}

Points transformed_points(const Points& points, const RigidTransform& transform)
{
    const Eigen::Map<const RowMajorMatrix> rotation(transform.rotation.data());
    const Eigen::Map<const Eigen::Vector3d> translation(transform.translation.data());
    Points result;
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d point = rotation *
            Eigen::Vector3d(points[i].x, points[i].y, points[i].z) + translation;
        result[i] = {
            static_cast<interaction::coordinate_type>(point.x()),
            static_cast<interaction::coordinate_type>(point.y()),
            static_cast<interaction::coordinate_type>(point.z())
        };
    }
    return result;
}

void check_fit(const char* name, const Points& source, const Points& target,
               const RigidTransform* expected = nullptr)
{
    RigidTransform transform;
    require(rigid_transform_from_points(source, target, transform), name, "Fit failed");
    const Eigen::Map<const RowMajorMatrix> rotation(transform.rotation.data());
    const Eigen::Map<const Eigen::Vector3d> translation(transform.translation.data());
    require(rotation.allFinite() && translation.allFinite(), name,
            "Non-finite transform for finite inputs");
    require((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
            std::abs(rotation.determinant() - 1.0) < 1e-10, name,
            "Fit must produce an orthogonal rotation without reflection");

    const Eigen::Matrix3d p = point_matrix(source);
    const Eigen::Matrix3d q = point_matrix(target);
    const Eigen::Vector3d source_centroid = p.rowwise().mean();
    const Eigen::Vector3d target_centroid = q.rowwise().mean();
    const Eigen::Matrix3d centered_source = p.colwise() - source_centroid;
    const Eigen::Matrix3d centered_target = q.colwise() - target_centroid;
    const Eigen::Matrix3d covariance = centered_source * centered_target.transpose();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // The singular values give the optimal squared residual independently of
    // the returned rotation, which need not be unique for degenerate triples.
    const double orientation =
        (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    const auto singular = svd.singularValues();
    const double energy = centered_source.squaredNorm() + centered_target.squaredNorm();
    const double optimal_error = std::max(0.0, energy -
        2.0 * (singular[0] + singular[1] + orientation * singular[2]));
    const Eigen::Matrix3d fitted = (rotation * p).colwise() + translation;
    require(std::abs((fitted - q).squaredNorm() - optimal_error) <= 1e-9 * (1.0 + energy),
            name, "Residual differs from the SVD least-squares objective");
    require((rotation * source_centroid + translation - target_centroid).norm() <=
            1e-10 * (1.0 + source_centroid.norm() + target_centroid.norm()), name,
            "Transformed centroids do not coincide");

    if (expected) {
        const Eigen::Map<const RowMajorMatrix> expected_rotation(expected->rotation.data());
        const Eigen::Map<const Eigen::Vector3d> expected_translation(expected->translation.data());
        require((rotation - expected_rotation).norm() < 1e-9 &&
                (translation - expected_translation).norm() < 1e-9, name,
                "Known rigid transform was not recovered");
    }
}

void test_alignment()
{
    const Points source{{{0.25, -2.0, 3.0}, {2.25, -1.0, 4.0}, {1.25, 2.0, 0.0}}};
    RigidTransform transform;
    check_fit("identity", source, source, &transform);
    transform.translation = {5.0, -3.0, 7.0};
    check_fit("translation", source, transformed_points(source, transform), &transform);

    // An exact cyclic rotation about (1,1,1) avoids float rounding in the fixture.
    transform.rotation = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    const Points rotated = transformed_points(source, transform);
    check_fit("oblique rotation", source, rotated, &transform);
    check_fit("reordered pairs", {source[2], source[0], source[1]},
              {rotated[2], rotated[0], rotated[1]}, &transform);

    transform.rotation = {-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0};
    check_fit("half turn", source, transformed_points(source, transform), &transform);

    Points distorted = rotated;
    distorted[0].x += 0.125;
    distorted[1].y -= 0.25;
    check_fit("noncongruent triangles", source, distorted);
    Points reflected = source;
    for (auto& point : reflected) point.x = -point.x;
    check_fit("reflected triangle", source, reflected);

    const Points collinear{{{-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}};
    const Points coincident{{{1.0, -2.0, 0.5}, {1.0, -2.0, 0.5}, {1.0, -2.0, 0.5}}};
    Points near_collinear = collinear;
    near_collinear[1].y = 1e-5f;
    check_fit("near collinear", near_collinear, transformed_points(near_collinear, transform));
    check_fit("collinear", collinear, transformed_points(collinear, transform));
    check_fit("coincident", coincident, transformed_points(coincident, transform));
    check_fit("collapsed source", coincident, rotated);
}

} // namespace

int main()
{
    try {
        // Run in separate processes: the production CPU selector is cached once.
        test_alignment();
        const char* mode = std::getenv("DOCKING_KABSCH_METHOD");
        const bool quaternion = mode && std::strcmp(mode, "quaternion") == 0;
        std::cout << "CPU rigid alignment tests passed ("
                  << (quaternion ? "quaternion" : "svd") << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
