#include "cuda_backend.hpp"

#include <cmath>
#include <iostream>
#include <vector>


namespace
{

bool same_point(
    const GpuPoint3& a,
    const GpuPoint3& b
)
{
    constexpr double tolerance =
        1e-12;

    return
        std::abs(a.x - b.x) < tolerance &&
        std::abs(a.y - b.y) < tolerance &&
        std::abs(a.z - b.z) < tolerance;
}

} // namespace


int main()
{
    const std::vector<GpuPoint3>
        original = {

            {1.0, 2.0, 3.0},
            {-4.5, 5.25, 6.75},
            {10.0, -20.0, 30.0}
        };


    GpuPointArrayDevice device_points;


    // ========================================================
    // Upload
    // ========================================================

    if (
        !cuda_upload_points(
            original,
            device_points
        )
    ) {
        std::cerr
            << "CUDA point upload failed\n";

        return 1;
    }


    if (
        device_points.points == nullptr ||
        device_points.count !=
            original.size()
    ) {
        std::cerr
            << "Invalid device point array\n";

        cuda_free_points(
            device_points
        );

        return 1;
    }


    std::cout
        << "CUDA point upload passed\n";


    // ========================================================
    // Download / round-trip validation
    // ========================================================

    std::vector<GpuPoint3>
        downloaded;


    if (
        !cuda_download_points(
            device_points,
            downloaded
        )
    ) {
        std::cerr
            << "CUDA point download failed\n";

        cuda_free_points(
            device_points
        );

        return 1;
    }


    if (
        downloaded.size() !=
        original.size()
    ) {
        std::cerr
            << "Unexpected downloaded point count\n";

        cuda_free_points(
            device_points
        );

        return 1;
    }


    for (
        std::size_t i = 0;
        i < original.size();
        ++i
    ) {
        if (
            !same_point(
                original[i],
                downloaded[i]
            )
        ) {
            std::cerr
                << "Point mismatch at index "
                << i
                << '\n';

            cuda_free_points(
                device_points
            );

            return 1;
        }
    }


    std::cout
        << "CUDA point round-trip passed\n";


    // ========================================================
    // Cleanup
    // ========================================================

    cuda_free_points(
        device_points
    );


    if (
        device_points.points != nullptr ||
        device_points.count != 0
    ) {
        std::cerr
            << "CUDA point cleanup failed\n";

        return 1;
    }


    std::cout
        << "CUDA geometry upload tests passed\n";


    return 0;
}