#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "triplet_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace
{

double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    return std::sqrt(
        interaction::distance2(a, b)
    );
}


double cross_norm_squared(
    const interaction::point3D& a,
    const interaction::point3D& b,
    const interaction::point3D& c)
{
    const double abx =
        static_cast<double>(b.x) -
        static_cast<double>(a.x);

    const double aby =
        static_cast<double>(b.y) -
        static_cast<double>(a.y);

    const double abz =
        static_cast<double>(b.z) -
        static_cast<double>(a.z);


    const double acx =
        static_cast<double>(c.x) -
        static_cast<double>(a.x);

    const double acy =
        static_cast<double>(c.y) -
        static_cast<double>(a.y);

    const double acz =
        static_cast<double>(c.z) -
        static_cast<double>(a.z);


    const double cross_x =
        aby * acz -
        abz * acy;

    const double cross_y =
        abz * acx -
        abx * acz;

    const double cross_z =
        abx * acy -
        aby * acx;


    return
        cross_x * cross_x +
        cross_y * cross_y +
        cross_z * cross_z;
}


std::array<std::size_t, 3> sorted_indices(
    std::array<std::size_t, 3> indices)
{
    std::sort(
        indices.begin(),
        indices.end()
    );

    return indices;
}


void print_usage(
    const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  "
        << program
        << " <complex_directory>\n";
}

} // namespace


int main(
    int argc,
    char* argv[])
{
    try {

        if (argc != 2) {

            print_usage(
                argv[0]
            );

            return 1;
        }


        // ====================================================
        // Paths
        // ====================================================

        std::filesystem::path complex_directory =
            std::filesystem::absolute(
                argv[1]
            ).lexically_normal();


        if (
            complex_directory
                .filename()
                .empty()
        ) {
            complex_directory =
                complex_directory
                    .parent_path();
        }


        const std::string complex_id =
            complex_directory
                .filename()
                .string();


        const auto ligand_path =
            complex_directory /
            (
                complex_id +
                "_ligand.mol2"
            );


        // ====================================================
        // Same ligand preparation used by production pipeline
        // ====================================================

        auto ligand =
            load_ligand(
                ligand_path.string()
            );


        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        const double ligand_diameter =
            compute_ligand_diameter(
                ligand_areas
            );


        const auto ligand_triplets =
            generate_triplets(
                ligand_matching_areas,
                ligand_diameter
            );


        // ====================================================
        // Record triplets actually accepted by production
        // ====================================================

        std::set<
            std::array<std::size_t, 3>
        > accepted_triplets;


        for (const auto& triplet :
             ligand_triplets) {

            accepted_triplets.insert(
                sorted_indices(
                    triplet
                        .matching_area_indices
                )
            );
        }


        std::cout
            << std::fixed
            << std::setprecision(6);


        // ====================================================
        // General info
        // ====================================================

        std::cout
            << "=============================================\n"
            << "TRIPLET FAILURE DIAGNOSTIC\n"
            << "=============================================\n"

            << "Complex: "
            << complex_id
            << '\n'

            << "Ligand atoms: "
            << ligand->getNumAtoms()
            << '\n'

            << "Physical ligand areas: "
            << ligand_areas.size()
            << '\n'

            << "Ligand matching areas: "
            << ligand_matching_areas.size()
            << '\n'

            << "Ligand diameter: "
            << ligand_diameter
            << " A\n"

            << "Production ligand triplets: "
            << ligand_triplets.size()
            << "\n";


        // ====================================================
        // Matching areas
        // ====================================================

        std::cout
            << "\n=== LIGAND MATCHING AREAS ===\n";


        for (
            std::size_t i = 0;
            i < ligand_matching_areas.size();
            ++i
        ) {

            const auto& area =
                ligand_matching_areas[i];


            std::cout
                << "area "
                << i

                << " | source="
                << area.source_area_index

                << " | key="
                << static_cast<int>(
                    area.key
                )

                << " | centroid=("
                << area.centroid.x
                << ", "
                << area.centroid.y
                << ", "
                << area.centroid.z
                << ")\n";
        }


        // ====================================================
        // Pairwise distances
        // ====================================================

        std::cout
            << "\n=== PAIRWISE DISTANCES ===\n";


        for (
            std::size_t i = 0;
            i < ligand_matching_areas.size();
            ++i
        ) {

            for (
                std::size_t j = i + 1;
                j < ligand_matching_areas.size();
                ++j
            ) {

                const auto& a =
                    ligand_matching_areas[i];

                const auto& b =
                    ligand_matching_areas[j];


                const double distance =
                    point_distance(
                        a.centroid,
                        b.centroid
                    );


                std::cout
                    << "("
                    << i
                    << ","
                    << j
                    << ")"

                    << " | sources=("
                    << a.source_area_index
                    << ","
                    << b.source_area_index
                    << ")"

                    << " | d="
                    << distance
                    << " A";


                if (
                    a.source_area_index ==
                    b.source_area_index
                ) {
                    std::cout
                        << " | SAME_SOURCE";
                }


                if (distance < 0.001) {

                    std::cout
                        << " | TOO_CLOSE";
                }


                if (
                    distance >
                    ligand_diameter
                ) {

                    std::cout
                        << " | TOO_FAR";
                }


                std::cout
                    << '\n';
            }
        }


        // ====================================================
        // Every possible combination
        // ====================================================

        std::cout
            << "\n=== ALL CANDIDATE TRIPLETS ===\n";


        std::size_t total_combinations = 0;

        std::size_t rejected_same_source = 0;
        std::size_t rejected_distance = 0;
        std::size_t rejected_aligned = 0;

        std::size_t production_accepted = 0;


        for (
            std::size_t i = 0;
            i < ligand_matching_areas.size();
            ++i
        ) {

            for (
                std::size_t j = i + 1;
                j < ligand_matching_areas.size();
                ++j
            ) {

                for (
                    std::size_t k = j + 1;
                    k < ligand_matching_areas.size();
                    ++k
                ) {

                    ++total_combinations;


                    const auto& a =
                        ligand_matching_areas[i];

                    const auto& b =
                        ligand_matching_areas[j];

                    const auto& c =
                        ligand_matching_areas[k];


                    const double dij =
                        point_distance(
                            a.centroid,
                            b.centroid
                        );

                    const double dik =
                        point_distance(
                            a.centroid,
                            c.centroid
                        );

                    const double djk =
                        point_distance(
                            b.centroid,
                            c.centroid
                        );


                    const bool same_source =
                        a.source_area_index ==
                            b.source_area_index
                        ||
                        a.source_area_index ==
                            c.source_area_index
                        ||
                        b.source_area_index ==
                            c.source_area_index;


                    const bool distance_failure =
                        dij < 0.001 ||
                        dik < 0.001 ||
                        djk < 0.001 ||
                        dij > ligand_diameter ||
                        dik > ligand_diameter ||
                        djk > ligand_diameter;


                    const double cross2 =
                        cross_norm_squared(
                            a.centroid,
                            b.centroid,
                            c.centroid
                        );


                    // Exact threshold used by
                    // triplet_generation.cpp.
                    const bool aligned =
                        cross2 < 1e-6;


                    const std::array<std::size_t, 3>
                        indices{
                            i,
                            j,
                            k
                        };


                    const bool accepted =
                        accepted_triplets.contains(
                            indices
                        );


                    if (accepted)
                        ++production_accepted;


                    if (same_source)
                        ++rejected_same_source;

                    if (distance_failure)
                        ++rejected_distance;

                    if (aligned)
                        ++rejected_aligned;


                    std::cout
                        << "triplet ("
                        << i
                        << ","
                        << j
                        << ","
                        << k
                        << ")"

                        << " | sources=("
                        << a.source_area_index
                        << ","
                        << b.source_area_index
                        << ","
                        << c.source_area_index
                        << ")"

                        << " | keys=("
                        << static_cast<int>(a.key)
                        << ","
                        << static_cast<int>(b.key)
                        << ","
                        << static_cast<int>(c.key)
                        << ")"

                        << " | d=("
                        << dij
                        << ", "
                        << dik
                        << ", "
                        << djk
                        << ")"

                        << " | cross2="
                        << cross2

                        << " | production="
                        << (
                            accepted
                            ? "ACCEPT"
                            : "REJECT"
                        );


                    if (!accepted) {

                        std::cout
                            << " | reason:";


                        if (same_source) {

                            std::cout
                                << " SAME_SOURCE";
                        }


                        if (
                            dij < 0.001 ||
                            dik < 0.001 ||
                            djk < 0.001
                        ) {

                            std::cout
                                << " TOO_CLOSE";
                        }


                        if (
                            dij > ligand_diameter ||
                            dik > ligand_diameter ||
                            djk > ligand_diameter
                        ) {

                            std::cout
                                << " TOO_FAR";
                        }


                        if (aligned) {

                            std::cout
                                << " ALIGNED";
                        }
                    }


                    std::cout
                        << '\n';
                }
            }
        }


        // ====================================================
        // Summary
        // ====================================================

        std::cout
            << "\n=== SUMMARY ===\n"

            << "Possible combinations: "
            << total_combinations
            << '\n'

            << "Production accepted: "
            << production_accepted
            << '\n'

            << "Production rejected: "
            << (
                total_combinations -
                production_accepted
            )
            << '\n'

            << "Combinations with SAME_SOURCE: "
            << rejected_same_source
            << '\n'

            << "Combinations with distance failure: "
            << rejected_distance
            << '\n'

            << "Combinations aligned (cross2 < 1e-6): "
            << rejected_aligned
            << '\n';


        return 0;

    }
    catch (const std::exception& error) {

        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}