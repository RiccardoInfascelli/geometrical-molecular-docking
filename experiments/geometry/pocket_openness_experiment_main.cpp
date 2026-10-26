#include "molecule_io.hpp"
#include "run_config.hpp"

#include <interaction/elements.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


namespace
{

struct Vec3
{
    double x{};
    double y{};
    double z{};
};


struct ProteinAtom
{
    Vec3 relative_position{};
    double distance_squared{};
    double vdw_radius{};
};


double atom_vdw_radius(
    const RDKit::Atom& atom)
{
    const auto element =
        interaction::parse_chemical_element(
            atom.getSymbol()
        );

    if (element.has_value()) {
        return interaction::get(
            element.value()
        ).vdw_radius;
    }

    return 1.7;
}


Vec3 ligand_heavy_atom_centroid(
    const RDKit::ROMol& ligand)
{
    const auto& conformer =
        ligand.getConformer();

    Vec3 centroid{};

    std::size_t count = 0;


    for (
        unsigned int i = 0;
        i < ligand.getNumAtoms();
        ++i
    ) {
        const auto* atom =
            ligand.getAtomWithIdx(i);

        if (atom->getAtomicNum() == 1)
            continue;


        const auto position =
            conformer.getAtomPos(i);

        centroid.x += position.x;
        centroid.y += position.y;
        centroid.z += position.z;

        ++count;
    }


    if (count == 0) {
        throw std::runtime_error(
            "Ligand has no heavy atoms"
        );
    }


    centroid.x /=
        static_cast<double>(count);

    centroid.y /=
        static_cast<double>(count);

    centroid.z /=
        static_cast<double>(count);


    return centroid;
}


std::vector<Vec3> fibonacci_directions(
    std::size_t count)
{
    std::vector<Vec3> directions;

    directions.reserve(count);


    constexpr double GOLDEN_ANGLE =
        2.39996322972865332;


    for (
        std::size_t i = 0;
        i < count;
        ++i
    ) {
        const double z =
            1.0 -
            2.0 *
            (
                static_cast<double>(i) +
                0.5
            ) /
            static_cast<double>(count);


        const double radius =
            std::sqrt(
                std::max(
                    0.0,
                    1.0 - z * z
                )
            );


        const double phi =
            GOLDEN_ANGLE *
            static_cast<double>(i);


        directions.push_back({
            radius * std::cos(phi),
            radius * std::sin(phi),
            z
        });
    }


    return directions;
}


std::vector<ProteinAtom>
prepare_protein_atoms(
    const RDKit::ROMol& protein,
    const Vec3& origin)
{
    const auto& conformer =
        protein.getConformer();


    std::vector<ProteinAtom> atoms;

    atoms.reserve(
        protein.getNumAtoms()
    );


    for (
        unsigned int i = 0;
        i < protein.getNumAtoms();
        ++i
    ) {
        const auto position =
            conformer.getAtomPos(i);


        const Vec3 relative{
            position.x - origin.x,
            position.y - origin.y,
            position.z - origin.z
        };


        const double distance_squared =
            relative.x * relative.x +
            relative.y * relative.y +
            relative.z * relative.z;


        atoms.push_back({
            relative,
            distance_squared,
            atom_vdw_radius(
                *protein.getAtomWithIdx(i)
            )
        });
    }


    return atoms;
}


/*
 * Minimum clearance between a ray starting at
 * the ligand centroid and the VdW surface of
 * any protein atom.
 *
 * Positive:
 *     ray stays outside protein VdW volume.
 *
 * Negative:
 *     ray intersects protein.
 */
double ray_clearance(
    const Vec3& direction,
    const std::vector<ProteinAtom>& atoms)
{
    double minimum_clearance =
        std::numeric_limits<double>::
            infinity();


    for (const auto& atom : atoms) {

        const auto& p =
            atom.relative_position;


        const double projection =
            p.x * direction.x +
            p.y * direction.y +
            p.z * direction.z;


        /*
         * Distance from atom center to the
         * forward half-line.
         *
         * If projection <= 0, the closest
         * point of the ray is its origin.
         */
        double distance_squared;


        if (projection > 0.0) {

            distance_squared =
                atom.distance_squared -
                projection * projection;

            distance_squared =
                std::max(
                    0.0,
                    distance_squared
                );
        }
        else {

            distance_squared =
                atom.distance_squared;
        }


        const double clearance =
            std::sqrt(
                distance_squared
            ) -
            atom.vdw_radius;


        minimum_clearance =
            std::min(
                minimum_clearance,
                clearance
            );
    }


    return minimum_clearance;
}


std::string provisional_classification(
    double open_fraction_probe_05)
{
    /*
     * Deliberately provisional.
     *
     * We will calibrate these thresholds by
     * visual inspection of the first ~20
     * complexes.
     */
    if (open_fraction_probe_05 >= 0.05)
        return "OPEN_CANDIDATE";

    if (open_fraction_probe_05 <= 0.005)
        return "CLOSED_CANDIDATE";

    return "BORDERLINE";
}

} // namespace


int main(int argc, char* argv[])
{
    try {

        const auto config =
            parse_run_config(
                argc,
                argv
            );


        auto protein =
            load_protein(
                config.protein_path.string()
            );


        auto ligand =
            load_ligand(
                config.ligand_path.string()
            );


        const Vec3 ligand_centroid =
            ligand_heavy_atom_centroid(
                *ligand
            );


        const auto protein_atoms =
            prepare_protein_atoms(
                *protein,
                ligand_centroid
            );


        constexpr std::size_t RAY_COUNT =
            4096;


        const auto directions =
            fibonacci_directions(
                RAY_COUNT
            );


        constexpr std::array<double, 4>
            PROBES{
                0.0,
                0.5,
                1.0,
                1.4
            };


        std::array<std::size_t, 4>
            open_counts{};


        double best_ray_clearance =
            -std::numeric_limits<double>::
                infinity();


        for (const auto& direction :
             directions) {

            const double clearance =
                ray_clearance(
                    direction,
                    protein_atoms
                );


            best_ray_clearance =
                std::max(
                    best_ray_clearance,
                    clearance
                );


            for (
                std::size_t i = 0;
                i < PROBES.size();
                ++i
            ) {
                if (
                    clearance >=
                    PROBES[i]
                ) {
                    ++open_counts[i];
                }
            }
        }


        std::array<double, 4>
            open_fractions{};


        for (
            std::size_t i = 0;
            i < PROBES.size();
            ++i
        ) {
            open_fractions[i] =
                static_cast<double>(
                    open_counts[i]
                ) /
                static_cast<double>(
                    RAY_COUNT
                );
        }


        const std::string classification =
            provisional_classification(
                open_fractions[1]
            );


        std::cout
            << std::fixed
            << std::setprecision(6)

            << config.complex_id
            << '\t'

            << protein->getNumAtoms()
            << '\t'

            << ligand->getNumAtoms()
            << '\t'

            << open_fractions[0]
            << '\t'

            << open_fractions[1]
            << '\t'

            << open_fractions[2]
            << '\t'

            << open_fractions[3]
            << '\t'

            << best_ray_clearance
            << '\t'

            << classification
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
