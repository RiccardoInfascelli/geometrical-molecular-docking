#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace
{

using Clock = std::chrono::steady_clock;


struct PdbAtom
{
    std::string line;
    std::string residue_key;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};


struct Comparison
{
    std::size_t matched = 0;
    std::size_t missing = 0;
    std::size_t extra = 0;

    double coverage = 0.0;

    std::vector<std::size_t> extra_indices;
};


double elapsed(
    const Clock::time_point& start,
    const Clock::time_point& end)
{
    return std::chrono::duration<double>(
        end - start
    ).count();
}


bool is_atom_line(
    const std::string& line)
{
    return
        line.rfind("ATOM  ", 0) == 0 ||
        line.rfind("HETATM", 0) == 0;
}


std::string residue_key(
    const std::string& line)
{
    if (line.size() < 27) {
        throw std::runtime_error(
            "Malformed PDB atom line"
        );
    }

    return
        line.substr(0, 6)  + "|" +
        line.substr(17, 3) + "|" +
        line.substr(21, 1) + "|" +
        line.substr(22, 4) + "|" +
        line.substr(26, 1);
}


PdbAtom parse_pdb_atom(
    const std::string& line)
{
    if (line.size() < 54) {
        throw std::runtime_error(
            "Malformed PDB coordinate line"
        );
    }

    PdbAtom atom;

    atom.line =
        line;

    atom.residue_key =
        residue_key(line);

    atom.x =
        std::stod(
            line.substr(30, 8)
        );

    atom.y =
        std::stod(
            line.substr(38, 8)
        );

    atom.z =
        std::stod(
            line.substr(46, 8)
        );

    return atom;
}


std::vector<PdbAtom> read_pdb_atoms(
    const std::filesystem::path& path)
{
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open PDB: " +
            path.string()
        );
    }

    std::vector<PdbAtom> atoms;

    std::string line;

    while (std::getline(file, line)) {

        if (!is_atom_line(line))
            continue;

        atoms.push_back(
            parse_pdb_atom(line)
        );
    }

    return atoms;
}


void write_local_protein(
    const std::vector<PdbAtom>& atoms,
    const std::unordered_set<std::string>& selected_residues,
    const std::filesystem::path& path)
{
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot write temporary PDB: " +
            path.string()
        );
    }

    for (const auto& atom : atoms) {

        if (
            selected_residues.contains(
                atom.residue_key
            )
        ) {
            file
                << atom.line
                << '\n';
        }
    }

    file << "END\n";
}


double centroid_distance(
    const MatchingArea& a,
    const MatchingArea& b)
{
    const double dx =
        static_cast<double>(a.centroid.x) -
        static_cast<double>(b.centroid.x);

    const double dy =
        static_cast<double>(a.centroid.y) -
        static_cast<double>(b.centroid.y);

    const double dz =
        static_cast<double>(a.centroid.z) -
        static_cast<double>(b.centroid.z);

    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


Comparison compare_matching_areas(
    const std::vector<MatchingArea>& reference,
    const std::vector<MatchingArea>& candidate)
{
    constexpr double tolerance =
        1e-3;

    std::vector<bool> used(
        candidate.size(),
        false
    );

    std::size_t matched = 0;


    for (const auto& reference_area :
         reference) {

        std::size_t best_index =
            candidate.size();

        double best_distance =
            std::numeric_limits<double>::
                infinity();


        for (
            std::size_t i = 0;
            i < candidate.size();
            ++i
        ) {

            if (used[i])
                continue;

            if (
                candidate[i].key !=
                reference_area.key
            ) {
                continue;
            }


            const double distance =
                centroid_distance(
                    reference_area,
                    candidate[i]
                );


            if (
                distance <= tolerance &&
                distance < best_distance
            ) {
                best_distance =
                    distance;

                best_index =
                    i;
            }
        }


        if (
            best_index ==
            candidate.size()
        ) {
            continue;
        }


        used[best_index] =
            true;

        ++matched;
    }


    Comparison result;

    result.matched =
        matched;

    result.missing =
        reference.size() - matched;

    result.extra =
        candidate.size() - matched;

    result.coverage =
        reference.empty()
            ? 100.0
            : (
                100.0 *
                static_cast<double>(matched) /
                static_cast<double>(
                    reference.size()
                )
            );

    for (
        std::size_t i = 0;
        i < candidate.size();
        ++i
    ) {
        if (!used[i]) {
            result.extra_indices.push_back(i);
        }
    }

    return result;
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


        const auto protein_atoms =
            read_pdb_atoms(
                config.protein_path
            );

        const auto pocket_atoms =
            read_pdb_atoms(
                config.pocket_path
            );


        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );


        // ====================================================
        // Reference:
        //
        // EXACT current production logic:
        //
        // full protein
        // -> find_protein_areas
        // -> logical matching areas
        // -> near-pocket filter
        // ====================================================

        const auto reference_start =
            Clock::now();

        const auto full_areas =
            get_protein_areas(
                *protein
            );

        const auto reference_end =
            Clock::now();


        const auto full_matching =
            build_protein_matching_areas(
                full_areas
            );


        const auto reference_matching =
            filter_matching_areas_near_pocket(
                full_matching,
                *pocket,
                config.pocket_centroid_distance
            );


        // ====================================================
        // Compute minimum distance of every FULL-PROTEIN
        // residue from any pocket atom.
        //
        // If one atom of a residue is within the buffer,
        // the WHOLE residue will be retained.
        // ====================================================

        std::unordered_map<
            std::string,
            double
        > residue_min_distance2;


        for (const auto& atom :
             protein_atoms) {

            double best_distance2 =
                std::numeric_limits<double>::
                    infinity();


            for (const auto& pocket_atom :
                 pocket_atoms) {

                const double dx =
                    atom.x -
                    pocket_atom.x;

                const double dy =
                    atom.y -
                    pocket_atom.y;

                const double dz =
                    atom.z -
                    pocket_atom.z;

                const double distance2 =
                    dx * dx +
                    dy * dy +
                    dz * dz;


                best_distance2 =
                    std::min(
                        best_distance2,
                        distance2
                    );
            }


            auto [it, inserted] =
                residue_min_distance2.emplace(
                    atom.residue_key,
                    best_distance2
                );


            if (!inserted) {

                it->second =
                    std::min(
                        it->second,
                        best_distance2
                    );
            }
        }


        constexpr double buffers[] = {
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0
        };


        std::cout
            << std::fixed
            << std::setprecision(6);


        std::cout
            << "=== LOCAL PROTEIN BUFFER EXPERIMENT ===\n\n"

            << "Full protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket atoms: "
            << pocket->getNumAtoms()
            << '\n'

            << "Full physical areas: "
            << full_areas.size()
            << '\n'

            << "Reference near-pocket matching areas: "
            << reference_matching.size()
            << '\n'

            << "Full find_protein_areas time: "
            << elapsed(
                reference_start,
                reference_end
            )
            << " s\n\n";


        std::cout
            << "buffer"
            << "\tresidues"
            << "\tatoms"
            << "\tphysical"
            << "\tnear_match"
            << "\tmatched"
            << "\tmissing"
            << "\textra"
            << "\tcoverage"
            << "\tload_s"
            << "\tareas_s"
            << '\n';


        for (const double buffer :
             buffers) {

            const double buffer2 =
                buffer * buffer;


            std::unordered_set<std::string>
                selected_residues;


            for (const auto& [key, min_distance2] :
                 residue_min_distance2) {

                // Tiny tolerance also makes buffer=0
                // robust to coordinate roundoff.
                if (
                    min_distance2 <=
                    buffer2 + 1e-8
                ) {
                    selected_residues.insert(
                        key
                    );
                }
            }


            std::size_t selected_atom_count =
                0;


            for (const auto& atom :
                 protein_atoms) {

                if (
                    selected_residues.contains(
                        atom.residue_key
                    )
                ) {
                    ++selected_atom_count;
                }
            }


            const auto temporary_path =
                std::filesystem::path(
                    "/tmp"
                ) /
                (
                    "gd_local_" +
                    config.complex_id +
                    "_" +
                    std::to_string(
                        static_cast<int>(buffer)
                    ) +
                    ".pdb"
                );


            write_local_protein(
                protein_atoms,
                selected_residues,
                temporary_path
            );


            try {

                const auto load_start =
                    Clock::now();

                auto local_protein =
                    load_protein(
                        temporary_path.string()
                    );

                const auto load_end =
                    Clock::now();


                const auto areas_start =
                    Clock::now();

                const auto local_areas =
                    get_protein_areas(
                        *local_protein
                    );

                const auto areas_end =
                    Clock::now();


                const auto local_matching =
                    build_protein_matching_areas(
                        local_areas
                    );


                const auto local_near_matching =
                    filter_matching_areas_near_pocket(
                        local_matching,
                        *pocket,
                        config.pocket_centroid_distance
                    );


                const auto comparison =
                    compare_matching_areas(
                        reference_matching,
                        local_near_matching
                    );


                std::cout
                    << buffer
                    << '\t'

                    << selected_residues.size()
                    << '\t'

                    << selected_atom_count
                    << '\t'

                    << local_areas.size()
                    << '\t'

                    << local_near_matching.size()
                    << '\t'

                    << comparison.matched
                    << '\t'

                    << comparison.missing
                    << '\t'

                    << comparison.extra
                    << '\t'

                    << comparison.coverage
                    << '\t'

                    << elapsed(
                        load_start,
                        load_end
                    )
                    << '\t'

                    << elapsed(
                        areas_start,
                        areas_end
                    )
                    << '\n';

                if (
    std::abs(buffer - 2.0) < 1e-9 &&
    !comparison.extra_indices.empty()
) {
    std::cout
        << "\n=== EXTRA MATCHING AREAS AT 2 A ===\n";

    for (
        const std::size_t index :
        comparison.extra_indices
    ) {
        const auto& area =
    local_near_matching[index];

const auto& physical =
    local_areas[
        area.source_area_index
    ];


const MatchingArea* nearest_reference =
    nullptr;

double nearest_distance =
    std::numeric_limits<double>::infinity();


for (const auto& reference_area :
     reference_matching) {

        const double distance =
            centroid_distance(
                area,
                reference_area
            );

        if (distance < nearest_distance) {

            nearest_distance =
                distance;

            nearest_reference =
                &reference_area;
        }
    }


    std::cout
        << "LOCAL EXTRA"
        << " key="
        << static_cast<int>(area.key)

        << " source="
        << area.source_area_index

        << " centroid=("
        << area.centroid.x << ", "
        << area.centroid.y << ", "
        << area.centroid.z << ")"

        << " pattern="
        << static_cast<int>(
            physical.pattern
        )

        << " class="
        << static_cast<int>(
            physical.pattern_class
        );


    if (nearest_reference != nullptr) {

        std::cout
            << " | nearest FULL:"
            << " distance="
            << nearest_distance

            << " key="
            << static_cast<int>(
                nearest_reference->key
            )

            << " source="
            << nearest_reference
                ->source_area_index;
    }


    std::cout << '\n';
    }

    std::cout << '\n';
}
            }
            catch (
                const std::exception& error
            ) {

                std::cout
                    << buffer
                    << "\tERROR: "
                    << error.what()
                    << '\n';
            }


            std::error_code ec;

            std::filesystem::remove(
                temporary_path,
                ec
            );
        }


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
