#include "pose_output.hpp"

#include "geometry.hpp"
#include "molecule_io.hpp"

#include <GraphMol/FileParsers/FileParsers.h>
#include <Geometry/point.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>


namespace
{

std::string rank_string(
    std::size_t rank)
{
    std::ostringstream stream;

    stream
        << std::setw(3)
        << std::setfill('0')
        << rank;

    return stream.str();
}


RDKit::RWMol transformed_ligand(
    const RDKit::RWMol& ligand,
    const RigidTransform& transform)
{
    RDKit::RWMol pose(
        ligand
    );


    auto& conformer =
        pose.getConformer();


    for (
        unsigned int atom_index = 0;
        atom_index < pose.getNumAtoms();
        ++atom_index
    ) {
        const auto position =
            conformer.getAtomPos(
                atom_index
            );


        const interaction::point3D original{
            static_cast<interaction::coordinate_type>(
                position.x
            ),
            static_cast<interaction::coordinate_type>(
                position.y
            ),
            static_cast<interaction::coordinate_type>(
                position.z
            )
        };


        const auto transformed =
            transform_point(
                original,
                transform
            );


        conformer.setAtomPos(
            atom_index,
            RDGeom::Point3D(
                static_cast<double>(
                    transformed.x
                ),
                static_cast<double>(
                    transformed.y
                ),
                static_cast<double>(
                    transformed.z
                )
            )
        );
    }


    return pose;
}


void write_ranking_csv(
    const std::filesystem::path& path,
    const std::vector<RankedPoseDiagnostic>& poses,
    std::size_t count)
{
    std::ofstream file(
        path
    );


    if (!file) {
        throw std::runtime_error(
            "Could not create ranking CSV: " +
            path.string()
        );
    }


    file
        << "rank,"
        << "rmsd,"
        << "weighted_interaction_score,"
        << "interaction_count,"
        << "satisfied_ligand_areas,"
        << "fast_weighted_score,"
        << "grid_compatible_areas,"
        << "occupied_atoms,"
        << "clash_fraction"
        << '\n';


    file
        << std::setprecision(10);


    for (
        std::size_t i = 0;
        i < count;
        ++i
    ) {
        const auto& pose =
            poses[i];


        file
            << (i + 1)
            << ','

            << pose.rmsd
            << ','

            << pose.weighted_interaction_score.weighted_score
            << ','

            << pose.weighted_interaction_score.total_interactions
            << ','

            << pose.weighted_interaction_score.satisfied_ligand_areas
            << ','

            << pose.fast_weighted_score
            << ','

            << pose.compatible_areas
            << ','

            << pose.occupied_atoms
            << ','

            << pose.clash_fraction
            << '\n';
    }
}


void write_pymol_script(
    const std::filesystem::path& path,
    std::size_t pose_count)
{
    std::ofstream file(
        path
    );


    if (!file) {
        throw std::runtime_error(
            "Could not create PyMOL script: " +
            path.string()
        );
    }


    file
        << "reinitialize\n"
        << "load protein.pdb, protein\n"
        << "load pocket.pdb, pocket\n"
        << "load crystal_ligand.pdb, crystal\n";


    for (
        std::size_t i = 0;
        i < pose_count;
        ++i
    ) {
        const std::string rank =
            rank_string(
                i + 1
            );

        file
            << "load pose_"
            << rank
            << ".pdb, pose_"
            << rank
            << "\n";
    }


    file
        << "\nhide everything\n"
        << "show cartoon, protein\n"
        << "show sticks, crystal\n"
        << "show sticks, pose_001\n"
        << "color green, crystal\n"
        << "color cyan, pose_001\n"
        << "zoom crystal, 12\n";


    for (
        std::size_t i = 1;
        i < pose_count;
        ++i
    ) {
        file
            << "disable pose_"
            << rank_string(
                   i + 1
               )
            << "\n";
    }


    file
        << "\n# Enable another pose with, for example:\n"
        << "# enable pose_002\n"
        << "# disable pose_001\n";
}

} // namespace


void save_pose_outputs(
    const RunResult& result,
    const RunConfig& config,
    std::size_t top_n)
{
    const std::filesystem::path
        output_directory =
            config.output_directory /
            result.complex_id;


    std::filesystem::create_directories(
        output_directory
    );


    // ========================================================
    // Copy reference structures
    // ========================================================

    std::filesystem::copy_file(
        config.protein_path,
        output_directory /
            "protein.pdb",
        std::filesystem::copy_options::
            overwrite_existing
    );


    std::filesystem::copy_file(
        config.pocket_path,
        output_directory /
            "pocket.pdb",
        std::filesystem::copy_options::
            overwrite_existing
    );


    std::filesystem::copy_file(
        config.ligand_path,
        output_directory /
            "crystal_ligand.mol2",
        std::filesystem::copy_options::
            overwrite_existing
    );


    // ========================================================
    // Load crystal ligand once
    // ========================================================

    const auto ligand =
        load_ligand(
            config.ligand_path.string()
        );


    RDKit::MolToPDBFile(
        *ligand,
        (
            output_directory /
            "crystal_ligand.pdb"
        ).string()
    );


    // ========================================================
    // Final ranking only
    // ========================================================

    const auto& ranking =
        result.pose_ranking
            .weighted_interaction_ranking;


    const std::size_t pose_count =
        std::min(
            top_n,
            ranking.size()
        );


    // ========================================================
    // Write predicted poses
    // ========================================================

    for (
        std::size_t i = 0;
        i < pose_count;
        ++i
    ) {
        const auto pose =
            transformed_ligand(
                *ligand,
                ranking[i].transform
            );


        const std::filesystem::path
            pose_path =
                output_directory /
                (
                    "pose_" +
                    rank_string(i + 1) +
                    ".pdb"
                );


        RDKit::MolToPDBFile(
            pose,
            pose_path.string()
        );
    }


    // ========================================================
    // Ranking metadata
    // ========================================================

    write_ranking_csv(
        output_directory /
            "ranking.csv",
        ranking,
        pose_count
    );


    // ========================================================
    // PyMOL helper
    // ========================================================

    write_pymol_script(
        output_directory /
            "view.pml",
        pose_count
    );
}