#include "pose_output.hpp"

#include "geometry.hpp"
#include "molecule_io.hpp"

#include <GraphMol/FileParsers/FileParsers.h>
#include <Geometry/point.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>


namespace
{

struct PoseCenters {
    std::array<double, 3> ligand, reference;
};

std::unique_ptr<RDKit::RWMol> load_center_reference(const std::filesystem::path& path) {
    Mol2Reader reader(path);
    std::string block, name;
    if (!reader.next(block, name))
        throw std::invalid_argument("Pose center reference contains no MOL2 record");
    auto reference = parse_ligand_mol2_block(block);
    if (reader.next(block, name))
        throw std::invalid_argument("Pose center reference must contain exactly one MOL2 record");
    return reference;
}

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
    std::size_t count,
    std::optional<PoseExportDetail> detail,
    const PoseCenters* centers)
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


    const bool compact = detail == PoseExportDetail::Compact;
    const bool full = detail == PoseExportDetail::Full;
    if (compact) {
        file << "rank,rmsd,weighted_interaction_score,fast_weighted_score,"
                "occupied_atoms,clash_fraction";
    } else {
        file << "rank,rmsd,weighted_interaction_score,interaction_count,"
                "satisfied_ligand_areas,fast_weighted_score,grid_compatible_areas,"
                "occupied_atoms,clash_fraction";
    }
    if (full) {
        file << ",r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz";
    }
    if (centers) file << ",center_distance_A";
    file << '\n' << std::setprecision(detail ? 17 : 10);


    for (
        std::size_t i = 0;
        i < count;
        ++i
    ) {
        const auto& pose =
            poses[i];


        file << i + 1 << ',' << pose.rmsd << ','
             << pose.weighted_interaction_score.weighted_score;
        if (!compact) {
            file << ',' << pose.weighted_interaction_score.total_interactions
                 << ',' << pose.weighted_interaction_score.satisfied_ligand_areas;
        }
        file << ',' << pose.fast_weighted_score;
        if (!compact) file << ',' << pose.compatible_areas;
        file << ',' << pose.occupied_atoms << ',' << pose.clash_fraction;
        if (full) {
            for (const double value : pose.transform.rotation) file << ',' << value;
            for (const double value : pose.transform.translation) file << ',' << value;
        }
        if (centers)
            file << ',' << pose_center_distance(centers->ligand, pose.transform, centers->reference);
        file << '\n';
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
    std::unique_ptr<RDKit::RWMol> ligand;
    std::optional<PoseCenters> centers;
    if (config.pose_center_distance) {
        if (config.center_reference_mol2.empty())
            throw std::invalid_argument("Pose center distance requires an explicit reference MOL2 file");
        ligand = load_ligand(config.ligand_path.string());
        const auto reference = load_center_reference(config.center_reference_mol2);
        centers = PoseCenters{heavy_atom_centroid(*ligand), heavy_atom_centroid(*reference)};
    }
    if (config.pose_export.mode == PoseExportMode::None) return;

    const auto& ranking = result.pose_ranking.weighted_interaction_ranking;
    auto export_options = config.pose_export;
    // Keep the existing top_n argument meaningful for experimental callers.
    export_options.top_k = top_n;
    const std::size_t pose_count = pose_export_count(export_options, ranking.size());

    const std::filesystem::path
        output_directory =
            config.output_directory /
            result.complex_id;


    std::filesystem::create_directories(
        output_directory
    );

    if (!config.export_structures) {
        write_ranking_csv(output_directory / "ranking.csv", ranking, pose_count,
                          config.pose_export.detail, centers ? &*centers : nullptr);
        return;
    }


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

    if (!ligand) ligand = load_ligand(config.ligand_path.string());


    RDKit::MolToPDBFile(
        *ligand,
        (
            output_directory /
            "crystal_ligand.pdb"
        ).string()
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
        pose_count,
        config.pose_export.detail,
        centers ? &*centers : nullptr
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
