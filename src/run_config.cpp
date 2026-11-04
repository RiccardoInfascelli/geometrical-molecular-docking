#include "run_config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void print_usage(
    const char* program)
{
    std::cout
        << "Usage:\n"
        << "  "
        << program
        << " [complex_directory] [options]\n\n"

        << "Options:\n"
        << "  --output <directory>   Output directory (default: results)\n"
        << "  --pose-output <mode>   Pose export: none, topk (default), all\n"
        << "  --top-k <K>            Number of poses exported in topk mode (default: 20)\n"
        << "  --pose-detail <level>  Ranking CSV detail: compact or full\n"
        << "  --no-structures        Skip PDB, PyMOL and reference structure copies\n"
        << "  --timings              Write compute/export/wall timings to timings.csv\n"
        << "  --pose-center-distance on|off  Export heavy-atom center distance (default off)\n"
        << "  --center-reference-mol2 FILE  Explicit reference required for center distance\n"
        << "  --quiet                Minimal output\n"
        << "  --debug                Detailed diagnostic output\n"
        << "  --help                 Show this help\n\n"

        << "Default complex:\n"
        << "  /workspace/datasets/refined-set/1a1e\n\n"
        << "Pose export does not change scoring or benchmark.csv. For geometric_docking,\n"
        << "all exports the available WeightedInteraction ranking after the Fast prefilter.\n";
    std::cout << "geometric_docking defaults to Quaternion; set DOCKING_KABSCH_METHOD=svd\n"
                 "to select SVD. Other applications retain their own alignment policy.\n";
}


void finalize_paths(
    RunConfig& config)
{
    config.complex_directory =
        std::filesystem::absolute(
            config.complex_directory
        ).lexically_normal();


    if (
        config.complex_directory
            .filename()
            .empty()
    ) {
        config.complex_directory =
            config.complex_directory
                .parent_path();
    }


    config.complex_id =
        config.complex_directory
            .filename()
            .string();


    if (config.complex_id.empty()) {
        throw std::runtime_error(
            "Could not determine complex ID"
        );
    }


    config.protein_path =
        config.complex_directory /
        (
            config.complex_id +
            "_protein.pdb"
        );


    config.ligand_path =
        config.complex_directory /
        (
            config.complex_id +
            "_ligand.mol2"
        );


    config.pocket_path =
        config.complex_directory /
        (
            config.complex_id +
            "_pocket.pdb"
        );
}

} // namespace


RunConfig parse_run_config(
    int argc,
    char* argv[])
{
    RunConfig config;

    // Without arguments, use the 1a1e complex.
    config.complex_directory =
        "/workspace/datasets/refined-set/1a1e";


    bool complex_directory_set =
        false;


    for (
        int i = 1;
        i < argc;
        ++i
    ) {
        const std::string argument =
            argv[i];


        if (argument == "--help") {

            print_usage(
                argv[0]
            );

            std::exit(0);
        }


        if (argument == "--output") {

            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "--output requires a directory"
                );
            }

            config.output_directory =
                argv[++i];

            continue;
        }


        if (argument == "--quiet") {

            config.output_mode =
                OutputMode::Quiet;

            continue;
        }

        if (argument == "--pose-output" || argument == "--top-k" ||
            argument == "--pose-detail") {
            if (i + 1 >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            const std::string value = argv[++i];
            if (argument == "--pose-output") {
                config.pose_export.mode = parse_pose_export_mode(value);
            } else if (argument == "--top-k") {
                config.pose_export.top_k = parse_pose_top_k(value);
            } else {
                config.pose_export.detail = parse_pose_export_detail(value);
            }
            continue;
        }

        if (argument == "--no-structures") {
            config.export_structures = false;
            continue;
        }

        if (argument == "--timings") {
            config.export_timings = true;
            continue;
        }

        if (argument == "--pose-center-distance") {
            if (i + 1 >= argc) throw std::runtime_error(argument + " requires on or off");
            const std::string value = argv[++i];
            if (value != "on" && value != "off")
                throw std::runtime_error(argument + " expects on or off");
            config.pose_center_distance = value == "on";
            continue;
        }

        if (argument == "--center-reference-mol2") {
            if (i + 1 >= argc || std::string(argv[i + 1]).empty() || argv[i + 1][0] == '-')
                throw std::runtime_error(argument + " requires a MOL2 file");
            config.center_reference_mol2 = argv[++i];
            continue;
        }


        if (argument == "--debug") {

            config.output_mode =
                OutputMode::Debug;

            continue;
        }


        if (
            !argument.empty() &&
            argument[0] == '-'
        ) {
            throw std::runtime_error(
                "Unknown option: " +
                argument
            );
        }


        if (complex_directory_set) {
            throw std::runtime_error(
                "Multiple complex directories provided"
            );
        }


        config.complex_directory =
            argument;

        complex_directory_set =
            true;
    }


    finalize_paths(
        config
    );

    if (config.pose_center_distance && config.center_reference_mol2.empty())
        throw std::runtime_error("--pose-center-distance on requires --center-reference-mol2 FILE");

    return config;
}

void apply_cpu_docking_environment()
{
    if (::setenv("DOCKING_KABSCH_METHOD", "quaternion", 0) != 0)
        throw std::runtime_error("Could not set the CPU docking Kabsch default");
}
