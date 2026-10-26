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
        << "  --quiet                Minimal output\n"
        << "  --debug                Detailed diagnostic output\n"
        << "  --help                 Show this help\n\n"

        << "Default complex:\n"
        << "  /workspace/datasets/refined-set/1a1e\n";
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


    return config;
}