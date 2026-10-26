// Diagnostic CPU entry point: keep computation separate from benchmark append,
// pose export and reporting. An export failure may follow a completed search.
#include "pipeline.hpp"
#include "pose_output.hpp"
#include "report.hpp"
#include "run_config.hpp"
#include "benchmark_csv.hpp"

#include <exception>
#include <iostream>


int main(
    int argc,
    char* argv[])
{
    try {

        const RunConfig config =
            parse_run_config(
                argc,
                argv
            );


        const RunResult result =
            run_pipeline(
                config
            );

        append_benchmark_csv(
            result,
            config.output_directory /
                "benchmark.csv"
        );


        save_pose_outputs(
            result,
            config,
            20
        );


        print_report(
            result,
            config
        );


        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "Error: "
            << e.what()
            << '\n';

        return 1;
    }
}
