#include "pipeline.hpp"
#include "pose_output.hpp"
#include "report.hpp"
#include "run_config.hpp"

#include <exception>
#include <iostream>

int main(
    int argc,
    char* argv[])
{
    try {

        auto config =
            parse_run_config(
                argc,
                argv
            );

        // Experiment only: accepts fractions through 1.0; the baseline stays 0.05.
        // The legacy "6%" banner below does not describe this configuration.
        config.operational_clash_threshold =
            1.0;

        config.clash_thresholds = {
            0.0,
            0.10,
            0.20,
            1.0
        };
        std::cout
            << "=== CLASH THRESHOLD EXPERIMENT ===\n"
            << "Operational threshold: 6%\n\n";

        const auto result =
            run_pipeline(
                config
            );

        print_report(
            result,
            config
        );

        save_pose_outputs(
            result,
            config,
            20
        );

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