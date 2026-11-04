// Diagnostic CPU entry point: keep computation separate from benchmark append,
// pose export and reporting. An export failure may follow a completed search.
#include "pipeline.hpp"
#include "pose_output.hpp"
#include "report.hpp"
#include "run_config.hpp"
#include "benchmark_csv.hpp"

#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
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

        apply_cpu_docking_environment();

        using Clock = std::chrono::steady_clock;
        const auto start = config.export_timings ? Clock::now() : Clock::time_point{};

        const RunResult result =
            run_pipeline(
                config
            );

        const auto computation_end = config.export_timings ? Clock::now() : Clock::time_point{};

        append_benchmark_csv(
            result,
            config.output_directory /
                "benchmark.csv"
        );


        save_pose_outputs(
            result,
            config,
            config.pose_export.top_k
        );

        if (config.export_timings) {
            const auto output_end = Clock::now();
            const auto seconds = [](Clock::duration duration) {
                return std::chrono::duration<double>(duration).count();
            };
            std::ofstream timings(config.output_directory / "timings.csv");
            timings.exceptions(std::ios::badbit | std::ios::failbit);
            // Wall starts after CLI parsing and ends before this file and the report.
            timings << "compute_s,export_s,wall_before_report_s\n" << std::setprecision(17)
                    << seconds(computation_end - start) << ','
                    << seconds(output_end - computation_end) << ','
                    << seconds(output_end - start) << '\n';
        }


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
