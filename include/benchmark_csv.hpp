#ifndef BENCHMARK_CSV_HPP
#define BENCHMARK_CSV_HPP

#include "pipeline.hpp"

#include <filesystem>

// Append one run, creating parent directories and a header for a new/empty file.
// Existing schemas are not checked. Top-k columns report minimum RMSD for
// evaluation, without changing the rankings; timing columns include subintervals.
void append_benchmark_csv(
    const RunResult& result,
    const std::filesystem::path& csv_path
);

#endif