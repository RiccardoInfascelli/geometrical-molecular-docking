// Benchmark standalone dell'indice CPU delle triplette proteiche.
// Esegue la stessa preparazione dei pose target usata da run_pipeline(),
// poi costruisce l'indice per UN cutoff; lanciare un processo per cutoff.
// Non esegue docking, scoring o alcun kernel CUDA.

#include "run_config.hpp"
#include "molecule_io.hpp"
#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "protein_grid.hpp"
#include "pose_targets.hpp"
#include "triplet_generation.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

// Linux-only: resident set size (KiB) of the current process.
// The difference before/after is an approximate host-memory increment,
// not the exact retained heap size, and NOT a GPU VRAM measurement.
double proc_status_mib(const std::string& field)
{
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream input(line);
        std::string name;
        double kib = 0.0;
        input >> name;
        if (name == field) {
            input >> kib;
            return kib / 1024.0;
        }
    }
    return -1.0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <complex_directory> <cutoff_angstrom>\n";
        return 2;
    }

    try {
        const double cutoff = std::stod(argv[2]);
        if (!std::isfinite(cutoff) || cutoff <= 0.0)
            throw std::runtime_error("Cutoff must be positive and finite");

        // Parse exactly the same default grid/bin parameters as the CPU docking.
        // argv[2] is our extra cutoff argument, so pass only the complex path.
        char* docking_argv[] = {argv[0], argv[1]};
        const RunConfig config = parse_run_config(2, docking_argv);

        // Same target-preparation sequence as src/pipeline.cpp.
        const auto protein = load_protein(config.protein_path.string());
        const auto pocket = load_pocket(config.pocket_path.string());
        const auto protein_areas = get_protein_areas(*protein);
        const auto protein_matching_areas =
            build_protein_matching_areas(protein_areas);

        auto protein_grid = build_empty_protein_grid(
            *pocket, config.grid_voxel_size, config.grid_margin);
        mark_protein_occupancy(*protein, protein_grid);
        mark_interaction_regions(protein_matching_areas, protein_grid);

        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas, *pocket,
                config.pocket_centroid_distance);
        const auto pose_targets = generate_pose_targets(
            pocket_matching_areas, protein_grid, *protein, *pocket);

        // Process per cutoff, so allocator leftovers from other indexes
        // do not contaminate the memory comparison.
        const double rss_before = proc_status_mib("VmRSS:");
        const auto start = std::chrono::steady_clock::now();
        const auto index = build_protein_triplet_hash_table(
            pose_targets, cutoff, config.triplet_bin_size);
        const auto stop = std::chrono::steady_clock::now();
        const double rss_after = proc_status_mib("VmRSS:");
        const double rss_peak = proc_status_mib("VmHWM:");

        std::size_t record_count = 0;
        std::size_t record_capacity = 0;
        for (const auto& entry : index) {
            record_count += entry.second.size();
            record_capacity += entry.second.capacity();
        }

        constexpr double mib = 1024.0 * 1024.0;
        const double compact_record_mib =
            static_cast<double>(record_count) *
            sizeof(ProteinTripletRecord) / mib;
        const double reserved_record_mib =
            static_cast<double>(record_capacity) *
            sizeof(ProteinTripletRecord) / mib;
        const double seconds =
            std::chrono::duration<double>(stop - start).count();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "cutoff_A,pose_targets,build_seconds,records,hash_keys,"
                  << "record_payload_MiB,record_vector_capacity_MiB,"
                  << "host_RSS_delta_MiB,host_RSS_peak_MiB\n";
        std::cout << cutoff << ','
                  << pose_targets.size() << ','
                  << seconds << ','
                  << record_count << ','
                  << index.size() << ','
                  << compact_record_mib << ','
                  << reserved_record_mib << ','
                  << (rss_after >= 0.0 && rss_before >= 0.0
                          ? rss_after - rss_before : -1.0) << ','
                  << rss_peak << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
