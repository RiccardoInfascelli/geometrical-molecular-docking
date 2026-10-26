#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "molecule_io.hpp"

#include <interaction/sanitize_mol.hpp>
#include <GraphMol/FileParsers/FileParsers.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct Results {
    std::vector<double> diameters;
    std::size_t processed = 0;
    std::size_t failures = 0;
    std::size_t insufficient_areas = 0;
};

double percentile(const std::vector<double>& sorted, double p) {
    const double position = p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

void emit_molecule(
    const std::string& source,
    const std::string& id,
    std::size_t index,
    std::unique_ptr<RDKit::RWMol> molecule,
    bool already_sanitized,
    std::ofstream& csv,
    Results& results
) {
    ++results.processed;
    std::size_t atoms = 0;
    std::size_t areas_count = 0;
    double diameter = 0.0;
    std::string status = "ok";

    try {
        if (!molecule) throw std::runtime_error("MOL2 parse returned null");
        if (!already_sanitized) interaction::sanitize_molecule(molecule);
        atoms = molecule->getNumAtoms();
        const auto areas = get_ligand_areas(*molecule);
        areas_count = areas.size();
        if (areas_count < 2) {
            status = "too_few_areas";
            ++results.insufficient_areas;
        } else {
            diameter = compute_ligand_diameter(areas);
            if (!std::isfinite(diameter))
                throw std::runtime_error("Nonfinite diameter");
            results.diameters.push_back(diameter);
        }
    } catch (const std::exception&) {
        status = "error";
        ++results.failures;
    }

    csv << csv_quote(source) << ',' << csv_quote(id) << ',' << index << ','
        << atoms << ',' << areas_count << ',';
    if (status == "ok") csv << std::setprecision(12) << diameter;
    csv << ',' << status << '\n';
}

std::string molecule_name(const std::string& block, std::size_t index) {
    std::istringstream input(block);
    std::string marker, name;
    std::getline(input, marker);
    std::getline(input, name);
    if (!name.empty() && name.back() == '\r') name.pop_back();
    return name.empty() ? std::to_string(index) : name;
}

void process_refined(const fs::path& root, std::ofstream& csv, Results& results) {
    if (!fs::is_directory(root)) throw std::runtime_error("Not a directory: " + root.string());
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() &&
            ends_with(entry.path().filename().string(), "_ligand.mol2"))
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    for (std::size_t index = 0; index < paths.size(); ++index) {
        const auto& path = paths[index];
        try {
            emit_molecule(path.string(), path.stem().string(), index + 1,
                          load_ligand(path.string()), true, csv, results);
        } catch (const std::exception&) {
            emit_molecule(path.string(), path.stem().string(), index + 1,
                          nullptr, true, csv, results);
        }
    }
}

void process_multi(const fs::path& path, std::size_t stride,
                   std::ofstream& csv, Results& results) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open MOL2: " + path.string());
    std::string line, block;
    std::size_t record_index = 0;
    bool in_record = false;
    bool selected = false;
    const auto flush = [&]() {
        if (!in_record || !selected) return;
        std::unique_ptr<RDKit::RWMol> molecule;
        try {
            molecule.reset(RDKit::Mol2BlockToMol(block, false, false));
        } catch (const std::exception&) {
            molecule.reset();
        }
        emit_molecule(path.string(), molecule_name(block, record_index),
                      record_index, std::move(molecule), false, csv, results);
    };
    while (std::getline(input, line)) {
        if (line.rfind("@<TRIPOS>MOLECULE", 0) == 0) {
            flush();
            ++record_index;
            in_record = true;
            selected = (record_index - 1) % stride == 0;
            block.clear();
        }
        if (selected) {
            block += line;
            block += '\n';
        }
    }
    flush();
    std::cout << "Total MOL2 sections scanned: " << record_index << '\n';
}

void print_summary(Results& results) {
    std::sort(results.diameters.begin(), results.diameters.end());
    const auto& d = results.diameters;
    std::cout << "Processed: " << results.processed << '\n'
              << "Valid diameters (>=2 areas): " << d.size() << '\n'
              << "Too few areas: " << results.insufficient_areas << '\n'
              << "Parse/sanitize/geometry errors: " << results.failures << '\n';
    if (d.empty()) return;
    const double sum = std::accumulate(d.begin(), d.end(), 0.0);
    const double mean = sum / static_cast<double>(d.size());
    double variance = 0.0;
    for (double value : d) variance += (value - mean) * (value - mean);
    variance /= static_cast<double>(d.size());
    std::cout << std::fixed << std::setprecision(3)
              << "Diameter (angstrom) min: " << d.front() << '\n'
              << "Mean: " << mean << '\n'
              << "Std dev: " << std::sqrt(variance) << '\n'
              << "P50: " << percentile(d, 0.50) << '\n'
              << "P75: " << percentile(d, 0.75) << '\n'
              << "P90: " << percentile(d, 0.90) << '\n'
              << "P95: " << percentile(d, 0.95) << '\n'
              << "P99: " << percentile(d, 0.99) << '\n'
              << "Max: " << d.back() << '\n';
    std::map<long long, std::size_t> histogram;
    constexpr double width = 2.0;
    for (double value : d)
        ++histogram[static_cast<long long>(std::floor(value / width))];
    std::cout << "Histogram, 2-angstrom bins:\n";
    for (const auto& bin : histogram)
        std::cout << '[' << bin.first * width << ','
                  << (bin.first + 1) * width << "): "
                  << bin.second << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage:\n"
                  << "  ligand_diameter_stats refined <directory> <output.csv>\n"
                  << "  ligand_diameter_stats multi <multi.mol2> <output.csv> [stride=1000]\n";
        return 2;
    }
    try {
        std::ofstream csv(argv[3]);
        if (!csv) throw std::runtime_error("Cannot create output CSV");
        csv << "source,molecule_id,record_index,atoms,interaction_areas,diameter_angstrom,status\n";
        Results results;
        const std::string mode = argv[1];
        if (mode == "refined" && argc == 4) {
            process_refined(argv[2], csv, results);
        } else if (mode == "multi") {
            const std::size_t stride = argc == 5 ? std::stoull(argv[4]) : 1000;
            if (stride == 0) throw std::runtime_error("Stride must be positive");
            process_multi(argv[2], stride, csv, results);
        } else {
            throw std::runtime_error("Unknown mode or invalid arguments");
        }
        csv.close();
        print_summary(results);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
