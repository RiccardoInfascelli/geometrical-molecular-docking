#include "quaternion_rmsd.hpp"
#include "conformer_generation.hpp"
#include "conformer_quality_input.hpp"

#include <GraphMol/Conformer.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Cloud = std::vector<std::array<double, 3>>;
using conformer_experiment::quaternion_aligned_rmsd;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12 * (1.0 + std::abs(expected)),
            "Unexpected quaternion aligned RMSD");
}

void rejects(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument for invalid quaternion RMSD input");
}

Cloud moved(const Cloud& source)
{
    Cloud result;
    for (const auto& p : source) result.push_back({-p[1] + 7, p[0] - 3, p[2] + 2});
    return result;
}

void test_quaternion_geometry()
{
    const Cloud source{{0, 0, 0}, {1, 0, 0}, {0, 2, 0}, {0, 0, 3}, {4, -1, 2}, {-2, 3, 1}};
    const auto target = moved(source);
    const auto original_source = source, original_target = target;
    close(quaternion_aligned_rmsd(source, target), 0.0);
    close(quaternion_aligned_rmsd(target, source), 0.0);
    require(source == original_source && target == original_target, "Quaternion fit mutated its input coordinates");
    auto changed = target;
    changed.back()[2] += 1.0;
    require(quaternion_aligned_rmsd(source, changed) > 0.1, "Fit omitted a point beyond the first three");
    changed = target;
    std::swap(changed[0], changed[1]);
    require(quaternion_aligned_rmsd(source, changed) > 0.1, "Fit silently permuted fixed atom correspondences");

    const Cloud tetrahedron{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
    auto mirror = tetrahedron;
    for (auto& point : mirror) point[0] = -point[0];
    // A reflection fits this centered tetrahedron exactly only with det=-1.
    // The optimal proper rotation has squared error 16 over four points.
    close(quaternion_aligned_rmsd(tetrahedron, mirror), 2.0);

    close(quaternion_aligned_rmsd({{1, 2, 3}}, {{100, -2, 0}}), 0.0);
    close(quaternion_aligned_rmsd({{0, 0, 0}, {2, 0, 0}}, {{4, 5, 6}, {4, 9, 6}}), 1.0);
    const Cloud line{{-2, 0, 0}, {0, 0, 0}, {3, 0, 0}};
    close(quaternion_aligned_rmsd(line, moved(line)), 0.0);
    close(quaternion_aligned_rmsd(line, {{2, 0, 0}, {0, 0, 0}, {-3, 0, 0}}), 0.0);
    close(quaternion_aligned_rmsd({{2, 2, 2}, {2, 2, 2}}, {{-3, 8, 1}, {-3, 8, 1}}), 0.0);
    close(quaternion_aligned_rmsd({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
                                   {{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}}), std::sqrt(2.0 / 3.0));
    for (const double scale : {1e-150, 1e150})
        close(quaternion_aligned_rmsd({{scale, 0, 0}, {-scale, 0, 0}},
                                     {{0, 2 * scale, 0}, {0, -2 * scale, 0}}) / scale, 1.0);
}

void test_quaternion_validation()
{
    rejects([] { quaternion_aligned_rmsd({}, {}); });
    rejects([] { quaternion_aligned_rmsd({{0, 0, 0}}, {}); });
    rejects([] { quaternion_aligned_rmsd({{0, 0, 0}}, {{0, 0, 0}, {1, 0, 0}}); });
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity()}) {
        rejects([&] { quaternion_aligned_rmsd({{invalid, 0, 0}}, {{0, 0, 0}}); });
        rejects([&] { quaternion_aligned_rmsd({{0, 0, 0}}, {{0, invalid, 0}}); });
    }
}

std::string ethanol(bool moved_coordinates)
{
    return std::string("@<TRIPOS>MOLECULE\nethanol\n3 2 1 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n") +
        (moved_coordinates
            ? "1 C1 20 30 40 C.3 1 ETH 0\n2 C2 21.5 30 40 C.3 1 ETH 0\n3 O1 22 31.2 40 O.3 1 ETH 0\n"
            : "1 C1 0 0 0 C.3 1 ETH 0\n2 C2 1.5 0 0 C.3 1 ETH 0\n3 O1 2 1.2 0 O.3 1 ETH 0\n") +
        "@<TRIPOS>BOND\n1 1 2 1\n2 2 3 1\n@<TRIPOS>SUBSTRUCTURE\n1 ETH 1\n";
}

void test_tiny_sequence()
{
    const auto first = conformer_experiment::prepare_quality_input(ethanol(false));
    const auto second = conformer_experiment::prepare_quality_input(ethanol(true));
    require(first.graph.getNumConformers() == 0 && second.graph.getNumConformers() == 0,
            "Reference coordinates reached generation graph");
    require(first.undefined_stereo == 0 && first.heavy_indices.size() == 3 &&
            first.atom_ids == std::vector<unsigned int>{1, 2, 3}, "Unexpected fixed atom mapping");
    require(RDKit::MolToSmiles(first.graph) == RDKit::MolToSmiles(second.graph),
            "Generation graph depends on native coordinate values");
    conformer_experiment::ConformerOptions options;
    options.count = 2;
    options.min_rmsd = 0;
    options.compute_diversity = false;
    const auto a = conformer_experiment::generate_conformers(first.graph, options);
    const auto b = conformer_experiment::generate_conformers(second.graph, options);
    require(a.embedded == 2 && a.conformers.size() == 2 && b.conformers.size() == 2 && !a.mmff_used,
            "Raw two-conformer sequence was filtered or minimized");
    for (std::size_t c = 0; c < 2; ++c) {
        require(a.conformers[c].id == b.conformers[c].id && !a.conformers[c].minimum_aligned_rmsd,
                "Raw ETKDG ordering/diversity policy changed");
        const auto x = conformer_experiment::heavy_coordinates(a.conformers[c].molecule, first.heavy_indices);
        const auto y = conformer_experiment::heavy_coordinates(b.conformers[c].molecule, second.heavy_indices);
        require(x == y, "Same-seed sequence depends on input coordinates");
        require(std::isfinite(quaternion_aligned_rmsd(x,
                    conformer_experiment::heavy_coordinates(first.reference, first.heavy_indices))),
                "Tiny synthetic aligned RMSD is not finite");
    }
}

} // namespace

int main()
{
    try {
        test_quaternion_geometry();
        test_quaternion_validation();
        test_tiny_sequence();
        std::cout << "Quaternion geometry quality smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
