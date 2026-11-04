#ifndef CONFORMER_QUALITY_INPUT_HPP
#define CONFORMER_QUALITY_INPUT_HPP

#include <GraphMol/RWMol.h>
#include <array>
#include <string>
#include <vector>

namespace conformer_experiment {

struct QualityInput {
    RDKit::RWMol graph;
    RDKit::RWMol reference;
    std::vector<unsigned int> atom_ids, heavy_indices;
    std::size_t undefined_stereo = 0, native_defined_stereo = 0;
};

// Coordinates are removed from the MOL2 text before parsing the generation
// graph. Native 3D stereochemistry is discarded, not used as an embedding hint.
// The independently parsed crystal molecule supplies evaluation coordinates only.
QualityInput prepare_quality_input(const std::string& mol2);
std::vector<std::array<double, 3>> heavy_coordinates(
    const RDKit::ROMol& molecule, const std::vector<unsigned int>& indices);

} // namespace conformer_experiment
#endif
