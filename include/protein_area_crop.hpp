#ifndef PROTEIN_AREA_CROP_HPP
#define PROTEIN_AREA_CROP_HPP

#include <GraphMol/RWMol.h>
#include <interaction/interaction_area.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

// Unset disables cropping. A supplied margin must be finite and positive, in A.
std::optional<double> protein_area_crop_margin();

// Retain whole PDB residues touching the pocket bounding box expanded by margin.
// Reparse and sanitize the retained ATOM/HETATM records, preserving their order.
// Expects a single-model PDB; other record types, including CONECT, are not copied.
std::unique_ptr<RDKit::RWMol> load_area_protein_crop(
    const std::filesystem::path& path, const RDKit::ROMol& pocket, double margin);

// Extract areas from either the original molecule or a separate cropped PDB.
// The full protein is never cropped: occupancy, clearance and Vinardo keep it.
// Returned areas own their coordinates and outlive the temporary cropped molecule.
std::vector<interaction::interaction_area> prepare_protein_areas(
    const std::filesystem::path& path, RDKit::RWMol& full_protein,
    const RDKit::ROMol& pocket);

#endif
