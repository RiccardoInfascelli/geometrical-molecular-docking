#include "protein_area_crop.hpp"

#include "interaction_areas.hpp"
#include "protein_grid.hpp"

#include <GraphMol/FileParsers/FileParsers.h>
#include <interaction/sanitize_mol.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

std::optional<double> protein_area_crop_margin()
{
    const char* value = std::getenv("DOCKING_AREA_CROP_MARGIN_A");
    if (!value) return std::nullopt;
    try {
        const std::string text(value);
        std::size_t consumed = 0;
        const double margin = std::stod(text, &consumed);
        if (consumed == text.size() && std::isfinite(margin) && margin > 0.0)
            return margin;
    } catch (const std::exception&) {
        // Report every invalid selector value with the same actionable message.
    }
    throw std::invalid_argument("DOCKING_AREA_CROP_MARGIN_A must be finite and > 0; unset it to disable cropping");
}

std::unique_ptr<RDKit::RWMol> load_area_protein_crop(
    const std::filesystem::path& path, const RDKit::ROMol& pocket, double margin)
{
    if (pocket.getNumAtoms() == 0) throw std::runtime_error("Empty pocket for area crop");
    if (!std::isfinite(margin) || margin <= 0.0)
        throw std::invalid_argument("Area crop margin must be finite and > 0");
    const auto bounds = compute_pocket_bounds(pocket);
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open protein PDB for area crop: " + path.string());

    std::vector<std::string> lines;
    std::set<std::string> included_residues;
    std::string line;
    while (std::getline(input, line)) {
        if ((line.rfind("ATOM  ", 0) != 0 && line.rfind("HETATM", 0) != 0) || line.size() < 54)
            continue;
        lines.push_back(line);
        try {
            const double x = std::stod(line.substr(30, 8));
            const double y = std::stod(line.substr(38, 8));
            const double z = std::stod(line.substr(46, 8));
            if (x >= bounds.xmin - margin && x <= bounds.xmax + margin &&
                y >= bounds.ymin - margin && y <= bounds.ymax + margin &&
                z >= bounds.zmin - margin && z <= bounds.zmax + margin) {
                // PDB residue name, chain, number and insertion code. Alternate
                // locations and ATOM/HETATM records share this residue identity.
                included_residues.insert(line.substr(17, 10));
            }
        } catch (const std::exception&) {
            // A malformed coordinate cannot select a residue.
        }
    }
    if (input.bad()) throw std::runtime_error("Cannot read protein PDB for area crop");

    std::ostringstream pdb;
    std::size_t retained = 0;
    for (const auto& atom_line : lines) {
        if (included_residues.count(atom_line.substr(17, 10))) {
            pdb << atom_line << '\n';
            ++retained;
        }
    }
    if (retained == 0) throw std::runtime_error("Area crop did not retain any PDB atoms");
    // Reconstruct connectivity through the same PDB parser as the historical crop;
    // do not serialize the already sanitized full protein or copy dangling CONECTs.
    pdb << "END\n";
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::PDBBlockToMol(pdb.str(), false, false));
    if (!molecule || molecule->getNumAtoms() == 0)
        throw std::runtime_error("Failed parsing cropped protein PDB");
    interaction::sanitize_molecule(molecule);
    return molecule;
}

std::vector<interaction::interaction_area> prepare_protein_areas(
    const std::filesystem::path& path, RDKit::RWMol& full_protein,
    const RDKit::ROMol& pocket)
{
    const auto margin = protein_area_crop_margin();
    std::unique_ptr<RDKit::RWMol> cropped;
    if (margin) cropped = load_area_protein_crop(path, pocket, *margin);
    auto& area_protein = cropped ? *cropped : full_protein;
    std::ostringstream log;
    log << std::setprecision(17) << "AREA_CROP mode=" << (margin ? "whole_residue_box" : "off")
        << " margin_A=";
    if (margin) log << *margin;
    else log << "off";
    log << " protein_atoms=" << full_protein.getNumAtoms()
        << " area_protein_atoms=" << area_protein.getNumAtoms() << '\n';
    std::clog << log.str();
    return get_protein_areas(area_protein);
}
