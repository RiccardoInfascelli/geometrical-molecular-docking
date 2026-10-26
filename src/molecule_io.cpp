#include "molecule_io.hpp"

#include <interaction/sanitize_mol.hpp>

#include <GraphMol/FileParsers/FileParsers.h>

#include <fstream>
#include <sstream>
#include <stdexcept>


namespace
{

std::unique_ptr<RDKit::RWMol> load_pdb(
    const std::string& path,
    bool sanitize)
{
    std::ifstream file(path);

    if (!file)
        throw std::runtime_error(
            "Cannot open PDB file: " + path
        );

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::unique_ptr<RDKit::RWMol> molecule(
        RDKit::PDBBlockToMol(
            buffer.str(),
            false,
            false
        )
    );

    if (!molecule)
        throw std::runtime_error(
            "Cannot parse PDB file: " + path
        );

    if (sanitize)
        interaction::sanitize_molecule(molecule);

    return molecule;
}

}


std::unique_ptr<RDKit::RWMol> load_protein(
    const std::string& path)
{
    return load_pdb(path, true);
}


std::unique_ptr<RDKit::RWMol> load_pocket(
    const std::string& path)
{
    return load_pdb(path, false);
}


std::unique_ptr<RDKit::RWMol> load_ligand(
    const std::string& path)
{
    std::ifstream file(path);

    if (!file)
        throw std::runtime_error(
            "Cannot open ligand file: " + path
        );

    std::stringstream buffer;
    buffer << file.rdbuf();

    return parse_ligand_mol2_block(buffer.str());
}

std::unique_ptr<RDKit::RWMol> parse_ligand_mol2_block(const std::string& block)
{
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::Mol2BlockToMol(block, false, false));

    if (!molecule)
        throw std::runtime_error(
            "Cannot parse ligand MOL2 record"
        );

    interaction::sanitize_molecule(molecule);

    return molecule;
}

Mol2Reader::Mol2Reader(const std::filesystem::path& path) : input_(path)
{
    if (!input_) throw std::runtime_error("Cannot open input MOL2");
}

bool Mol2Reader::next(std::string& block, std::string& name)
{
    block.clear();
    std::string line;
    if (pending_marker_.empty()) {
        while (std::getline(input_, line)) {
            if (line.rfind("@<TRIPOS>MOLECULE", 0) == 0) {
                pending_marker_ = line;
                break;
            }
        }
    }
    if (pending_marker_.empty()) return false;
    ++index_;
    block = pending_marker_ + '\n';
    pending_marker_.clear();
    if (!std::getline(input_, name)) name.clear();
    if (!name.empty() && name.back() == '\r') name.pop_back();
    if (name.empty()) name = std::to_string(index_);
    block += name + '\n';
    while (std::getline(input_, line)) {
        if (line.rfind("@<TRIPOS>MOLECULE", 0) == 0) {
            pending_marker_ = line;
            break;
        }
        block += line;
        block += '\n';
    }
    return true;
}
