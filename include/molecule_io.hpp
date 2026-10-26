#ifndef MOLECULE_IO_HPP
#define MOLECULE_IO_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <GraphMol/RWMol.h>

// Return an owned molecule; parsing disables automatic hydrogen removal.
// Open/parse failures throw.
// Protein PDB and ligand MOL2 are sanitized through the interaction library;
// pocket PDB is loaded without sanitization as a geometric reference.
std::unique_ptr<RDKit::RWMol> load_protein(const std::string& path);
std::unique_ptr<RDKit::RWMol> load_pocket(const std::string& path);
std::unique_ptr<RDKit::RWMol> load_ligand(const std::string& path);

// Parse and sanitize one complete MOL2 record, preserving explicit hydrogens.
std::unique_ptr<RDKit::RWMol> parse_ligand_mol2_block(const std::string& block);

// Streams one MOL2 text record at a time, retaining the next marker between calls.
// RDKit parsing, sanitization and the response to malformed records belong to
// callers. Names lose a trailing CR; empty names become one-based record indices.
// A marker must start at column zero; text before the first marker is skipped.
// Returned lines end with '\n'; other characters are retained.
class Mol2Reader
{
public:
    explicit Mol2Reader(const std::filesystem::path& path);

    // Returns false when no record remains, clearing block but retaining name/index.
    bool next(std::string& block, std::string& name);
    std::size_t index() const { return index_; }

private:
    std::ifstream input_;
    std::string pending_marker_;
    std::size_t index_ = 0;
};

#endif // MOLECULE_IO_HPP
