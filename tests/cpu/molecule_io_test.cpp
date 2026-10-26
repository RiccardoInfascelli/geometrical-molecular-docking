#include "molecule_io.hpp"
#include "protein_area_crop.hpp"
#include "interaction_areas.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MonomerInfo.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Own only a newly created temporary directory, never an existing input path.
struct TemporaryDirectory
{
    fs::path path = fs::temp_directory_path() /
        ("docking-molecule-io-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));

    TemporaryDirectory()
    {
        require(fs::create_directory(path), "Cannot create a fresh test directory");
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

void write_input(const fs::path& path, const std::string& text)
{
    std::ofstream file(path, std::ios::binary);
    require(static_cast<bool>(file), "Cannot create synthetic MOL2 input");
    file << text;
    file.close();
    require(static_cast<bool>(file), "Cannot write synthetic MOL2 input");
}

void test_records(const fs::path& directory)
{
    const auto path = directory / "records.mol2";
    write_input(path,
        "preamble\n"
        "@<TRIPOS>MOLECULE\nfirst\nfirst body\n"
        "  @<TRIPOS>MOLECULE\n"
        "@<TRIPOS>MOLECULE extra\n\nsecond body\n"
        "@<TRIPOS>MOLECULE\nthird\nlast body");
    Mol2Reader reader(path);
    std::string block, name;
    require(reader.index() == 0, "Reader index must start at zero");
    require(reader.next(block, name) && reader.index() == 1 && name == "first" &&
            block == "@<TRIPOS>MOLECULE\nfirst\nfirst body\n  @<TRIPOS>MOLECULE\n",
            "Preamble or indented marker handling changed");
    require(reader.next(block, name) && reader.index() == 2 && name == "2" &&
            block == "@<TRIPOS>MOLECULE extra\n2\nsecond body\n",
            "Marker prefix recognition or empty-name fallback changed");
    require(reader.next(block, name) && reader.index() == 3 && name == "third" &&
            block == "@<TRIPOS>MOLECULE\nthird\nlast body\n",
            "Final record without a newline changed");
    for (int i = 0; i < 2; ++i) {
        block = "stale";
        require(!reader.next(block, name) && block.empty() && name == "third" &&
                reader.index() == 3, "EOF must preserve the last name and index");
    }
}

void test_record_edges(const fs::path& directory)
{
    std::string block, name;
    const auto crlf_path = directory / "crlf.mol2";
    write_input(crlf_path, "@<TRIPOS>MOLECULE\r\n spaced name \r\nBODY\r\n");
    Mol2Reader crlf(crlf_path);
    require(crlf.next(block, name) && name == " spaced name " && crlf.index() == 1 &&
            block == "@<TRIPOS>MOLECULE\r\n spaced name \nBODY\r\n",
            "Only the name's trailing CR should be removed");

    const auto missing_name_path = directory / "missing-name.mol2";
    write_input(missing_name_path, "@<TRIPOS>MOLECULE");
    Mol2Reader missing_name(missing_name_path);
    require(missing_name.next(block, name) && name == "1" && missing_name.index() == 1 &&
            block == "@<TRIPOS>MOLECULE\n1\n", "Missing-name fallback changed");

    const auto adjacent_path = directory / "adjacent.mol2";
    write_input(adjacent_path, "@<TRIPOS>MOLECULE\n@<TRIPOS>MOLECULE\nbody\n");
    Mol2Reader adjacent(adjacent_path);
    require(adjacent.next(block, name) && name == "@<TRIPOS>MOLECULE" &&
            adjacent.index() == 1 && block == "@<TRIPOS>MOLECULE\n@<TRIPOS>MOLECULE\nbody\n" &&
            !adjacent.next(block, name), "A marker consumed as a name must not be reinterpreted");

    const auto empty_path = directory / "empty.mol2";
    write_input(empty_path, "");
    Mol2Reader empty(empty_path);
    block = "stale";
    name = "unchanged";
    require(!empty.next(block, name) && block.empty() && name == "unchanged" &&
            empty.index() == 0, "Empty input handling changed");

    bool open_error = false;
    try {
        Mol2Reader missing(directory / "does-not-exist.mol2");
    } catch (const std::runtime_error& error) {
        open_error = std::string(error.what()) == "Cannot open input MOL2";
    }
    require(open_error, "MOL2 open error changed");
}

struct ScopedCropEnvironment
{
    static constexpr const char* name = "DOCKING_AREA_CROP_MARGIN_A";
    std::optional<std::string> previous;

    ScopedCropEnvironment()
    {
        if (const char* value = std::getenv(name)) previous = value;
    }

    ~ScopedCropEnvironment()
    {
        if (previous) ::setenv(name, previous->c_str(), 1);
        else ::unsetenv(name);
    }

    void set(const char* value)
    {
        require((value ? ::setenv(name, value, 1) : ::unsetenv(name)) == 0,
                "Cannot set the crop test environment");
    }
};

template <class Function>
void require_throws(Function action, const char* message)
{
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

std::string pdb_carbon(unsigned int serial, const char* record,
                       char chain, int residue, char insertion,
                       double x, double y, double z)
{
    std::string line(80, ' ');
    const auto integer_field = [](unsigned int value, int width) {
        std::ostringstream field;
        field << std::setw(width) << value;
        return field.str();
    };
    const auto coordinate_field = [](double value) {
        std::ostringstream field;
        field << std::fixed << std::setprecision(3) << std::setw(8) << value;
        return field.str();
    };
    line.replace(0, 6, record);
    line.replace(6, 5, integer_field(serial, 5));
    const std::string atom_name = "C" + std::to_string(serial);
    line.replace(12, atom_name.size(), atom_name);
    line.replace(17, 3, "ALA");
    line[21] = chain;
    line.replace(22, 4, integer_field(static_cast<unsigned int>(residue), 4));
    line[26] = insertion;
    line.replace(30, 8, coordinate_field(x));
    line.replace(38, 8, coordinate_field(y));
    line.replace(46, 8, coordinate_field(z));
    line.replace(54, 6, "  1.00");
    line.replace(60, 6, "  0.00");
    line[77] = 'C';
    return line + '\n';
}

struct HeavyAtomPosition
{
    unsigned int atomic_number;
    int serial;
    double x, y, z;

    bool operator==(const HeavyAtomPosition&) const = default;
};

std::vector<HeavyAtomPosition> heavy_atom_positions(const RDKit::ROMol& molecule)
{
    std::vector<HeavyAtomPosition> positions;
    const auto& conformer = molecule.getConformer();
    for (const auto* atom : molecule.atoms()) {
        if (atom->getAtomicNum() == 1) continue;
        const auto* residue = dynamic_cast<const RDKit::AtomPDBResidueInfo*>(
            atom->getMonomerInfo());
        require(residue != nullptr, "PDB roundtrip lost atom residue metadata");
        const auto& position = conformer.getAtomPos(atom->getIdx());
        positions.push_back({static_cast<unsigned int>(atom->getAtomicNum()), residue->getSerialNumber(),
                             position.x, position.y, position.z});
    }
    return positions;
}

void test_crop_environment()
{
    ScopedCropEnvironment environment;
    environment.set(nullptr);
    require(!protein_area_crop_margin(), "Unset crop selector must disable cropping");
    environment.set("8");
    const auto margin = protein_area_crop_margin();
    require(margin && *margin == 8.0, "Positive crop margin was not parsed");
    for (const char* invalid : {"", "0", "-8", "nan", "inf", "8junk", "off"}) {
        environment.set(invalid);
        require_throws([] { protein_area_crop_margin(); },
                       "Invalid crop selector must be rejected");
    }
}

void test_whole_residue_crop(const fs::path& directory)
{
    const auto protein_path = directory / "crop-protein.pdb";
    const auto pocket_path = directory / "crop-pocket.pdb";
    // The pocket bounds are [-2, 2] on every axis; margin 8 gives [-10, 10].
    write_input(pocket_path,
        pdb_carbon(1, "ATOM  ", 'P', 1, ' ', -2.0, -2.0, -2.0) +
        pdb_carbon(2, "ATOM  ", 'P', 2, ' ', 2.0, 2.0, 2.0) + "END\n");
    // Keep input order and serials, including an outside atom preceding the
    // selecting atom and HETATM sharing exactly the same residue key.
    write_input(protein_path,
        pdb_carbon(79, "ATOM  ", 'A', 1, ' ', 30.0, 30.0, 30.0) +
        pdb_carbon(3, "ATOM  ", 'A', 1, ' ', 10.0, 10.0, 10.0) +
        pdb_carbon(66, "HETATM", 'A', 1, ' ', -30.0, -30.0, -30.0) +
        pdb_carbon(99, "ATOM  ", 'B', 1, ' ', 45.0, 45.0, 45.0) +
        pdb_carbon(80, "ATOM  ", 'A', 1, 'A', 60.0, 60.0, 60.0) +
        pdb_carbon(31, "ATOM  ", 'A', 2, ' ', 10.001, -10.0, -10.0) + "END\n");
    auto full = load_protein(protein_path.string());
    const auto pocket = load_pocket(pocket_path.string());
    const auto original_positions = heavy_atom_positions(*full);
    const auto original_atom_count = full->getNumAtoms();
    const auto original_bond_count = full->getNumBonds();
    require(original_positions.size() == 6, "Synthetic full PDB lost heavy atoms");

    const auto crop = load_area_protein_crop(protein_path, *pocket, 8.0);
    const std::vector<HeavyAtomPosition> expected{
        {6, 79, 30.0, 30.0, 30.0},
        {6, 3, 10.0, 10.0, 10.0},
        {6, 66, -30.0, -30.0, -30.0}};
    require(heavy_atom_positions(*crop) == expected,
            "Crop must use inclusive box bounds, whole residues and original order/serials");

    ScopedCropEnvironment environment;
    environment.set("8");
    prepare_protein_areas(protein_path, *full, *pocket);
    environment.set(nullptr);
    // With cropping disabled, feature extraction must use the supplied full
    // molecule and must not reopen the source PDB.
    const auto baseline = get_protein_areas(*full);
    const auto disabled = prepare_protein_areas(directory / "absent-full-protein.pdb", *full, *pocket);
    require(disabled.size() == baseline.size(), "Disabled crop changed the area count");
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        const auto& a = baseline[i];
        const auto& b = disabled[i];
        require(a.pattern == b.pattern && a.pattern_class == b.pattern_class &&
                a.num_atoms == b.num_atoms && a.centroid.x == b.centroid.x &&
                a.centroid.y == b.centroid.y && a.centroid.z == b.centroid.z,
                "Disabled crop changed area identity or order");
        for (std::size_t j = 0; j < a.num_atoms; ++j)
            require(a.x[j] == b.x[j] && a.y[j] == b.y[j] && a.z[j] == b.z[j],
                    "Disabled crop changed area coordinates");
    }
    require(full->getNumAtoms() == original_atom_count &&
            full->getNumBonds() == original_bond_count &&
            heavy_atom_positions(*full) == original_positions,
            "Area preparation must leave the original full molecule unchanged");

    require_throws([&] {
        load_area_protein_crop(directory / "missing-protein.pdb", *pocket, 8.0);
    }, "Missing crop input must be rejected");
    const auto far_path = directory / "crop-far.pdb";
    write_input(far_path,
        pdb_carbon(1, "ATOM  ", 'A', 1, ' ', 100.0, 100.0, 100.0) + "END\n");
    require_throws([&] { load_area_protein_crop(far_path, *pocket, 8.0); },
                   "A crop selecting no atoms must be rejected");
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        require_throws([&] { load_area_protein_crop(protein_path, *pocket, invalid); },
                       "Invalid direct crop margins must be rejected");
    }
}

} // namespace

int main()
{
    try {
        const TemporaryDirectory directory;
        test_records(directory.path);
        test_record_edges(directory.path);
        test_crop_environment();
        test_whole_residue_crop(directory.path);
        std::cout << "Molecule I/O tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
