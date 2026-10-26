#include "molecule_io.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

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

} // namespace

int main()
{
    try {
        const TemporaryDirectory directory;
        test_records(directory.path);
        test_record_edges(directory.path);
        std::cout << "MOL2 streaming tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
