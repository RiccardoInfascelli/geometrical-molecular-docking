"""Prepare external MOL2 conformers without changing the docking engine."""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time


@dataclass
class Mol2Template:
    lines: list
    name_line: int
    atom_lines: list
    coordinates: list

    @classmethod
    def parse(cls, text):
        lines = text.splitlines(keepends=True)
        markers = [i for i, line in enumerate(lines)
                   if line.startswith("@<TRIPOS>MOLECULE") and line.strip() == "@<TRIPOS>MOLECULE"]
        if len(markers) != 1 or markers[0] + 2 >= len(lines):
            raise ValueError("Expected one complete MOL2 molecule")
        atom_lines = []
        in_atoms = False
        for i, line in enumerate(lines):
            if line.startswith("@<TRIPOS>"):
                in_atoms = line.strip() == "@<TRIPOS>ATOM"
            elif in_atoms and line.strip() and not line.lstrip().startswith("#"):
                atom_lines.append(i)
        expected = int(lines[markers[0] + 2].split()[0])
        if not atom_lines or len(atom_lines) != expected:
            raise ValueError("MOL2 atom count differs from its ATOM section")
        coordinates = []
        for ordinal, line_index in enumerate(atom_lines, 1):
            fields = lines[line_index].split()
            if len(fields) < 6 or int(fields[0]) != ordinal:
                raise ValueError("This experiment requires ordered MOL2 atom IDs 1..N")
            point = tuple(float(value) for value in fields[2:5])
            if not all(math.isfinite(value) for value in point):
                raise ValueError("MOL2 contains nonfinite coordinates")
            coordinates.append(point)
        return cls(lines, markers[0] + 1, atom_lines, coordinates)

    def render(self, coordinates, name):
        if len(coordinates) != len(self.atom_lines) or "\n" in name or "\r" in name:
            raise ValueError("Invalid coordinate count or molecule name")
        lines = self.lines.copy()
        ending = "\r\n" if lines[self.name_line].endswith("\r\n") else "\n"
        lines[self.name_line] = name + ending
        for index, point in zip(self.atom_lines, coordinates):
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("Invalid exported coordinates")
            tokens = list(re.finditer(r"\S+", lines[index]))
            # Keep atom names, types, IDs, charges, bonds and all other sections.
            lines[index] = (lines[index][:tokens[2].start()] +
                            " ".join(f"{value:.8f}" for value in point) +
                            lines[index][tokens[4].end():])
        result = "".join(lines)
        return result if result.endswith("\n") else result + "\n"


def rdkit_modules():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem, rdMolAlign
    except ImportError as error:
        raise RuntimeError("RDKit Python is not installed in this interpreter; no dependencies were installed") from error
    if not hasattr(AllChem, "ETKDGv3"):
        raise RuntimeError("This RDKit version does not provide ETKDGv3")
    for name in ("EmbedMultipleConfs", "MMFFHasAllMoleculeParams", "MMFFOptimizeMoleculeConfs"):
        if not hasattr(AllChem, name):
            raise RuntimeError(f"This RDKit installation does not provide the required API {name}")
    return Chem, rdBase, AllChem, rdMolAlign


def graph_signature(molecule):
    atoms = [(atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
              int(atom.GetChiralTag()), atom.GetIsAromatic()) for atom in molecule.GetAtoms()]
    bonds = sorted((min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), str(bond.GetBondType()),
                    bond.GetIsAromatic(), int(bond.GetStereo())) for bond in molecule.GetBonds())
    return atoms, bonds


def load_mol2(Chem, text):
    # Match the engine's MOL2 functional-group cleanup; never rewrite source fields.
    molecule = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False, cleanupSubstructures=True)
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError("RDKit could not parse a single coordinate conformer")
    return molecule


def select_conformers(candidates, rmsd, maximum, minimum_rmsd, energy_window):
    """Greedy energy-first selection; rmsd is an aligned, fixed-map distance."""
    usable = []
    for row in candidates:
        if row["status"] == "mmff_unsupported" and row["energy"] is None:
            usable.append(row)
        elif row["status"] == "converged" and row["energy"] is not None and math.isfinite(row["energy"]):
            usable.append(row)
    if not usable:
        raise ValueError("No valid converged conformers; do not silently change the force field")
    with_energy = all(row["energy"] is not None for row in usable)
    if not with_energy and any(row["energy"] is not None for row in usable):
        raise ValueError("Do not mix minimized and unminimized conformers in one selection")
    if with_energy:
        usable.sort(key=lambda row: (row["energy"], row["conf_id"]))
    selected = []
    best_energy = usable[0]["energy"]
    for row in usable:
        if with_energy and row["energy"] > best_energy + energy_window:
            row["reason"] = "outside_energy_window"
            continue
        row["minimum_aligned_rmsd"] = min(
            (rmsd(row["conf_id"], other["conf_id"]) for other in selected), default=None)
        if row["minimum_aligned_rmsd"] is not None and row["minimum_aligned_rmsd"] < minimum_rmsd:
            row["reason"] = "geometric_duplicate"
        elif len(selected) >= maximum:
            row["reason"] = "selection_limit"
        else:
            row["reason"] = "selected"
            selected.append(row)
    return selected


def prepare(input_path, output, count=10, candidates_count=30, seed=20261023,
            minimum_rmsd=0.5, energy_window=10.0, max_iterations=500):
    started = time.perf_counter()
    if output.exists():
        raise ValueError("Output directory already exists")
    if count < 1 or candidates_count < count or max_iterations < 1 or not 0 <= seed <= 2147483647:
        raise ValueError("Invalid conformer count, candidate count, iteration limit or seed")
    if not all(math.isfinite(value) and value >= 0 for value in (minimum_rmsd, energy_window)):
        raise ValueError("Diversity threshold and energy window must be finite and non-negative")
    Chem, rdBase, AllChem, rdMolAlign = rdkit_modules()
    text = input_path.read_text()
    template = Mol2Template.parse(text)
    source = load_mol2(Chem, text)
    if source.GetNumAtoms() != len(template.atom_lines) or len(Chem.GetMolFrags(source)) != 1:
        raise ValueError("Expected a single connected ligand with unchanged MOL2 atom order")
    n_atoms = source.GetNumAtoms()
    for i, expected in enumerate(template.coordinates):
        actual = source.GetConformer().GetAtomPosition(i)
        if any(abs(actual[k] - expected[k]) > 1e-6 for k in range(3)):
            raise ValueError("RDKit atom order does not match MOL2 coordinates")
        atom = source.GetAtomWithIdx(i)
        if atom.HasProp("_TriposAtomName") and atom.GetProp("_TriposAtomName") != template.lines[template.atom_lines[i]].split()[1]:
            raise ValueError("RDKit atom names do not match MOL2 order")
    heavy = [atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        raise ValueError("No heavy atoms available for diversity and docking RMSD")

    work = Chem.AddHs(Chem.Mol(source))
    for i in range(n_atoms):
        original, copied = source.GetAtomWithIdx(i), work.GetAtomWithIdx(i)
        if (original.GetAtomicNum(), original.GetIsotope(), original.GetFormalCharge()) != (
                copied.GetAtomicNum(), copied.GetIsotope(), copied.GetFormalCharge()):
            raise ValueError("Adding working hydrogens changed existing atom identity/order")
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.pruneRmsThresh = -1.0
    parameters.enforceChirality = True
    parameters.clearConfs = True
    embed_start = time.perf_counter()
    ids = list(AllChem.EmbedMultipleConfs(work, numConfs=candidates_count, params=parameters))
    embedding_seconds = time.perf_counter() - embed_start
    if not ids:
        raise ValueError("ETKDGv3 did not generate conformers")
    mmff_start = time.perf_counter()
    mmff_supported = bool(AllChem.MMFFHasAllMoleculeParams(work))
    if mmff_supported:
        optimized = AllChem.MMFFOptimizeMoleculeConfs(
            work, numThreads=1, maxIters=max_iterations, mmffVariant="MMFF94")
        if len(optimized) != len(ids):
            raise ValueError("MMFF result count does not match the embedded conformers")
    else:
        optimized = [(None, None)] * len(ids)
    mmff_seconds = time.perf_counter() - mmff_start
    candidates = []
    for conf_id, (status, energy) in zip(ids, optimized):
        valid_energy = energy is not None and math.isfinite(energy)
        candidates.append({"conf_id": conf_id, "energy": energy if valid_energy else None,
                           "status": "mmff_unsupported" if not mmff_supported else
                           "converged" if status == 0 and valid_energy else "optimization_failed",
                           "reason": "optimization_failed", "minimum_aligned_rmsd": None})
    atom_map = [(i, i) for i in heavy]

    def rmsd(first, second):
        value, _ = rdMolAlign.GetAlignmentTransform(
            work, work, prbCid=first, refCid=second, atomMap=atom_map, reflect=False)
        if not math.isfinite(value):
            raise ValueError("Nonfinite diversity RMSD")
        return value

    selected = select_conformers(candidates, rmsd, count, minimum_rmsd, energy_window)
    baseline_lines = template.lines.copy()
    ending = "\r\n" if baseline_lines[template.name_line].endswith("\r\n") else "\n"
    baseline_lines[template.name_line] = "baseline" + ending
    baseline_block = "".join(baseline_lines)
    blocks = [baseline_block if baseline_block.endswith("\n") else baseline_block + "\n"]
    records = [{"record_index": 1, "molecule_id": "baseline", "source": "crystal_original",
                "rdkit_conf_id": "", "mmff_energy_kcal_mol": "", "mmff_status": "not_minimized",
                "minimum_aligned_rmsd_A": ""}]
    signature = graph_signature(source)
    for ordinal, row in enumerate(selected, 1):
        conformer = work.GetConformer(row["conf_id"])
        coordinates = [tuple(conformer.GetAtomPosition(i)) for i in range(n_atoms)]
        name = f"conformer_{ordinal:03d}"
        block = template.render(coordinates, name)
        exported = load_mol2(Chem, block)
        if graph_signature(exported) != signature:
            raise ValueError("Coordinate export changed atom order, chemistry or stereochemistry")
        blocks.append(block)
        records.append({"record_index": ordinal + 1, "molecule_id": name, "source": "etkdgv3",
                        "rdkit_conf_id": row["conf_id"], "mmff_energy_kcal_mol": row["energy"],
                        "mmff_status": row["status"], "minimum_aligned_rmsd_A": row["minimum_aligned_rmsd"]})
    if graph_signature(load_mol2(Chem, blocks[0])) != signature:
        raise ValueError("Baseline round-trip changed molecule identity")

    output.mkdir(parents=True, exist_ok=False)
    (output / "conformers.mol2").write_text("".join(blocks))
    (output / "reference.mol2").write_text(text)
    with (output / "conformers.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "candidates.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    protocol = {"input": str(input_path.resolve()), "rdkit_version": rdBase.rdkitVersion,
                "python": sys.version, "method": "ETKDGv3", "seed": seed, "num_threads": 1,
                "embedding_prune_rms_threshold": -1.0, "enforce_chirality": True, "clear_working_conformers": True,
                "mol2_parser": {"sanitize": True, "removeHs": False, "cleanupSubstructures": True},
                "requested_candidates": candidates_count, "embedded_candidates": len(ids),
                "requested_selected": count, "selected": len(selected), "mmff_variant": "MMFF94",
                "mmff_supported": mmff_supported, "mmff_max_iterations": max_iterations,
                "energy_window_kcal_mol": energy_window, "diversity_aligned_heavy_rmsd_A": minimum_rmsd,
                "diversity_mapping": "fixed atom indices, proper alignment, no symmetry search",
                "original_atoms": n_atoms,
                "original_explicit_H": sum(atom.GetAtomicNum() == 1 for atom in source.GetAtoms()),
                "working_added_H": work.GetNumAtoms() - n_atoms,
                "export": "MOL2 names and coordinates only; original fields and original H retained",
                "baseline": "original coordinates, not MMFF minimized; first record",
                "mmff_fallback": "none; unsupported molecules retain ETKDG order and no energy",
                "embedding_s": embedding_seconds, "mmff_s": mmff_seconds,
                "preparation_wall_s": time.perf_counter() - started}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-conformers", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--min-rmsd", type=float, default=0.5)
    parser.add_argument("--energy-window", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    try:
        if args.check:
            _, rdBase, AllChem, _ = rdkit_modules()
            print(json.dumps({"python": sys.version.split()[0], "rdkit": rdBase.rdkitVersion,
                              "ETKDGv3": True, "MMFF94_API": hasattr(AllChem, "MMFFOptimizeMoleculeConfs")}))
        else:
            if args.input is None or args.output is None:
                parser.error("--input and --output are required")
            result = prepare(args.input, args.output, args.num_conformers, args.num_candidates,
                             args.seed, args.min_rmsd, args.energy_window, args.max_iterations)
            print(f"Prepared baseline + {result['selected']} selected conformers in {args.output}")
            if not result["mmff_supported"]:
                print("MMFF94 does not support this molecule: energies are unavailable; selection used diversity only")
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Conformer preparation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
