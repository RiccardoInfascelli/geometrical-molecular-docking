"""Synthetic checks for conformer preparation; no scientific inputs are read."""

import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from prepare_conformers import Mol2Template, prepare, select_conformers
from run_experiment import summarize


def temporary_directory():
    runs = Path(__file__).resolve().parent / "runs"
    runs.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="synthetic-test-", dir=runs)


METHANE = """@<TRIPOS>MOLECULE
synthetic_methane
5 4 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 LIG1        -0.4000
      2 H1          0.6291    0.6291    0.6291 H         1 LIG1         0.1000
      3 H2         -0.6291   -0.6291    0.6291 H         1 LIG1         0.1000
      4 H3         -0.6291    0.6291   -0.6291 H         1 LIG1         0.1000
      5 H4          0.6291   -0.6291   -0.6291 H         1 LIG1         0.1000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
     3    1    4 1
     4    1    5 1
@<TRIPOS>SUBSTRUCTURE
     1 LIG1         1 TEMP              0 ****  ****    0 ROOT
"""


def candidate(conf_id, energy, status="converged", **extra):
    return {"conf_id": conf_id, "energy": energy, "status": status,
            "reason": "optimization_failed", "minimum_aligned_rmsd": None, **extra}


class Mol2TemplateTest(unittest.TestCase):
    def test_export_changes_only_name_and_coordinates(self):
        template = Mol2Template.parse(METHANE)
        coordinates = [(x + 0.125, y - 0.25, z + 0.5)
                       for x, y, z in template.coordinates]
        exported = template.render(coordinates, "conformer_001")
        reread = Mol2Template.parse(exported)
        atom_lines = set(template.atom_lines)
        for index, (before, after) in enumerate(zip(template.lines, reread.lines)):
            if index == template.name_line:
                self.assertEqual(after, "conformer_001\n")
            elif index in atom_lines:
                original, changed = before.split(), after.split()
                self.assertEqual(original[:2], changed[:2])
                self.assertEqual(original[5:], changed[5:])
            else:
                self.assertEqual(before, after)
        self.assertEqual(len(reread.coordinates), 5)
        self.assertEqual([reread.lines[i].split()[1] for i in reread.atom_lines],
                         ["C1", "H1", "H2", "H3", "H4"])
        for actual, expected in zip(reread.coordinates, coordinates):
            for first, second in zip(actual, expected):
                self.assertAlmostEqual(first, second, places=8)

    def test_windows_line_endings_are_preserved(self):
        template = Mol2Template.parse(METHANE.replace("\n", "\r\n"))
        exported = template.render(template.coordinates, "windows")
        self.assertEqual(exported.count("\r\n"), exported.count("\n"))
        self.assertIn("windows\r\n", exported)

    def test_rejects_multiple_molecules_wrong_count_and_bad_order(self):
        cases = [METHANE + METHANE,
                 METHANE.replace("5 4 1 0 0", "6 4 1 0 0"),
                 METHANE.replace("      2 H1", "      1 H1"),
                 METHANE.replace("      2 H1", "      6 H1")]
        for text in cases:
            with self.subTest(text=text[:70]), self.assertRaises(ValueError):
                Mol2Template.parse(text)

    def test_rejects_nonfinite_or_malformed_coordinates(self):
        for value in ("nan", "inf", "-inf", "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Mol2Template.parse(METHANE.replace("0.0000", value, 1))

    def test_export_rejects_incomplete_coordinates_and_record_injection(self):
        template = Mol2Template.parse(METHANE)
        bad_coordinates = [template.coordinates[:-1],
                           [(math.nan, 0, 0)] + template.coordinates[1:],
                           [(0, 0)] + template.coordinates[1:]]
        for coordinates in bad_coordinates:
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                template.render(coordinates, "bad")
        for name in ("bad\n@<TRIPOS>MOLECULE", "bad\rname"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                template.render(template.coordinates, name)


class ConformerSelectionTest(unittest.TestCase):
    def test_energy_first_window_and_postoptimization_diversity(self):
        rows = [candidate(3, 1.0), candidate(1, 0.0),
                candidate(2, 0.1), candidate(4, 15.0)]
        calls = []

        def rmsd(first, second):
            calls.append((first, second))
            return 0.2 if {first, second} == {1, 2} else 0.8

        selected = select_conformers(rows, rmsd, maximum=10,
                                     minimum_rmsd=0.5, energy_window=10.0)
        self.assertEqual([row["conf_id"] for row in selected], [1, 3])
        self.assertEqual(rows[2]["reason"], "geometric_duplicate")
        self.assertEqual(rows[3]["reason"], "outside_energy_window")
        self.assertFalse(any(4 in call for call in calls))
        self.assertEqual(rows[0]["minimum_aligned_rmsd"], 0.8)

    def test_equal_energies_have_deterministic_id_tie_break(self):
        rows = [candidate(9, -2.0), candidate(3, -2.0), candidate(7, -1.0)]
        selected = select_conformers(rows, lambda first, second: 1.0,
                                     maximum=2, minimum_rmsd=0.5, energy_window=10.0)
        self.assertEqual([row["conf_id"] for row in selected], [3, 9])
        self.assertEqual(rows[2]["reason"], "selection_limit")

    def test_zero_diversity_threshold_and_energy_boundary_are_inclusive(self):
        rows = [candidate(0, 0.0), candidate(1, 10.0), candidate(2, 10.01)]
        selected = select_conformers(rows, lambda first, second: 0.0,
                                     maximum=10, minimum_rmsd=0.0, energy_window=10.0)
        self.assertEqual([row["conf_id"] for row in selected], [0, 1])

    def test_failed_optimization_is_not_selected_even_at_lower_energy(self):
        rows = [candidate(0, -100.0, "optimization_failed"), candidate(1, 2.0)]
        selected = select_conformers(rows, lambda first, second: 1.0,
                                     maximum=10, minimum_rmsd=0.5, energy_window=10.0)
        self.assertEqual([row["conf_id"] for row in selected], [1])
        self.assertEqual(rows[0]["reason"], "optimization_failed")

    def test_converged_status_requires_a_finite_energy(self):
        for invalid in (None, math.nan, math.inf, -math.inf):
            rows = [candidate(0, invalid), candidate(1, 2.0)]
            with self.subTest(energy=invalid):
                selected = select_conformers(rows, lambda first, second: 1.0,
                                             maximum=10, minimum_rmsd=0.5,
                                             energy_window=10.0)
                self.assertEqual([row["conf_id"] for row in selected], [1])

    def test_rejects_mixed_energy_and_unsupported_protocols(self):
        rows = [candidate(0, 0.0), candidate(1, None, "mmff_unsupported")]
        with self.assertRaises(ValueError):
            select_conformers(rows, lambda first, second: 1.0,
                              maximum=10, minimum_rmsd=0.5, energy_window=10.0)

    def test_unsupported_mmff_keeps_embedding_order_without_inventing_energy(self):
        rows = [candidate(conf_id, None, "mmff_unsupported") for conf_id in (9, 3, 7)]
        selected = select_conformers(rows, lambda first, second: 1.0,
                                     maximum=2, minimum_rmsd=0.5, energy_window=0.0)
        self.assertEqual([row["conf_id"] for row in selected], [9, 3])
        self.assertTrue(all(row["energy"] is None for row in rows))
        self.assertEqual(rows[2]["reason"], "selection_limit")

    def test_reference_rmsd_cannot_choose_conformers(self):
        rows = [candidate(0, 0.0, native_rmsd=9.0),
                candidate(1, 1.0, native_rmsd=0.1)]
        selected = select_conformers(rows, lambda first, second: 1.0,
                                     maximum=1, minimum_rmsd=0.5, energy_window=10.0)
        self.assertEqual(selected[0]["conf_id"], 0)

    def test_no_usable_candidates_fails_explicitly(self):
        for rows in ([], [candidate(0, None, "optimization_failed")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_conformers(rows, lambda first, second: 1.0,
                                  maximum=10, minimum_rmsd=0.5, energy_window=10.0)


class ExperimentSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.preparation = self.root / "preparation"
        self.docking = self.root / "docking"
        self.output = self.root / "analysis"
        for path in (self.preparation, self.docking, self.output):
            path.mkdir()
        self.metadata = [{"record_index": index, "molecule_id": name,
                          "source": "crystal_original" if index == 1 else "etkdgv3"}
                         for index, name in enumerate(("baseline", "conformer_001", "conformer_002"), 1)]
        self.poses = [
            self.pose(1, 0, 100.0, 0.1), self.pose(1, 1, 90.0, 0.05),
            self.pose(2, 0, 20.0, 3.0), self.pose(2, 1, 12.0, 0.7),
            self.pose(3, 0, 18.0, 1.1), self.pose(3, 1, 7.0, 0.3)]
        self.write_inputs()

    def pose(self, record, pose_id, score, rmsd):
        return {"record_index": record, "molecule_id": self.metadata[record - 1]["molecule_id"],
                "pose_id": pose_id, "fast_score": score, "rmsd_heavy_A": rmsd}

    @staticmethod
    def write(path, rows):
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read(path):
        with path.open(newline="") as source:
            return list(csv.DictReader(source))

    def record(self, index):
        poses = [pose for pose in self.poses if pose["record_index"] == index]
        top = min(poses, key=lambda pose: (-pose["fast_score"], pose["pose_id"])) if poses else None
        return {"record_index": index, "molecule_id": self.metadata[index - 1]["molecule_id"],
                "status": "ok" if poses else "no_poses", "ligand_triplets": 10,
                "candidate_matches": 100, "transforms_generated": 110, "survivors": len(poses),
                "top1_pose_id": top["pose_id"] if top else "",
                "top1_fast_score": top["fast_score"] if top else "",
                "top1_rmsd_heavy_A": top["rmsd_heavy_A"] if top else "",
                "success_lt_2A": int(top["rmsd_heavy_A"] < 2.0) if top else 0,
                "best_available_rmsd_heavy_A": min(pose["rmsd_heavy_A"] for pose in poses) if poses else "",
                "read_parse_preparation_s": 0.1, "pose_generation_s": 0.2,
                "fast_scoring_s": 0.3, "rmsd_evaluation_s": 0.4}

    def write_inputs(self, record_indices=(1, 2, 3)):
        self.write(self.preparation / "conformers.csv", self.metadata)
        self.write(self.docking / "records.csv", [self.record(index) for index in record_indices])
        self.write(self.docking / "poses.csv", self.poses)

    def test_fast_selection_and_oracle_are_separate_and_baseline_is_excluded(self):
        timings = summarize(self.preparation, self.docking, self.output)
        comparisons = {row["case"]: row for row in self.read(self.output / "comparison.csv")}
        chosen = comparisons["ensemble_selected_by_fast"]
        oracle = comparisons["ensemble_best_available_rmsd"]
        self.assertEqual(chosen["record_index"], "2")
        self.assertEqual(chosen["pose_id"], "0")
        self.assertEqual(float(chosen["rmsd_heavy_A"]), 3.0)
        self.assertEqual(chosen["success_lt_2A"], "0")
        self.assertEqual(oracle["record_index"], "3")
        self.assertEqual(oracle["pose_id"], "1")
        self.assertEqual(float(oracle["rmsd_heavy_A"]), 0.3)
        self.assertEqual(oracle["success_lt_2A"], "1")
        self.assertAlmostEqual(timings["baseline_online_s"], 0.6)
        self.assertAlmostEqual(timings["ensemble_online_s"], 1.2)
        summary = self.read(self.output / "summary.csv")
        self.assertAlmostEqual(float(summary[1]["delta_top1_rmsd_vs_baseline_A"]), 2.9)

    def test_fast_ties_use_record_then_pose_id(self):
        self.poses[2]["fast_score"] = 20.0
        self.poses[3]["fast_score"] = 20.0
        self.poses[4]["fast_score"] = 20.0
        self.write_inputs()
        summarize(self.preparation, self.docking, self.output)
        chosen = self.read(self.output / "comparison.csv")[1]
        self.assertEqual((chosen["record_index"], chosen["pose_id"]), ("2", "0"))

    def test_no_poses_keeps_missing_rmsd_and_failed_success_distinct(self):
        self.poses = [pose for pose in self.poses if pose["record_index"] == 1]
        self.write_inputs()
        summarize(self.preparation, self.docking, self.output)
        summary = self.read(self.output / "summary.csv")
        for row in summary[1:]:
            self.assertEqual(row["status"], "no_poses")
            self.assertEqual(row["top1_rmsd_heavy_A"], "")
            self.assertEqual(row["best_available_rmsd_heavy_A"], "")
            self.assertEqual(row["success_lt_2A"], "0")
        for row in self.read(self.output / "comparison.csv")[1:]:
            self.assertEqual(row["status"], "no_poses")
            self.assertEqual(row["rmsd_heavy_A"], "")
            self.assertEqual(row["success_lt_2A"], "0")

    def test_truncated_pose_export_is_rejected(self):
        self.write(self.docking / "poses.csv", self.poses[:-1])
        with self.assertRaisesRegex(ValueError, "every survivor"):
            summarize(self.preparation, self.docking, self.output)

    def test_duplicate_pose_id_is_rejected(self):
        self.write(self.docking / "poses.csv", self.poses + [self.poses[-1]])
        with self.assertRaisesRegex(ValueError, "duplicate pose"):
            summarize(self.preparation, self.docking, self.output)

    def test_missing_record_is_rejected_by_default(self):
        self.poses = [pose for pose in self.poses if pose["record_index"] != 3]
        self.write_inputs(record_indices=(1, 2))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(self.preparation, self.docking, self.output)

    def test_partial_run_does_not_claim_complete_ensemble_accuracy(self):
        self.poses = [pose for pose in self.poses if pose["record_index"] != 3]
        self.write_inputs(record_indices=(1, 2))
        summarize(self.preparation, self.docking, self.output, allow_incomplete=True)
        summary = self.read(self.output / "summary.csv")
        self.assertEqual(summary[2]["status"], "not_processed")
        self.assertEqual(summary[2]["top1_rmsd_heavy_A"], "")
        self.assertEqual(summary[2]["success_lt_2A"], "")
        for row in self.read(self.output / "comparison.csv")[1:]:
            self.assertEqual(row["status"], "incomplete_run")
            self.assertEqual(row["success_lt_2A"], "")


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit Python unavailable; synthetic integration skipped")
class RDKitPreparationTest(unittest.TestCase):
    def test_small_explicit_h_molecule_round_trip_and_existing_output_protection(self):
        from rdkit import Chem
        from prepare_conformers import graph_signature, load_mol2

        with temporary_directory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.mol2"
            source.write_text(METHANE)
            result = root / "prepared"
            protocol = prepare(source, result, count=2, candidates_count=3,
                               minimum_rmsd=0.0, energy_window=100.0)
            self.assertEqual(protocol["original_atoms"], 5)
            self.assertEqual(protocol["original_explicit_H"], 4)
            self.assertEqual(protocol["selected"], 2)
            expected = graph_signature(load_mol2(Chem, METHANE))
            blocks = (result / "conformers.mol2").read_text().split("@<TRIPOS>MOLECULE\n")[1:]
            self.assertEqual(len(blocks), 3)
            for block in blocks:
                molecule = load_mol2(Chem, "@<TRIPOS>MOLECULE\n" + block)
                self.assertEqual(graph_signature(molecule), expected)
            with (result / "conformers.csv").open() as input_file:
                records = list(csv.DictReader(input_file))
            self.assertEqual([row["record_index"] for row in records], ["1", "2", "3"])
            self.assertEqual(records[0]["source"], "crystal_original")
            self.assertEqual((result / "reference.mol2").read_text(), METHANE)
            before = (result / "conformers.mol2").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(source, result, count=2, candidates_count=3)
            self.assertEqual((result / "conformers.mol2").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
