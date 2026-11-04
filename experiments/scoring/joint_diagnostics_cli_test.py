"""Exercise diagnostic CLI failures without molecular datasets or pose search."""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = Path(sys.argv.pop(1)).resolve()
ETHANE_MOL2 = """@<TRIPOS>MOLECULE
ethane
2 1 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0
2 C2 1.5 0.0 0.0 C.3 1 LIG 0.0
@<TRIPOS>BOND
1 1 2 1
"""
GLYCEROL_MOL2 = """@<TRIPOS>MOLECULE
glycerol
6 5 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0
2 C2 1.5 0.0 0.0 C.3 1 LIG 0.0
3 C3 2.8 0.6 0.0 C.3 1 LIG 0.0
4 O1 -0.4 1.3 0.2 O.3 1 LIG 0.0
5 O2 1.5 -1.3 0.2 O.3 1 LIG 0.0
6 O3 3.5 1.7 -0.2 O.3 1 LIG 0.0
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
3 1 4 1
4 2 5 1
5 3 6 1
"""


class JointDiagnosticsCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="joint-diagnostics-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.complex = self.root / "missing-complex"
        self.output = self.root / "output"

    def run_driver(self, *options):
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("DOCKING_"):
                del environment[name]
        return subprocess.run(
            [str(EXECUTABLE), str(self.complex), "--joint-output", str(self.output), *options],
            env=environment, capture_output=True, text=True, timeout=10, check=False)

    def status(self):
        with (self.output / "status.csv").open(newline="") as source:
            rows = list(csv.reader(source))
        self.assertTrue(all(len(row) == 2 for row in rows))
        self.assertEqual(rows[0], ["key", "value"])
        return dict(rows[1:])

    def test_diagnostics_rejects_prefilter_before_creating_output(self):
        result = self.run_driver("--diagnostics", "--prefilter", "500")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("omit --prefilter", result.stderr)
        self.assertFalse(self.output.exists())

    def test_reference_requires_diagnostics(self):
        result = self.run_driver("--reference", str(self.root / "reference.mol2"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--reference requires --diagnostics", result.stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_reference_has_quoted_failure_status(self):
        missing = self.root / 'missing,"reference.mol2'
        result = self.run_driver("--diagnostics", "--reference", str(missing))
        self.assertNotEqual(result.returncode, 0)
        status = self.status()
        self.assertEqual(status["status"], "invalid_reference")
        self.assertEqual(status["stage"], "reference_loading")
        self.assertEqual(status["completed"], "0")
        self.assertEqual(status["input_survivors"], "")
        self.assertIn(str(missing), status["message"])
        self.assertGreaterEqual(float(status["wall_before_status_s"]), 0)
        self.assertFalse((self.output / "summary.csv").exists())

    def test_input_failure_is_distinct_from_invalid_reference(self):
        reference = self.root / "reference.mol2"
        reference.write_text(ETHANE_MOL2)
        result = self.run_driver("--diagnostics", "--reference", str(reference))
        self.assertNotEqual(result.returncode, 0)
        status = self.status()
        self.assertEqual(status["status"], "error")
        self.assertEqual(status["stage"], "protein_preparation")
        self.assertEqual(status["completed"], "0")

    def test_legacy_failure_does_not_create_diagnostic_files(self):
        result = self.run_driver()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.output / "status.csv").exists())

    def test_synthetic_empty_pool_exports_complete_diagnostics(self):
        self.complex.mkdir()
        prefix = self.complex / self.complex.name
        pdb = "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \nEND\n"
        Path(str(prefix) + "_protein.pdb").write_text(pdb)
        Path(str(prefix) + "_pocket.pdb").write_text(pdb)
        Path(str(prefix) + "_ligand.mol2").write_text(GLYCEROL_MOL2)
        result = self.run_driver("--diagnostics", "--cutoff", "24")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.status()["status"], "no_poses")
        self.assertEqual(self.status()["completed"], "1")
        for filename, count in (("poses.csv", 0), ("rankings.csv", 0),
                                ("summary.csv", 10), ("topk_metrics.csv", 200)):
            with (self.output / filename).open(newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
                self.assertEqual(len(rows), count)
                self.assertTrue(all(None not in row and None not in row.values() for row in rows))
        with (self.output / "protocol.csv").open(newline="") as source:
            rows = list(csv.reader(source))
        self.assertTrue(all(len(row) == 2 for row in rows))
        protocol = dict(rows[1:])
        self.assertEqual(protocol["input_survivors"], "0")
        self.assertEqual(protocol["exported_poses"], "0")
        self.assertEqual(protocol["fast_top_n"], "none")
        self.assertEqual(protocol["hybrid_alpha"], "0.5")
        self.assertEqual(protocol["hybrid_lambda"], "0.3")
        self.assertIn("direct heavy atoms", protocol["rmsd"])

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("preserve this output\n")
        result = self.run_driver("--diagnostics")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output directory already exists", result.stderr)
        self.assertEqual(sentinel.read_text(), "preserve this output\n")
        self.assertEqual(list(self.output.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
