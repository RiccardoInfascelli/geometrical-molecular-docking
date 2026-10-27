"""Mocked wrapper checks; no molecular inputs, CUDA or installed RDKit needed."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_1a1e


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "complex").mkdir()
        self.executable = self.root / "driver"
        self.executable.write_text("synthetic placeholder, never executed\n")
        self.executable.chmod(0o700)

    def args(self, *extra):
        return run_1a1e.arguments(["--complex", str(self.root / "complex"),
            "--executable", str(self.executable), "--output", str(self.root / "new"), *extra])

    def test_four_independent_runs_keep_parameters_and_record_evidence(self):
        args = self.args("--mmff", "--write-sdf", "--fast-prefilter", "500", "--crop-margin", "8")

        def complete(argv, **kwargs):
            output = Path(argv[2])
            self.assertFalse(output.exists())
            output.mkdir()
            # The wrapper intentionally does not depend on evolving summary fields.
            (output / "summary.csv").write_text("synthetic summary\n")
            self.assertIn("--mmff", argv)
            self.assertIn("--write-sdf", argv)
            self.assertNotIn("--kabsch", argv)
            self.assertNotIn("--original-only", argv)
            self.assertEqual(argv[argv.index("--fast-prefilter") + 1], "500")
            self.assertEqual(argv[argv.index("--seed") + 1], "20261027")
            return subprocess.CompletedProcess(argv, 0)

        with patch.object(run_1a1e, "provenance", return_value={"head": "synthetic"}), \
                patch.object(run_1a1e.subprocess, "run", side_effect=complete) as process:
            self.assertEqual(run_1a1e.run(args), 0)
        self.assertEqual(process.call_count, 4)
        counts = [int(call.args[0][call.args[0].index("--num-conformers") + 1])
                  for call in process.call_args_list]
        self.assertEqual(counts, [1, 5, 10, 20])
        protocol = json.loads((args.output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "complete")
        self.assertEqual(len(protocol["runs"]), 4)
        with (args.output / "runs.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertTrue(all(float(row["external_wall_s"]) >= 0 for row in rows))
        for count in counts:
            directory = args.output / f"conformers-{count}"
            for name in ("command.json", "stdout.log", "stderr.log"):
                self.assertTrue((directory / name).is_file())

    def test_failure_keeps_partial_output_and_stops_without_fabricated_runs(self):
        args = self.args()

        def fail(argv, **kwargs):
            output = Path(argv[2])
            output.mkdir()
            (output / "partial.csv").write_text("synthetic partial output\n")
            kwargs["stderr"].write("synthetic failure\n")
            return subprocess.CompletedProcess(argv, 7)

        with patch.object(run_1a1e, "provenance", return_value={}), \
                patch.object(run_1a1e.subprocess, "run", side_effect=fail) as process:
            self.assertEqual(run_1a1e.run(args), 1)
        self.assertEqual(process.call_count, 1)
        protocol = json.loads((args.output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "failed")
        self.assertEqual(len(protocol["runs"]), 1)
        self.assertEqual(protocol["runs"][0]["returncode"], 7)
        self.assertTrue((args.output / "conformers-1/docking/partial.csv").is_file())
        self.assertFalse((args.output / "conformers-5").exists())

    def test_missing_summary_is_failure_and_existing_output_is_refused(self):
        args = self.args()
        with patch.object(run_1a1e, "provenance", return_value={}), \
                patch.object(run_1a1e.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(run_1a1e.run(args), 1)
        with patch.object(run_1a1e.subprocess, "run") as process:
            with self.assertRaises(FileExistsError):
                run_1a1e.run(args)
        process.assert_not_called()

    def test_defaults_and_optional_input_paths_are_forwarded_without_reading(self):
        args = self.args("--ligand", str(self.root / "ligand.mol2"),
                         "--reference", str(self.root / "reference.mol2"))
        argv = run_1a1e.command(args, 10, self.root / "output")
        self.assertNotIn("--mmff", argv)
        self.assertNotIn("--write-sdf", argv)
        self.assertEqual(argv[argv.index("--fast-prefilter") + 1], "none")
        self.assertEqual(argv[argv.index("--crop-margin") + 1], "off")
        self.assertEqual(argv[argv.index("--ligand") + 1], str(self.root / "ligand.mol2"))
        self.assertEqual(argv[argv.index("--reference") + 1], str(self.root / "reference.mol2"))

    def test_code_hashes_include_untracked_sources_and_exclude_scientific_inputs(self):
        directory = self.root / "sources"
        directory.mkdir()
        for name in ("driver.cpp", "api.hpp", "CMakeLists.txt", "run.py", "library.mol2", "README.md"):
            (directory / name).write_text("synthetic " + name + "\n")
        result = run_1a1e.code_provenance(self.executable, directory)
        self.assertEqual(set(result["sources"]), {"driver.cpp", "api.hpp", "CMakeLists.txt", "run.py"})
        self.assertEqual(result["executable"]["sha256"],
                         hashlib.sha256(self.executable.read_bytes()).hexdigest())
        original = result["sources"]["driver.cpp"]["sha256"]
        (directory / "driver.cpp").write_text("synthetic updated source\n")
        self.assertNotEqual(original,
                            run_1a1e.code_provenance(self.executable, directory)["sources"]["driver.cpp"]["sha256"])


if __name__ == "__main__":
    unittest.main()
