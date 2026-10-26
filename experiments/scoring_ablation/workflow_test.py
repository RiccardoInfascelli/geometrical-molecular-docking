"""Test runner safety with synthetic files and mock executables, without RDKit."""

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parent


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-", dir=ROOT / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.complex = self.directory / "1a1e"
        self.complex.mkdir()
        for suffix in ("protein.pdb", "pocket.pdb", "ligand.mol2"):
            (self.complex / ("1a1e_" + suffix)).write_text("synthetic input; not a molecule\n")
        self.build = self.directory / "build"
        (self.build / "engine").mkdir(parents=True)
        self.executable(self.build / "engine" / "vinardo_rescore_experiment", """
            import csv, json, os, pathlib, sys
            output = pathlib.Path(sys.argv[sys.argv.index('--joint-output') + 1])
            output.mkdir()
            mode = os.environ.get('ABLATION_TEST_MODE', 'ok')
            with (output / 'protocol.csv').open('w', newline='') as target:
                csv.writer(target).writerows([
                    ['key', 'value'], ['pose_pool', 'fast_top_n' if mode == 'prefilter' else 'all_clash_survivors'],
                    ['fast_top_n', '1' if mode == 'prefilter' else 'none'],
                    ['input_survivors', '2'], ['exported_poses', '2']])
            with (output / 'poses.csv').open('w', newline='') as target:
                writer = csv.writer(target)
                writer.writerow(['record_index', 'pose_id'])
                writer.writerow([1, 0])
                if mode != 'truncated': writer.writerow([1, 0 if mode == 'duplicate' else 1])
            (output / 'arguments.json').write_text(json.dumps(sys.argv[1:]))
        """)
        self.executable(self.build / "scoring_ablation", """
            import json, pathlib, sys
            output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
            output.mkdir()
            (output / 'arguments.json').write_text(json.dumps(sys.argv[1:]))
        """)

    @staticmethod
    def executable(path, body):
        path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body).lstrip())
        path.chmod(0o700)  # Only mock files created by this test.

    def run_workflow(self, mode="ok"):
        environment = os.environ.copy()
        environment["ABLATION_TEST_MODE"] = mode
        return subprocess.run(
            ["bash", str(ROOT / "run_1a1e.sh"), str(self.complex),
             str(self.directory / "output"), str(self.build)],
            env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )

    def test_complete_pool_and_reference(self):
        import json
        original = {path.name: path.read_bytes() for path in self.complex.iterdir()}
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.directory / "output"
        generation = json.loads((output / "generation" / "arguments.json").read_text())
        analysis = json.loads((output / "ablation" / "arguments.json").read_text())
        self.assertNotIn("--prefilter", generation)
        self.assertEqual(analysis[analysis.index("--pool-size") + 1], "2")
        self.assertEqual(analysis[analysis.index("--pool-kind") + 1], "all-survivors")
        self.assertEqual(analysis[analysis.index("--reference") + 1],
                         str(self.complex / "1a1e_ligand.mol2"))
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.complex.iterdir()})
        self.assertTrue((output / "commands.txt").is_file())

    def test_existing_output_is_not_overwritten(self):
        output = self.directory / "output"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep original\n")
        result = self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "keep original\n")
        self.assertFalse((output / "generation").exists())

    def test_truncated_pool_is_rejected(self):
        result = self.run_workflow("truncated")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.directory / "output" / "ablation").exists())

    def test_prefilter_is_not_misreported_as_full_pool(self):
        result = self.run_workflow("prefilter")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.directory / "output" / "ablation").exists())

    def test_duplicate_ids_are_rejected(self):
        result = self.run_workflow("duplicate")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.directory / "output" / "ablation").exists())


if __name__ == "__main__":
    unittest.main()
