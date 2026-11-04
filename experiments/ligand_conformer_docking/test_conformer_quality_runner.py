"""Mocked runner checks and one explicitly enabled, tiny real-executable smoke."""

import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_conformer_quality as runner


def write_csv(path, fields, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def engine_outputs(directory, status="ok", success="0"):
    directory.mkdir()
    write_csv(directory / "summary.csv", ["status", "success_lt_2", "seed", "error"],
              [{"status": status, "success_lt_2": success, "seed": "20261027", "error": ""}])
    write_csv(directory / "budgets.csv", ["budget", "available", "status"],
              [{"budget": str(value), "available": "1", "status": status} for value in runner.BUDGETS])
    write_csv(directory / "conformers.csv", ["conformer_id"], [])
    write_csv(directory / "atom_mapping.csv", ["atom_index"], [])
    (directory / "protocol.json").write_text(json.dumps({"version": 1, "status": status}))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.dataset = self.root / "synthetic-inputs"
        self.dataset.mkdir()
        self.list = self.root / "cases.txt"
        self.list.write_text("# Explicit synthetic list\naaaa\nbbbb\n")
        for case_id in ("aaaa", "bbbb"):
            directory = self.dataset / case_id
            directory.mkdir()
            (directory / (case_id + "_ligand.mol2")).write_text("synthetic placeholder; never parsed\n")
        self.executable = self.root / "quality"
        self.executable.write_text("synthetic code placeholder, never executed\n")
        self.executable.chmod(0o700)
        self.output = self.root / "batch"

    def args(self, resume=False, extra=()):
        return runner.arguments(["--complex-list", str(self.list), "--dataset-root", str(self.dataset),
            "--executable", str(self.executable), "--resume" if resume else "--output", str(self.output), *extra])

    def executor(self, statuses=None):
        statuses = statuses or {"aaaa": "ok", "bbbb": "incomplete"}

        def execute(argv, **kwargs):
            ligand = Path(argv[argv.index("--ligand") + 1])
            output = Path(argv[argv.index("--output") + 1])
            self.assertFalse(output.exists())
            self.assertEqual(argv[argv.index("--seed") + 1], "20261027")
            self.assertEqual(argv[argv.index("--max-conformers") + 1], "200")
            case_id = ligand.parent.name
            engine_outputs(output, statuses[case_id])
            return subprocess.CompletedProcess(argv, 1 if statuses[case_id] == "error" else 0)
        return execute

    def test_once_per_case_aggregates_and_resume_skips_unsuccessful_and_incomplete(self):
        with patch.object(runner.subprocess, "run", side_effect=self.executor()) as process:
            self.assertEqual(runner.run(self.args()), 0)
        self.assertEqual(process.call_count, 2)
        _, rows = runner.read_csv(self.output / "summary.csv")
        self.assertEqual([row["case_id"] for row in rows], ["aaaa", "bbbb"])
        self.assertEqual(rows[0]["success_lt_2"], "0")
        _, budgets = runner.read_csv(self.output / "budgets.csv")
        self.assertEqual([row["case_id"] for row in budgets], ["aaaa"] * 7 + ["bbbb"] * 7)
        with patch.object(runner.subprocess, "run") as process:
            self.assertEqual(runner.run(self.args(resume=True)), 0)
        process.assert_not_called()

    def test_terminal_stereo_rejection_and_error_are_retained_and_not_retried(self):
        statuses = {"aaaa": "stereo_ambiguous", "bbbb": "error"}
        with patch.object(runner.subprocess, "run", side_effect=self.executor(statuses)):
            self.assertEqual(runner.run(self.args()), 1)
        _, failures = runner.read_csv(self.output / "failures.csv")
        self.assertEqual([row["status"] for row in failures], ["stereo_ambiguous", "error"])
        with patch.object(runner.subprocess, "run") as process:
            self.assertEqual(runner.run(self.args(resume=True)), 1)
        process.assert_not_called()

    def test_interrupted_attempt_is_preserved_and_resume_uses_new_directory(self):
        with patch.object(runner.subprocess, "run", side_effect=KeyboardInterrupt):
            self.assertEqual(runner.run(self.args(extra=("--limit", "1"))), 130)
        case_dir = runner.case_directory(self.output, 1, "aaaa")
        previous = (case_dir / "attempt-001/status.json").read_bytes()
        with patch.object(runner.subprocess, "run", side_effect=self.executor()) as process:
            self.assertEqual(runner.run(self.args(resume=True, extra=("--limit", "1"))), 0)
        self.assertEqual(process.call_count, 1)
        self.assertEqual((case_dir / "attempt-001/status.json").read_bytes(), previous)
        self.assertTrue((case_dir / "attempt-002/result/summary.csv").is_file())

    def test_protocol_changes_are_rejected_before_any_execution(self):
        with patch.object(runner.subprocess, "run", side_effect=self.executor()):
            self.assertEqual(runner.run(self.args()), 0)
        for extra in (("--seed", "9"), ("--max-conformers", "100"),
                      ("--allow-undefined-stereo",), ("--no-coordinates",)):
            with self.subTest(extra=extra), patch.object(runner.subprocess, "run") as process:
                with self.assertRaisesRegex(ValueError, "Resume protocol differs"):
                    runner.run(self.args(resume=True, extra=extra))
                process.assert_not_called()
        self.executable.write_text("changed synthetic executable\n")
        with patch.object(runner.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "Resume protocol differs"):
                runner.run(self.args(resume=True))
            process.assert_not_called()

    def test_coordinate_output_option_is_forwarded_and_recorded(self):
        args = self.args(extra=("--no-coordinates",))
        cases = runner.read_cases(args.complex_list, args.dataset_root)
        self.assertFalse(runner.expected_protocol(args, cases)["coordinates_export"])
        self.assertIn("--no-coordinates", runner.command(args, cases[0], self.root / "result"))
        defaults = self.args()
        self.assertTrue(runner.expected_protocol(defaults, cases)["coordinates_export"])
        self.assertNotIn("--no-coordinates", runner.command(defaults, cases[0], self.root / "result"))

    def test_legacy_coordinate_default_is_accepted_without_rewriting_manifest(self):
        with patch.object(runner.subprocess, "run", side_effect=self.executor()):
            self.assertEqual(runner.run(self.args()), 0)
        path = self.output / "manifest.json"
        manifest = runner.read_json(path)
        del manifest["protocol"]["coordinates_export"]
        path.write_text(json.dumps(manifest))
        original = path.read_bytes()
        with patch.object(runner.subprocess, "run") as process:
            self.assertEqual(runner.run(self.args(resume=True)), 0)
        process.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_truncated_terminal_budgets_are_preserved_and_retried_in_new_attempt(self):
        with patch.object(runner.subprocess, "run", side_effect=self.executor()):
            runner.run(self.args())
        case_dir = runner.case_directory(self.output, 1, "aaaa")
        old = case_dir / "attempt-001/result/budgets.csv"
        write_csv(old, ["budget", "available", "status"], [{"budget": "1", "available": "1", "status": "ok"}])
        old_bytes = old.read_bytes()
        with patch.object(runner.subprocess, "run", side_effect=self.executor()) as process:
            self.assertEqual(runner.run(self.args(resume=True)), 0)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(old.read_bytes(), old_bytes)
        self.assertTrue((case_dir / "attempt-002/result/summary.csv").is_file())

    def test_intact_engine_outputs_recover_interrupted_wrapper_without_inventing_exit_code(self):
        def interrupt_after_output(argv, **kwargs):
            self.executor()(argv, **kwargs)
            raise KeyboardInterrupt

        extra = ("--limit", "1")
        with patch.object(runner.subprocess, "run", side_effect=interrupt_after_output):
            self.assertEqual(runner.run(self.args(extra=extra)), 130)
        with patch.object(runner.subprocess, "run") as process:
            self.assertEqual(runner.run(self.args(resume=True, extra=extra)), 0)
        process.assert_not_called()
        status = runner.read_json(runner.case_directory(self.output, 1, "aaaa") / "attempt-001/status.json")
        self.assertEqual(status["phase"], "complete")
        self.assertIsNone(status["returncode"])
        self.assertTrue(status["recovered_from_outputs"])

    def test_list_order_and_targeted_input_stat_are_checked_on_resume(self):
        with patch.object(runner.subprocess, "run", side_effect=self.executor()):
            runner.run(self.args())
        self.list.write_text("bbbb\naaaa\n")
        with patch.object(runner.subprocess, "run") as process:
            with self.assertRaises(ValueError):
                runner.run(self.args(resume=True))
            process.assert_not_called()
        self.list.write_text("aaaa\nbbbb\n")
        (self.dataset / "aaaa/aaaa_ligand.mol2").write_text("changed synthetic placeholder\n")
        with patch.object(runner.subprocess, "run") as process:
            with self.assertRaises(ValueError):
                runner.run(self.args(resume=True))
            process.assert_not_called()

    def test_missing_outputs_are_recorded_and_next_case_still_runs(self):
        calls = []

        def execute(argv, **kwargs):
            calls.append(argv)
            if len(calls) == 1:
                kwargs["stderr"].write("synthetic failure before outputs\n")
                return subprocess.CompletedProcess(argv, 7)
            return self.executor()(argv, **kwargs)

        with patch.object(runner.subprocess, "run", side_effect=execute):
            self.assertEqual(runner.run(self.args()), 1)
        _, summaries = runner.read_csv(self.output / "summary.csv")
        self.assertEqual([row["case_id"] for row in summaries], ["bbbb"])
        _, failures = runner.read_csv(self.output / "failures.csv")
        self.assertEqual(failures[0]["case_id"], "aaaa")
        self.assertEqual(failures[0]["returncode"], "7")

    def test_explicit_directories_duplicate_ids_and_stereo_passthrough(self):
        self.list.write_text(str(self.dataset / "aaaa") + "\n")
        cases = runner.read_cases(self.list, self.dataset)
        args = self.args(extra=("--allow-undefined-stereo",))
        self.assertIn("--allow-undefined-stereo", runner.command(args, cases[0], self.output / "new"))
        self.list.write_text("aaaa\n" + str(self.dataset / "aaaa") + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate case ID"):
            runner.read_cases(self.list, self.dataset)


@unittest.skipUnless(os.environ.get("CONFORMER_QUALITY_EXECUTABLE"),
                     "Set CONFORMER_QUALITY_EXECUTABLE to enable the synthetic runner smoke")
class RealRunnerSmoke(unittest.TestCase):
    def test_csv_stereo_rejection_and_resume(self):
        executable = Path(os.environ["CONFORMER_QUALITY_EXECUTABLE"]).resolve()
        deadline = time.monotonic() + 20

        def invoke(argv, expected_code=0):
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, "Synthetic smoke exceeded its 20-second budget")
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, start_new_session=True) as process:
                try:
                    stdout, stderr = process.communicate(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    self.fail("Synthetic runner smoke timed out; its subprocess group was stopped")
            self.assertEqual(process.returncode, expected_code, stdout + stderr)
            return stdout + stderr

        def checked_rows(path):
            _, rows = runner.read_csv(path)
            return rows

        # The old three-neighbour C.3 fixture relied on an unverified implicit H.
        # Four explicit, different substituents make this stereo expectation
        # independent of MOL2 hydrogen inference. CO2 needs no implicit H either.
        fixtures = {
            "achiral": """@<TRIPOS>MOLECULE
carbon_dioxide
3 2 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 C1 0 0 0 C.1 1 CO2 0
2 O1 -1.16 0 0 O.2 1 CO2 0
3 O2 1.16 0 0 O.2 1 CO2 0
@<TRIPOS>BOND
1 1 2 2
2 1 3 2
@<TRIPOS>SUBSTRUCTURE
1 CO2 1
""",
            "ambiguous": """@<TRIPOS>MOLECULE
four_explicit_substituents
5 4 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 C1 0 0 0 C.3 1 CHI 0
2 F1 1 1 1 F 1 CHI 0
3 CL1 1 -1 -1 Cl 1 CHI 0
4 BR1 -1 1 -1 Br 1 CHI 0
5 I1 -1 -1 1 I 1 CHI 0
@<TRIPOS>BOND
1 1 2 1
2 1 3 1
3 1 4 1
4 1 5 1
@<TRIPOS>SUBSTRUCTURE
1 CHI 1
""",
        }
        with tempfile.TemporaryDirectory(prefix="conformer-quality-runner-smoke-") as temporary:
            root = Path(temporary)
            dataset = root / "synthetic-inputs"
            inputs = {}
            for case_id, block in fixtures.items():
                directory = dataset / case_id
                directory.mkdir(parents=True)
                ligand = directory / (case_id + "_ligand.mol2")
                ligand.write_text(block)
                inputs[ligand] = ligand.read_bytes()
            complex_list = root / "synthetic-cases.txt"
            complex_list.write_text("achiral\nambiguous\n")
            output = root / "new-output"
            argv = [sys.executable, "-B", str(Path(runner.__file__).resolve()),
                    "--complex-list", str(complex_list), "--dataset-root", str(dataset),
                    "--executable", str(executable), "--max-conformers", "2", "--seed", "7",
                    "--output", str(output)]
            invoke(argv)

            summaries = checked_rows(output / "summary.csv")
            self.assertEqual([row["case_id"] for row in summaries], ["achiral", "ambiguous"])
            self.assertEqual([row["status"] for row in summaries], ["ok", "stereo_ambiguous"])
            self.assertEqual([row["generated"] for row in summaries], ["2", "0"])
            self.assertGreaterEqual(int(summaries[1]["undefined_stereo"]), 1)
            self.assertEqual(summaries[1]["best_rmsd"], "")
            self.assertEqual(summaries[1]["success_lt_2"], "")
            self.assertEqual(len(checked_rows(output / "budgets.csv")), 14)
            failures = checked_rows(output / "failures.csv")
            self.assertEqual([(row["case_id"], row["status"]) for row in failures],
                             [("ambiguous", "stereo_ambiguous")])

            first = output / "cases/000001-achiral/attempt-001/result"
            second = output / "cases/000002-ambiguous/attempt-001/result"
            conformers = checked_rows(first / "conformers.csv")
            self.assertEqual([row["sequence_index"] for row in conformers], ["1", "2"])
            self.assertTrue(all(math.isfinite(float(row["aligned_heavy_rmsd"])) for row in conformers))
            self.assertEqual(len(checked_rows(first / "atom_mapping.csv")), 3)
            self.assertEqual(len(checked_rows(first / "conformer_coordinates.csv")), 6)
            self.assertEqual(checked_rows(second / "conformers.csv"), [])
            self.assertEqual(len(checked_rows(second / "atom_mapping.csv")), 5)
            budgets = checked_rows(first / "budgets.csv")
            self.assertEqual([int(row["budget"]) for row in budgets], list(runner.BUDGETS))
            self.assertEqual(budgets[0]["complete"], "1")
            self.assertEqual(budgets[0]["best_rmsd"], conformers[0]["aligned_heavy_rmsd"])
            self.assertEqual(budgets[1]["available"], "2")
            self.assertEqual(budgets[1]["status"], "not_requested")
            self.assertEqual(float(budgets[1]["best_rmsd"]),
                             min(float(row["aligned_heavy_rmsd"]) for row in conformers))
            for directory, calls in ((first, 1), (second, 0)):
                protocol = runner.read_json(directory / "protocol.json")
                self.assertEqual(protocol["generation_calls"], calls)
                self.assertEqual(protocol["max_conformers"], 2)
                self.assertEqual(protocol["seed"], 7)
                self.assertFalse(protocol["mmff"])
                self.assertFalse(protocol["native_coordinates_in_generation"])
                self.assertFalse(protocol["native_stereo_in_generation"])
                self.assertFalse(protocol["allow_undefined_stereo"])
                self.assertIn("Horn quaternion", protocol["alignment"])

            snapshot = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
            resume = argv.copy()
            resume[resume.index("--output")] = "--resume"
            stdout = invoke(resume)
            self.assertEqual(stdout.count("reuse terminal result"), 2)
            self.assertEqual({path for path in output.rglob("*") if path.is_file()}, set(snapshot))
            # Only the batch progress timestamp may change; results, attempts,
            # aggregates and the manifest must remain byte-for-byte identical.
            for path, content in snapshot.items():
                if path != output / "status.json":
                    self.assertEqual(path.read_bytes(), content, str(path))
            self.assertFalse(list((output / "cases").glob("*/attempt-002")))
            self.assertFalse(runner.read_json(output / "manifest.json")["protocol"]["allow_undefined_stereo"])
            incompatible = invoke(resume + ["--allow-undefined-stereo"], expected_code=1)
            self.assertIn("Resume protocol differs", incompatible)
            for path, content in snapshot.items():
                if path != output / "status.json":
                    self.assertEqual(path.read_bytes(), content, str(path))

            # Explicit graph-only generation is a separate protocol and output.
            # Its poses remain labelled by the unresolved stereo count; no
            # correct/native stereoisomer is assumed from the observed RMSD.
            ambiguous_list = root / "ambiguous-only.txt"
            ambiguous_list.write_text("ambiguous\n")
            allowed_output = root / "allowed-output"
            allowed = argv.copy()
            allowed[allowed.index("--complex-list") + 1] = str(ambiguous_list)
            allowed[allowed.index("--output") + 1] = str(allowed_output)
            allowed.append("--allow-undefined-stereo")
            invoke(allowed)
            allowed_summary = checked_rows(allowed_output / "summary.csv")
            self.assertEqual(len(allowed_summary), 1)
            self.assertEqual(allowed_summary[0]["case_id"], "ambiguous")
            self.assertEqual(allowed_summary[0]["status"], "ok")
            self.assertEqual(allowed_summary[0]["generated"], "2")
            self.assertGreaterEqual(int(allowed_summary[0]["undefined_stereo"]), 1)
            self.assertTrue(runner.read_json(allowed_output / "manifest.json")["protocol"]["allow_undefined_stereo"])
            allowed_case = allowed_output / "cases/000001-ambiguous/attempt-001/result"
            allowed_protocol = runner.read_json(allowed_case / "protocol.json")
            self.assertTrue(allowed_protocol["allow_undefined_stereo"])
            self.assertFalse(allowed_protocol["native_coordinates_in_generation"])
            self.assertFalse(allowed_protocol["native_stereo_in_generation"])
            self.assertFalse(allowed_protocol["mmff"])
            self.assertEqual(allowed_protocol["generation_calls"], 1)
            self.assertEqual([row["sequence_index"] for row in checked_rows(allowed_case / "conformers.csv")],
                             ["1", "2"])
            allowed_snapshot = {path: path.read_bytes() for path in allowed_output.rglob("*") if path.is_file()}
            allowed_resume = allowed.copy()
            allowed_resume[allowed_resume.index("--output")] = "--resume"
            self.assertEqual(invoke(allowed_resume).count("reuse terminal result"), 1)
            self.assertEqual({path for path in allowed_output.rglob("*") if path.is_file()}, set(allowed_snapshot))
            for path, content in allowed_snapshot.items():
                if path != allowed_output / "status.json":
                    self.assertEqual(path.read_bytes(), content, str(path))
            self.assertFalse(list((allowed_output / "cases").glob("*/attempt-002")))

            coordinate_output = root / "without-coordinates"
            without_coordinates = allowed.copy()
            without_coordinates[without_coordinates.index("--output") + 1] = str(coordinate_output)
            without_coordinates.append("--no-coordinates")
            invoke(without_coordinates)
            coordinate_case = coordinate_output / "cases/000001-ambiguous/attempt-001/result"
            self.assertFalse((coordinate_case / "conformer_coordinates.csv").exists())
            for name in ("conformers.csv", "budgets.csv", "atom_mapping.csv"):
                self.assertEqual((coordinate_case / name).read_bytes(), (allowed_case / name).read_bytes())
            coordinate_summary = checked_rows(coordinate_output / "summary.csv")[0]
            for key, value in allowed_summary[0].items():
                if not key.endswith("_s"):
                    self.assertEqual(coordinate_summary[key], value, key)
            self.assertFalse(runner.read_json(coordinate_case / "protocol.json")["coordinates_export"])
            coordinate_resume = without_coordinates.copy()
            coordinate_resume[coordinate_resume.index("--output")] = "--resume"
            self.assertEqual(invoke(coordinate_resume).count("reuse terminal result"), 1)
            self.assertIn("Resume protocol differs",
                          invoke(allowed_resume + ["--no-coordinates"], expected_code=1))
            for path, content in inputs.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
