#!/usr/bin/env python3
"""Check clearance comparison parsing using synthetic logs, without docking inputs."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_clearance_pruning import (
    CSV_FIELDS, compare_rows, main as run_campaign, parse_output, pose_status, run_one,
)


FIELDS = (
    "clearance targets protein_triplets candidate_matches transforms survivors_5pct "
    "best_rmsd_5pct best_rmsd_10pct index_s search_s targets_removed "
    "protein_prep_s clearance_s ligand_prep_s filter_s"
).split()
HEADER = "\t".join(FIELDS)


def make_row(**changes):
    row = dict(zip(FIELDS, (
        "disabled", "100", "1000", "2000", "2400", "40", "1.2", "1.1",
        "0.5", "2.0", "0", "0.4", "0.1", "0.2", "0.01",
    )))
    row.update({key: str(value) for key, value in changes.items()})
    return row


def pruned_row(**changes):
    row = make_row(
        clearance="1.250000", targets=80, protein_triplets=500,
        candidate_matches=1000, transforms=1200, survivors_5pct=20,
        best_rmsd_5pct="1.4", best_rmsd_10pct="1.3", index_s="0.2",
        search_s="1.0", targets_removed=20, filter_s="0.015",
    )
    row.update({key: str(value) for key, value in changes.items()})
    return row


def row_text(row):
    return "\t".join(row[field] for field in FIELDS)


def log_text(*rows):
    return "\n".join((
        "=== CLEARANCE PRUNING EXPERIMENT ===", "Complex: synthetic",
        "Preparation is shared; its times are repeated in both rows.", HEADER,
        *(row_text(row) for row in rows), "",
    ))


class ParseOutputTests(unittest.TestCase):
    def test_complete_comparison_preserves_all_fields(self):
        baseline, pruned = make_row(), pruned_row()
        parsed = parse_output(log_text(baseline, pruned))
        self.assertEqual(parsed, {"disabled": baseline, "pruned": pruned})
        self.assertTrue(all(isinstance(value, str)
                            for row in parsed.values() for value in row.values()))

    def test_threshold_is_numeric_and_configurable(self):
        baseline, pruned = make_row(), pruned_row(clearance="0.750000")
        self.assertEqual(parse_output(log_text(baseline, pruned), threshold=0.75),
                         {"disabled": baseline, "pruned": pruned})

    def test_partial_output_keeps_completed_rows(self):
        baseline, pruned = make_row(), pruned_row()
        self.assertEqual(parse_output(log_text(baseline)), {"disabled": baseline})
        self.assertEqual(parse_output(log_text(pruned)), {"pruned": pruned})
        self.assertEqual(parse_output(log_text()), {})

    def test_nonfinite_rmsd_values_are_preserved(self):
        baseline = make_row(best_rmsd_5pct="inf", best_rmsd_10pct="nan")
        parsed = parse_output(log_text(baseline))
        self.assertEqual(parsed["disabled"]["best_rmsd_5pct"], "inf")
        self.assertEqual(parsed["disabled"]["best_rmsd_10pct"], "nan")

    def test_missing_header_is_rejected(self):
        for text in ("", "Complex: synthetic\n", row_text(make_row())):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_output(text)

    def test_malformed_rows_are_rejected(self):
        for row in (
            row_text(make_row()).rsplit("\t", 1)[0],
            row_text(make_row()) + "\textra",
            row_text(make_row(candidate_matches="many")),
            row_text(make_row(index_s="invalid")),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_output(HEADER + "\n" + row + "\n")

    def test_duplicate_rows_are_rejected(self):
        for row in (make_row(), pruned_row()):
            with self.subTest(clearance=row["clearance"]), self.assertRaises(ValueError):
                parse_output(log_text(row, row))

    def test_unknown_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_output(log_text(make_row(), pruned_row(clearance="1.5")))


class PoseStatusTests(unittest.TestCase):
    def test_statuses_do_not_turn_missing_or_nonfinite_rmsd_into_zero(self):
        self.assertEqual(pose_status(None), "missing")
        self.assertEqual(pose_status(make_row()), "ok")
        self.assertEqual(pose_status(make_row(survivors_5pct=0, best_rmsd_5pct="inf")),
                         "no_poses")
        for rmsd in ("nan", "inf", "-inf"):
            with self.subTest(rmsd=rmsd):
                self.assertEqual(pose_status(make_row(best_rmsd_5pct=rmsd)), "invalid_rmsd")

    def test_nonempty_5pct_set_requires_valid_rmsd_for_both_thresholds(self):
        for rmsd in ("nan", "inf", "-1"):
            with self.subTest(rmsd_10pct=rmsd):
                baseline = make_row(best_rmsd_10pct=rmsd)
                self.assertEqual(pose_status(baseline), "invalid_rmsd")
                self.assertIsNone(compare_rows(baseline, pruned_row())["best_rmsd_5pct_delta_A"])
        # With no 5% survivors, a finite diagnostic at 10% remains legitimate.
        self.assertEqual(pose_status(make_row(survivors_5pct=0, best_rmsd_5pct="inf",
                                             best_rmsd_10pct="1.8")), "no_poses")


class CompareRowsTests(unittest.TestCase):
    def test_reductions_and_speedup_have_explicit_direction(self):
        comparison = compare_rows(make_row(), pruned_row())
        self.assertAlmostEqual(comparison["targets_reduction_pct"], 20.0)
        for field in ("protein_triplets", "candidate_matches", "transforms", "survivors_5pct"):
            with self.subTest(field=field):
                self.assertAlmostEqual(comparison[field + "_reduction_pct"], 50.0)
        self.assertAlmostEqual(comparison["index_s_reduction_pct"], 60.0)
        self.assertAlmostEqual(comparison["search_s_reduction_pct"], 50.0)
        self.assertAlmostEqual(comparison["index_search_s_reduction_pct"], 52.0)
        self.assertAlmostEqual(comparison["index_search_speedup"], 2.5 / 1.2)
        # Positive delta means pruning made the best available RMSD worse.
        self.assertAlmostEqual(comparison["best_rmsd_5pct_delta_A"], 0.2)

    def test_zero_denominators_do_not_produce_false_reductions(self):
        baseline = make_row(
            protein_triplets=0, candidate_matches=0, transforms=0,
            survivors_5pct=0, best_rmsd_5pct="inf", index_s=0, search_s=0,
        )
        pruned = pruned_row(
            protein_triplets=0, candidate_matches=0, transforms=0,
            survivors_5pct=0, best_rmsd_5pct="inf", index_s=0, search_s=0,
        )
        comparison = compare_rows(baseline, pruned)
        for field in ("protein_triplets", "candidate_matches", "transforms",
                      "survivors_5pct", "index_s", "search_s", "index_search_s"):
            with self.subTest(field=field):
                self.assertIsNone(comparison[field + "_reduction_pct"])
        self.assertIsNone(comparison["index_search_speedup"])
        self.assertIsNone(comparison["best_rmsd_5pct_delta_A"])

    def test_rmsd_delta_requires_poses_and_finite_values_in_both_rows(self):
        for changes in ({"survivors_5pct": 0, "best_rmsd_5pct": "inf"},
                        {"best_rmsd_5pct": "nan"}, {"best_rmsd_5pct": "inf"}):
            for side in ("disabled", "pruned"):
                with self.subTest(changes=changes, side=side):
                    baseline = make_row(**changes) if side == "disabled" else make_row()
                    pruned = pruned_row(**changes) if side == "pruned" else pruned_row()
                    self.assertIsNone(compare_rows(baseline, pruned)["best_rmsd_5pct_delta_A"])

    def test_shared_preparation_times_must_agree(self):
        for field in ("protein_prep_s", "clearance_s", "ligand_prep_s"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_rows(make_row(), pruned_row(**{field: "9.0"}))

    def test_target_count_before_filtering_must_agree(self):
        with self.assertRaises(ValueError):
            compare_rows(make_row(), pruned_row(targets_removed=19))


class RunOneTests(unittest.TestCase):
    def run_mocked(self, stdout_text, *, returncode=0, timeout=False):
        stderr_text = "Synthetic process diagnostic.\n"

        def fake_run(command, *, stdout, stderr, timeout, check):
            self.assertFalse(check)
            self.assertEqual(command[-2:], ["--compare-clearance", "1.25"])
            stdout.write(stdout_text)
            stderr.write(stderr_text)
            if timeout_run:
                raise subprocess.TimeoutExpired(command, timeout)
            return subprocess.CompletedProcess(command, returncode)

        timeout_run = timeout
        with tempfile.TemporaryDirectory(prefix="clearance-test-") as directory:
            output = Path(directory)
            with patch("run_clearance_pruning.subprocess.run", side_effect=fake_run) as run:
                row = run_one(Path("/synthetic/docking"), Path("/synthetic/complex"),
                              1.25, output, 3.0)
            run.assert_called_once()
            self.assertEqual((output / row["stdout_log"]).read_text(), stdout_text)
            self.assertEqual((output / row["stderr_log"]).read_text(), stderr_text)
            return row

    def test_failed_exit_with_complete_output_does_not_compute_comparison(self):
        with patch("run_clearance_pruning.compare_rows") as compare:
            row = self.run_mocked(log_text(make_row(), pruned_row()), returncode=2)
        compare.assert_not_called()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["exit_code"], 2)
        self.assertEqual(row["baseline_status"], "ok")
        self.assertEqual(row["pruned_status"], "ok")
        self.assertEqual(row["baseline_candidate_matches"], "2000")
        self.assertNotIn("index_search_speedup", row)

    def test_missing_poses_and_invalid_rmsd_are_not_process_failures(self):
        variants = (
            (pruned_row(survivors_5pct=0, best_rmsd_5pct="inf"), "no_poses"),
            (pruned_row(best_rmsd_5pct="nan"), "invalid_rmsd"),
        )
        for pruned, status in variants:
            with self.subTest(status=status):
                row = self.run_mocked(log_text(make_row(), pruned))
                self.assertEqual(row["status"], status)
                self.assertEqual(row["exit_code"], 0)
                self.assertEqual(row["pruned_status"], status)
                self.assertIsNone(row["best_rmsd_5pct_delta_A"])
                self.assertEqual(row["pruned_best_rmsd_5pct"], pruned["best_rmsd_5pct"])

    def test_successful_exit_with_partial_output_is_invalid(self):
        row = self.run_mocked(log_text(make_row()))
        self.assertEqual(row["status"], "invalid_output")
        self.assertEqual(row["exit_code"], 0)
        self.assertEqual(row["baseline_status"], "ok")
        self.assertEqual(row["pruned_status"], "missing")
        self.assertNotIn("index_search_speedup", row)

    def test_timeout_preserves_partial_stdout_and_stderr(self):
        row = self.run_mocked(log_text(make_row()), timeout=True)
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["exit_code"], "")
        self.assertEqual(row["baseline_status"], "ok")
        self.assertEqual(row["pruned_status"], "missing")
        self.assertNotIn("index_search_speedup", row)

    def test_invalid_rmsd_is_not_hidden_by_other_variant_having_no_poses(self):
        for invalid_side in ("baseline", "pruned"):
            with self.subTest(invalid_side=invalid_side):
                empty = {"survivors_5pct": 0, "best_rmsd_5pct": "inf"}
                invalid = {"best_rmsd_10pct": "nan"}
                baseline = make_row(**(invalid if invalid_side == "baseline" else empty))
                pruned = pruned_row(**(invalid if invalid_side == "pruned" else empty))
                row = self.run_mocked(log_text(baseline, pruned))
                self.assertEqual(row["status"], "invalid_rmsd")
                self.assertEqual(row[invalid_side + "_status"], "invalid_rmsd")
                self.assertIsNone(row["best_rmsd_5pct_delta_A"])


class CampaignTests(unittest.TestCase):
    def test_csv_contains_every_case_and_existing_output_is_not_overwritten(self):
        def fake_run(command, *, stdout, stderr, timeout, check):
            self.assertFalse(check)
            invalid = Path(command[1]).name == "case_invalid"
            pruned = pruned_row(best_rmsd_10pct="nan") if invalid else pruned_row()
            stdout.write(log_text(make_row(), pruned))
            stderr.write("Synthetic diagnostics.\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory(prefix="clearance-campaign-test-") as directory:
            root = Path(directory)
            executable, ids, output = root / "fake-docking", root / "complexes.txt", root / "output"
            executable.write_text("Not an executable; subprocess.run is mocked.\n")
            ids.write_text("case_ok\ncase_invalid\n")
            args = ["run_clearance_pruning.py", "--executable", str(executable),
                    "--complex-root", str(root / "synthetic-inputs"),
                    "--complex-list", str(ids), "--output", str(output)]
            with patch("sys.argv", args), patch("run_clearance_pruning.os.access", return_value=True), \
                    patch("run_clearance_pruning.subprocess.run", side_effect=fake_run) as run, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                # Invalid numerical output must make the campaign return nonzero,
                # even when the experiment process itself returned zero.
                self.assertEqual(run_campaign(), 1)
                self.assertEqual(run.call_count, 2)
                with (output / "comparison.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
                    rows = list(reader)
                self.assertEqual([row["complex_id"] for row in rows], ["case_ok", "case_invalid"])
                self.assertEqual([row["status"] for row in rows], ["ok", "invalid_rmsd"])
                self.assertEqual(rows[1]["pruned_best_rmsd_10pct"], "nan")
                self.assertEqual(rows[1]["best_rmsd_5pct_delta_A"], "")
                self.assertTrue(all(None not in row and len(row) == len(CSV_FIELDS) for row in rows))
                preserved = {path.name: path.read_bytes() for path in output.iterdir()}
                with self.assertRaises(SystemExit) as refusal:
                    run_campaign()
                self.assertEqual(refusal.exception.code, 2)
                self.assertEqual(run.call_count, 2)
                self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, preserved)


if __name__ == "__main__":
    unittest.main()
