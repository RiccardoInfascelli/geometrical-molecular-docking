"""Synthetic, CPU-only checks. No docking executable or scientific input is read."""

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_benchmark as benchmark


def write_table(path, fields, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(output, statuses=("ok", "error", "no_poses"), score="1.5", summary=True,
            threads=1, openmp=True):
    output.mkdir(parents=True)
    records, poses, timings = [], [], []
    for index, status in enumerate(statuses, start=1):
        records.append({"record_index": index, "molecule_id": f"id-{index}",
                        "status": status, "exported_poses": int(status == "ok"),
                        "error": "synthetic malformed molecule" if status == "error" else ""})
        if status == "ok":
            poses.append({"record_index": index, "molecule_id": f"id-{index}", "rank": 1,
                          "alignment_index": 12, "fast_weighted_score": score,
                          "r00": "1", "tx": "0.25"})
        if status != "error":
            timings.append({"record_index": index, "read_parse_s": "0.1", "search_completed_s": "0.5",
                            "query_preparation_s": "0.2", "gpu_search_s": "0.3",
                            "query_threads_requested": threads, "query_threads_used": threads})
    write_table(output / "records.csv", list(records[0]), records)
    write_table(output / "poses.csv", ["record_index", "molecule_id", "rank", "alignment_index",
                "fast_weighted_score", "r00", "tx"], poses)
    write_table(output / "timings.csv", ["record_index", "read_parse_s", "search_completed_s",
                "query_preparation_s", "gpu_search_s", "query_threads_requested", "query_threads_used"], timings)
    if summary:
        errors = statuses.count("error")
        values = {"status": "complete_with_errors" if errors else "complete",
                  "processed": len(statuses), "successful_records": len(statuses) - errors,
                  "failed_records": errors, "startup_s": "2", "wall_before_summary_s": "9",
                  "query_threads_requested": threads, "query_openmp_available": int(openmp),
                  "query_parallel_min_triplets": 2048}
        write_table(output / "summary.csv", ["key", "value"],
                    [{"key": key, "value": value} for key, value in values.items()])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, *extra):
        return benchmark.arguments(["--protein", str(self.root / "protein.pdb"),
            "--pocket", str(self.root / "pocket.pdb"), "--ligands", str(self.root / "library.mol2"),
            "--executable", str(self.root / "gpu_docking"), "--output", str(self.root / "campaign"),
            *extra])

    def create_input_placeholders(self):
        for name in ("protein.pdb", "pocket.pdb", "library.mol2", "gpu_docking"):
            (self.root / name).write_text("synthetic test placeholder\n")
        (self.root / "gpu_docking").chmod(0o700)

    def test_defaults_and_baseline_are_unambiguous(self):
        args = self.args()
        self.assertEqual(args.limit, 1000)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.threads, [1, 2, 4, 8])
        self.assertEqual(args.ranking, list(benchmark.RANKINGS))
        self.assertEqual(args.crop_margin, "off")
        self.assertEqual(self.args("--threads", "8", "2", "2").threads, [1, 2, 8])

    def test_commands_keep_scientific_parameters_fixed(self):
        args = self.args("--query-method", "optimized", "--crop-margin", "8")
        for ranking in benchmark.RANKINGS:
            command = benchmark.command(args, ranking, "cuda", self.root / "new")
            self.assertIn("--continue-on-error", command)
            self.assertIn("--timings", command)
            for flag, value in (("--cutoff", "24"), ("--kabsch", "quaternion"),
                                ("--query-method", "optimized"), ("--top-k", "10"),
                                ("--diagnostics", "off")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual("--fast-prefilter" in command, ranking != "fast")
            self.assertEqual("--rescoring-backend" in command, ranking != "fast")
            if ranking == "hybrid":
                self.assertEqual(command[command.index("--hybrid-lambda") + 1], "0.3")
                self.assertEqual(command[command.index("--vinardo-alpha") + 1], "0.5")
        command = benchmark.command(self.args("--top-k", "25"), "fast", "not-applicable", self.root / "new")
        self.assertEqual(command[command.index("--top-k") + 1], "25")

    def test_environment_disables_inherited_crop_and_profiling(self):
        for legacy_value in ("1", "0"):
            with self.subTest(legacy_value=legacy_value):
                inherited = {"DOCKING_AREA_CROP_MARGIN_A": "99", "DOCKING_PROFILE_STAGES": "1",
                             "DOCKING_STAGE_TIMINGS": legacy_value,
                             "DOCKING_KABSCH_TIMINGS": legacy_value, "CUDA_VISIBLE_DEVICES": "0"}
                with patch.dict(os.environ, inherited, clear=True):
                    environment = benchmark.run_environment(self.args(), 4)
                self.assertNotIn("DOCKING_AREA_CROP_MARGIN_A", environment)
                self.assertEqual(environment["DOCKING_QUERY_THREADS"], "4")
                self.assertEqual(environment["OMP_DYNAMIC"], "FALSE")
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(environment["DOCKING_PROFILE_STAGES"], "0")
                for key in benchmark.PRESENCE_ENABLED_TIMERS:
                    self.assertNotIn(key, environment)

    def test_completed_errors_count_successes_and_timing_rows(self):
        output = self.root / "docking"
        fixture(output)
        result = benchmark.inspect_output(output, 1, 3, 10.0)
        self.assertEqual(result["status"], "complete_with_errors")
        self.assertEqual(result["completed_records"], 3)
        self.assertEqual(result["successful_records"], 2)
        self.assertEqual(result["failed_records"], 1)
        self.assertEqual(result["records_with_poses"], 1)
        self.assertEqual(result["successful_records_per_s"], 0.2)
        self.assertEqual(result["completed_records_per_s"], 0.3)
        self.assertEqual(result["startup_s"], 2.0)
        self.assertEqual(result["online_wall_s"], 7.0)
        self.assertEqual(result["successful_online_records_per_s"], 2 / 7)
        self.assertEqual(result["timings"]["search_completed_s"],
                         {"sum_s": 1.0, "mean_s": 0.5, "count": 2})
        self.assertNotIn("query_threads_used", result["timings"])
        self.assertEqual(result["query_thread_use"]["counts"], {1: 2})

    def test_failfast_partial_output_never_claims_requested_throughput(self):
        output = self.root / "partial"
        fixture(output, statuses=("ok",), summary=False)
        result = benchmark.inspect_output(output, 1, 1000, 10.0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["completed_records"], 1)
        self.assertIsNone(result["successful_records_per_s"])
        self.assertIsNone(result["completed_records_per_s"])
        self.assertIsNone(result["online_wall_s"])
        self.assertIsNone(result["successful_online_records_per_s"])
        self.assertTrue(result["issues"])

    def test_online_throughput_requires_a_positive_valid_interval(self):
        output = self.root / "online-interval"
        fixture(output)
        fields, rows = benchmark.read_table(output / "summary.csv")
        for value, expected_status in (("2", "complete_with_errors"), ("1", "incomplete")):
            with self.subTest(wall_before_summary_s=value):
                for row in rows:
                    if row["key"] == "wall_before_summary_s":
                        row["value"] = value
                write_table(output / "summary.csv", fields, rows)
                result = benchmark.inspect_output(output, 1, 3, 10.0)
                self.assertEqual(result["status"], expected_status)
                self.assertIsNone(result["successful_online_records_per_s"])

    def test_short_input_or_wrong_exit_code_is_incomplete(self):
        output = self.root / "short"
        fixture(output)
        for code, limit in ((0, 3), (1, 1000)):
            with self.subTest(code=code, limit=limit):
                result = benchmark.inspect_output(output, code, limit, 10.0)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["successful_records_per_s"])

    def test_all_error_records_are_completed_without_inventing_timing_values(self):
        output = self.root / "errors"
        fixture(output, statuses=("error", "error"))
        result = benchmark.inspect_output(output, 1, 2, 10.0)
        self.assertEqual(result["status"], "complete_with_errors")
        self.assertEqual(result["successful_records_per_s"], 0.0)
        self.assertEqual(result["timings"]["query_preparation_s"],
                         {"sum_s": 0, "mean_s": None, "count": 0})

    def test_completed_error_summary_must_agree_with_full_record_table(self):
        output = self.root / "bad-summary"
        fixture(output)
        fields, rows = benchmark.read_table(output / "summary.csv")
        for row in rows:
            if row["key"] == "failed_records":
                row["value"] = "0"
        write_table(output / "summary.csv", fields, rows)
        result = benchmark.inspect_output(output, 1, 3, 10.0)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["successful_records_per_s"])
        self.assertIn("failed_records", result["issues"][0])

    def test_nested_timing_columns_are_never_added_to_wall_or_each_other(self):
        output = self.root / "nested-timings"
        fixture(output)
        fields, rows = benchmark.read_table(output / "timings.csv")
        for row in rows:
            row["top_k_selection_s"], row["hybrid_normalization_s"] = "0.4", "0.1"
        write_table(output / "timings.csv", fields + ["top_k_selection_s", "hybrid_normalization_s"], rows)
        result = benchmark.inspect_output(output, 1, 3, 10.0)
        result.update(ranking="hybrid", backend="cpu", threads=1, comparison={"equal": True})
        group = benchmark.aggregate([result])[0]
        self.assertEqual(group["external_wall_s_mean"], 10.0)
        self.assertEqual(group["timing_column_sum_s_mean"]["search_completed_s"], 1.0)
        self.assertEqual(group["timing_column_sum_s_mean"]["query_preparation_s"], 0.4)
        self.assertEqual(group["timing_column_sum_s_mean"]["gpu_search_s"], 0.6)
        self.assertEqual(group["timing_column_sum_s_mean"]["top_k_selection_s"], 0.8)
        self.assertEqual(group["timing_column_sum_s_mean"]["hybrid_normalization_s"], 0.2)

    def test_actual_thread_fallback_is_reported_separately_from_timing(self):
        output = self.root / "fallback"
        fixture(output, threads=4)
        fields, rows = benchmark.read_table(output / "timings.csv")
        for row in rows:
            row["query_threads_used"] = "1"
        write_table(output / "timings.csv", fields, rows)
        result = benchmark.inspect_output(output, 1, 3, 10.0, requested_threads=4)
        self.assertEqual(result["status"], "complete_with_errors")
        self.assertTrue(result["query_thread_use"]["all_serial"])
        self.assertEqual(result["query_thread_use"]["parallel_records"], 0)
        self.assertNotIn("query_threads_requested", result["timings"])

    def test_strict_order_scores_transforms_and_errors_are_all_compared(self):
        baseline, candidate = self.root / "baseline", self.root / "candidate"
        fixture(baseline)
        fixture(candidate, score="1.50000001")
        self.assertFalse(benchmark.compare_tables(baseline, candidate)["equal"])
        self.assertTrue(benchmark.compare_tables(baseline, candidate, 1e-6)["equal"])
        fields, rows = benchmark.read_table(candidate / "poses.csv")
        rows[0]["tx"] = "0.3"
        write_table(candidate / "poses.csv", fields, rows)
        self.assertFalse(benchmark.compare_tables(baseline, candidate, 1e-6)["equal"])
        rows[0]["tx"] = "0.25"
        rows[0]["alignment_index"] = "13"
        write_table(candidate / "poses.csv", fields, rows)
        self.assertFalse(benchmark.compare_tables(baseline, candidate, 100)["equal"])
        fields, rows = benchmark.read_table(candidate / "records.csv")
        rows[1]["error"] = "different synthetic error"
        write_table(candidate / "records.csv", fields, rows)
        self.assertTrue(any(":error:" in item for item in
                            benchmark.compare_tables(baseline, candidate, 100)["differences"]))
        write_table(candidate / "records.csv", fields, list(reversed(rows)))
        self.assertFalse(benchmark.compare_tables(baseline, candidate, 100)["equal"])

    def test_full_campaign_uses_mocked_processes_and_fast_has_one_backend(self):
        self.create_input_placeholders()
        args = self.args("--limit", "3", "--ranking", "fast", "hybrid", "--threads", "2",
                         "--backend", "cpu", "cuda", "--repeats", "1")

        def fake_run(argv, **kwargs):
            self.assertEqual(kwargs["env"]["DOCKING_QUERY_THREADS"],
                             kwargs["env"]["OMP_NUM_THREADS"])
            fixture(Path(argv[4]), threads=int(kwargs["env"]["DOCKING_QUERY_THREADS"]))
            return subprocess.CompletedProcess(argv, 1)

        with patch.object(benchmark, "campaign_metadata", return_value={}), \
                patch.object(benchmark.subprocess, "run", side_effect=fake_run) as process:
            self.assertEqual(benchmark.run_campaign(args), 0)
        self.assertEqual(process.call_count, 6)  # Fast 2 + Hybrid (CPU/CUDA) 4.
        runs = json.loads((args.output / "runs.json").read_text())
        self.assertEqual(len(runs), 6)
        self.assertEqual({run["backend"] for run in runs if run["ranking"] == "fast"}, {"not-applicable"})
        self.assertTrue(all(run["comparison"]["equal"] for run in runs))
        self.assertTrue((args.output / runs[0]["directory"] / "stderr.log").is_file())
        with self.assertRaises(FileExistsError), patch.object(benchmark.subprocess, "run") as process:
            benchmark.run_campaign(args)
        process.assert_not_called()

    def test_campaign_stops_on_incomplete_process_and_keeps_evidence(self):
        self.create_input_placeholders()
        args = self.args("--limit", "3", "--ranking", "fast", "--threads", "2", "--repeats", "1")

        def fake_run(argv, **kwargs):
            fixture(Path(argv[4]), statuses=("ok",), summary=False)
            kwargs["stderr"].write("synthetic startup/runtime failure\n")
            return subprocess.CompletedProcess(argv, 1)

        with patch.object(benchmark, "campaign_metadata", return_value={}), \
                patch.object(benchmark.subprocess, "run", side_effect=fake_run) as process:
            self.assertEqual(benchmark.run_campaign(args), 1)
        self.assertEqual(process.call_count, 1)
        summary = json.loads((args.output / "summary.json").read_text())
        self.assertEqual(summary[0]["completed_repeats"], 0)
        self.assertIsNone(summary[0]["external_wall_speedup_vs_threads_1"])

    def test_full_strict_matrix_keeps_ranking_backend_error_patterns_separate(self):
        self.create_input_placeholders()
        args = self.args("--limit", "3", "--threads", "2", "--repeats", "2")

        def fake_run(argv, **kwargs):
            ranking = argv[argv.index("--ranking") + 1]
            backend = argv[argv.index("--rescoring-backend") + 1] if "--rescoring-backend" in argv else "none"
            statuses = {
                "fast": ("ok", "error", "no_poses"),
                "weighted-interaction": ("ok", "error", "error"),
                "vinardo-rep50": ("ok", "no_triplets", "no_poses"),
                "hybrid": ("error", "ok", "error" if backend == "cuda" else "no_poses"),
            }[ranking]
            score = str(benchmark.RANKINGS.index(ranking) + 1) + (".2" if backend == "cuda" else ".1")
            for key in benchmark.PRESENCE_ENABLED_TIMERS:
                self.assertNotIn(key, kwargs["env"])
            fixture(Path(argv[4]), statuses=statuses, score=score,
                    threads=int(kwargs["env"]["DOCKING_QUERY_THREADS"]))
            return subprocess.CompletedProcess(argv, int("error" in statuses))

        with patch.object(benchmark, "campaign_metadata", return_value={}), \
                patch.object(benchmark.subprocess, "run", side_effect=fake_run) as process, \
                patch.dict(os.environ, {key: "0" for key in benchmark.PRESENCE_ENABLED_TIMERS}), \
                patch("builtins.print"):
            self.assertEqual(benchmark.run_campaign(args), 0)
        self.assertEqual(process.call_count, 28)  # Seven ranking/backend groups, two threads/repeats.
        runs = json.loads((args.output / "runs.json").read_text())
        self.assertTrue(all(run["comparison"]["equal"] for run in runs))
        self.assertEqual({run["failed_records"] for run in runs}, {0, 1, 2})
        for run in runs:
            self.assertEqual(run["baseline_directory"],
                f"{run['ranking']}/{run['backend']}/threads-1/repeat-1/docking")

    def test_parallel_campaign_rejects_serial_build_after_first_baseline(self):
        self.create_input_placeholders()
        args = self.args("--limit", "3", "--ranking", "fast", "--threads", "2", "--repeats", "1")

        def fake_run(argv, **kwargs):
            fixture(Path(argv[4]), openmp=False)
            return subprocess.CompletedProcess(argv, 1)

        with patch.object(benchmark, "campaign_metadata", return_value={}), \
                patch.object(benchmark.subprocess, "run", side_effect=fake_run) as process:
            self.assertEqual(benchmark.run_campaign(args), 1)
        self.assertEqual(process.call_count, 1)
        runs = json.loads((args.output / "runs.json").read_text())
        self.assertIn("ENABLE_OPENMP=ON", runs[0]["issues"][0])

    def test_optional_hardware_metadata_failure_is_saved(self):
        with patch.object(benchmark.subprocess, "run", side_effect=FileNotFoundError("synthetic missing tool")):
            result = benchmark.metadata_command(["nvidia-smi"])
        self.assertEqual(result["command"], ["nvidia-smi"])
        self.assertIn("synthetic missing tool", result["error"])


if __name__ == "__main__":
    unittest.main()
