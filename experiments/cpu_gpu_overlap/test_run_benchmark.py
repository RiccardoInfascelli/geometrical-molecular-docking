"""Mocked runner tests: no CUDA, molecular input or real subprocess is executed."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("overlap_runner", Path(__file__).with_name("run_benchmark.py"))
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def write_table(path, fields, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(output, mode="off", threads=1, limit=3, changed_score=False, fake_overlap=False,
            ranking="fast", alpha=None, hybrid_lambda=None, fast_prefilter="none",
            rescoring_only=False, no_poses=False, pose_output="topk", crop_margin="off"):
    output.mkdir()
    records, poses, timings, timeline = [], [], [], []
    for index in range(1, limit + 1):
        error = index == 3
        records.append({"record_index": index, "molecule_id": f"id-{index}",
            "status": "error" if error else "no_poses" if no_poses else "ok", "survivors": 0 if error or no_poses else 2,
            "candidate_matches": 0 if error else 10, "alignments": 0 if error else 12,
            "exported_poses": 0 if error or no_poses else 2, "sdf": "",
            "error": "parsing/sanitization: synthetic invalid record" if error else ""})
        if ranking != "fast":
            records[-1]["candidate_pool"] = 0 if error or no_poses else 2
        if not error:
            for rank in (() if no_poses else (1, 2)):
                poses.append({"record_index": index, "molecule_id": f"id-{index}", "rank": rank,
                    "alignment_index": rank + 10, "fast_weighted_score": 99 if changed_score else 3 - rank,
                    "r00": 1, "tx": 0.25})
                if ranking != "fast":
                    poses[-1].update(pose_id=rank - 1, weighted_interaction_score=3 - rank,
                        vinardo_like_score=rank, vinardo_adjusted_score=rank,
                        hybrid_score=(1 if rank == 1 else -1) + int(changed_score))
            timings.append({"record_index": index, "query_preparation_s": 0.2,
                "search_completed_s": 0.5, "gpu_search_s": 0.3,
                "query_threads_requested": threads, "query_threads_used": threads})
            if ranking != "fast" and mode != "legacy":
                timings[-1].update(rescoring_completed_s=0.1, dock_completed_s=0.5,
                    weighted_interaction_scoring_s=0.04, vinardo_scoring_s=0.04,
                    hybrid_normalization_s=0.01)
        if error:
            query_start = query_end = search_start = search_end = ""
        elif index == 1:
            query_start, query_end, search_start, search_end = 0, 0.2, 0.2, 0.8
        elif mode == "on" and not fake_overlap:
            query_start, query_end, search_start, search_end = 0.3, 0.5, 0.8, 1.2
            if rescoring_only:
                query_start, query_end, search_start, search_end = 0.65, 0.75, 0.8, 1.2
        else:
            query_start, query_end, search_start, search_end = 0.8, 1, 1, 1.4
        timeline.append({"record_index": index, "status": records[-1]["status"],
            "query_start_s": query_start, "query_end_s": query_end,
            "search_start_s": search_start, "search_end_s": search_end,
            "consumer_wait_s": 0.01})
        if ranking != "fast":
            timeline[-1].update(dock_start_s=search_start, dock_end_s=search_end,
                rescoring_start_s="" if error else search_end - 0.2, rescoring_end_s=search_end)
            timeline[-1]["search_end_s"] = "" if error else search_end - 0.2
    write_table(output / "records.csv", list(records[0]), records)
    write_table(output / "poses.csv", list(poses[0]) if poses else
                ["record_index", "molecule_id", "rank", "alignment_index", "fast_weighted_score", "r00", "tx"], poses)
    write_table(output / "timings.csv", list(timings[0]), timings)
    write_table(output / "overlap_timeline.csv", list(timeline[0]), timeline)
    errors = int(limit >= 3)
    overlap = mode == "on" and limit >= 2
    summary = {"status": "complete_with_errors" if errors else "complete", "processed": limit,
        "successful_records": limit - errors, "failed_records": errors, "startup_s": 2,
        "wall_before_summary_s": 9, "online_wall_s": 1.5, "cpu_gpu_overlap": mode,
        "queue_capacity": 1, "consumer_wait_s": 0.03, "producer_queue_wait_s": 0.1,
        "query_search_host_overlap_s": 0.2 if overlap else 0, "query_search_overlap_records": int(overlap),
        "query_threads_requested": threads, "query_openmp_available": 1,
        "query_parallel_min_triplets": 2048, "kabsch_method_effective": "quaternion",
        "query_method_effective": "reference", "ranking_method": "vinardo" if ranking == "vinardo-rep50" else ranking,
        "sdf_export": 0, "rescoring_backend": "cuda", "fast_prefilter": fast_prefilter,
        "interaction_region": "pocket", "vinardo_alpha": alpha if alpha is not None else (0.5 if ranking == "vinardo-rep50" else 1),
        "hybrid_lambda": 0.3 if hybrid_lambda is None else hybrid_lambda,
        "ranking_requested": ranking, "index_policy": "shared_index_only", "index_cutoff_A": 24,
        "target_clearance_enabled": 0, "min_target_clearance_A": "off",
        "chemical_area_crop_margin_A": crop_margin, "timings_enabled": 1, "diagnostics_effective": "off",
        "pose_output": pose_output, "pose_detail": "full",
        "overlap_trace": 0, "nvtx_available": 1}
    if ranking != "fast":
        for phase in ("search", "rescoring", "dock"):
            interval, count = BENCHMARK.observed_overlap(
                [{k: str(v) for k, v in row.items()} for row in timeline], phase)
            summary[f"query_{phase}_host_overlap_s"] = interval
            summary[f"query_{phase}_overlap_records"] = count
    write_table(output / "summary.csv", ["key", "value"], [{"key": k, "value": v} for k, v in summary.items()])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.complex = self.root / "1a1e"
        self.complex.mkdir()
        for name in ("1a1e_protein.pdb", "1a1e_pocket.pdb"):
            (self.complex / name).write_text("placeholder, never parsed\n")
        self.executable = self.root / "gpu_docking"
        self.executable.write_text("not executed\n")
        self.executable.chmod(0o700)
        self.library = self.root / "library.mol2"
        self.library.write_text("placeholder, never parsed\n")
        (self.root / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nENABLE_CUDA:BOOL=ON\nENABLE_OPENMP:BOOL=ON\n")

    def args(self, *extra):
        return BENCHMARK.arguments(["--complex", str(self.complex), "--ligands", str(self.library),
            "--executable", str(self.executable), "--output", str(self.root / "campaign"), *extra])

    def test_fixed_protocol_and_smoke_scope(self):
        args = self.args()
        self.assertEqual((args.limit, args.repeats, args.threads), (1000, 3, [1, 8]))
        argv = BENCHMARK.command(args, "on", self.root / "docking")
        for key, value in (("--ranking", "fast"), ("--kabsch", "quaternion"),
                           ("--query-method", "reference"), ("--cutoff", "24"),
                           ("--cpu-gpu-overlap", "on"), ("--pose-output", "topk")):
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn("--no-sdf", argv)
        smoke = self.args("--smoke-only")
        self.assertEqual((smoke.limit, smoke.repeats), (1, 1))
        self.assertNotIn("--cpu-gpu-overlap", BENCHMARK.command(smoke, "legacy", self.root / "legacy"))
        self.assertEqual(BENCHMARK.command(smoke, "on", self.root / "on")[-3:], ["all", "--pose-detail", "full"])
        self.assertEqual(self.args("--repeats", "2").repeats, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--repeats", "0")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--smoke-only", "--limit", "1000")

    def test_direct_online_wall_and_expected_record_errors(self):
        args = self.args("--limit", "3")
        output = self.root / "docking"
        fixture(output, mode="on", threads=8)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 8, "on")
        self.assertEqual(result["status"], "complete_with_errors")
        self.assertEqual(result["online_wall_s"], 1.5)  # Not the old 9-2 proxy.
        self.assertAlmostEqual(result["successful_online_records_per_s"], 2 / 1.5)
        self.assertEqual((result["failed_records"], result["successful_records"]), (1, 2))
        self.assertTrue(result["host_overlap_observed"])
        self.assertEqual(result["timings"]["gpu_search_s"]["sum_s"], 0.6)

    def test_fake_overlap_and_wrong_exit_code_never_produce_valid_rates(self):
        args = self.args("--limit", "3")
        output = self.root / "docking"
        fixture(output, mode="on", fake_overlap=True)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "on")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("timeline", result["issues"][-1])
        self.assertIsNone(result["successful_online_records_per_s"])
        result = BENCHMARK.inspect_output(args, output, 0, 10, 1, "on")
        self.assertEqual(result["status"], "incomplete")

    def test_timeline_compares_different_records_and_rejects_impossible_dependencies(self):
        timeline = [{"record_index": "1", "query_start_s": "0", "query_end_s": "1",
                     "search_start_s": "1", "search_end_s": "4"},
                    {"record_index": "2", "query_start_s": "2", "query_end_s": "3",
                     "search_start_s": "4", "search_end_s": "5"}]
        self.assertEqual(BENCHMARK.observed_overlap(timeline), (1, 1))
        timeline[1]["search_start_s"] = "2.5"
        with self.assertRaisesRegex(ValueError, "before its queries"):
            BENCHMARK.observed_overlap(timeline)

    def mock_campaign(self, args, mutate=False, no_overlap=False):
        commands = []

        def complete(argv, **kwargs):
            mode = argv[argv.index("--cpu-gpu-overlap") + 1] if "--cpu-gpu-overlap" in argv else "legacy"
            threads = int(kwargs["env"]["DOCKING_QUERY_THREADS"])
            commands.append((threads, mode))
            value = lambda flag, fallback=None: argv[argv.index(flag) + 1] if flag in argv else fallback
            fixture(Path(argv[4]), mode=mode, threads=threads, limit=args.limit,
                    changed_score=mutate and mode == "on", fake_overlap=no_overlap, ranking=value("--ranking"),
                    alpha=float(value("--vinardo-alpha")) if "--vinardo-alpha" in argv else None,
                    hybrid_lambda=float(value("--hybrid-lambda")) if "--hybrid-lambda" in argv else None,
                    fast_prefilter=value("--fast-prefilter", "none"),
                    pose_output=value("--pose-output", "topk"), crop_margin=args.crop_margin)
            return subprocess.CompletedProcess(argv, int(args.limit >= 3))

        with patch.object(BENCHMARK, "check_resources", return_value={}), \
                patch.object(BENCHMARK, "build_metadata", return_value={}), \
                patch.object(BENCHMARK.subprocess, "run", side_effect=complete), \
                contextlib.redirect_stdout(io.StringIO()):
            code = BENCHMARK.run_campaign(args)
        return code, commands

    def test_all_configurations_alternate_and_compare_identical_science(self):
        args = self.args("--limit", "3")
        code, commands = self.mock_campaign(args)
        self.assertEqual(code, 0)
        self.assertEqual(commands, [(1, "off"), (1, "on"), (8, "off"), (8, "on"),
                                    (1, "on"), (8, "off"), (8, "on"), (1, "off"),
                                    (8, "off"), (8, "on"), (1, "off"), (1, "on")])
        _, rows = BENCHMARK.EXISTING.read_table(args.output / "runs.csv")
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["online_speedup_vs_same_thread_OFF"] == "1.0" for r in rows))
        _, summary = BENCHMARK.EXISTING.read_table(args.output / "summary.csv")
        self.assertTrue(all(r["valid_repeats"] == "3" for r in summary))
        self.assertTrue(json.loads((args.output / "status.json").read_text())["host_overlap_demonstrated"])
        with self.assertRaises(FileExistsError):
            BENCHMARK.run_campaign(args)

    def test_single_repeat_supports_threads_one_four_eight_for_each_ranking(self):
        for ranking in BENCHMARK.RANKINGS:
            with self.subTest(ranking=ranking):
                args = self.args("--ranking", ranking, "--limit", "3", "--threads", "1", "4", "8",
                                 "--repeats", "1")
                args.output = self.root / f"single-repeat-{ranking}"
                code, commands = self.mock_campaign(args)
                self.assertEqual(code, 0)
                self.assertEqual(commands, [(1, "off"), (1, "on"), (4, "off"), (4, "on"),
                                            (8, "off"), (8, "on")])
                _, rows = BENCHMARK.EXISTING.read_table(args.output / "summary.csv")
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(row["valid_repeats"] == "1" for row in rows))
                self.assertTrue(all(row["repetition_scope"] == "single_repeat_exploratory" for row in rows))
                self.assertTrue(all(float(row["online_speedup_vs_same_thread_OFF_median"]) == 1
                                    for row in rows))

    def test_effective_protocol_records_explicit_scoring_parameters(self):
        args = self.args("--ranking", "hybrid", "--vinardo-alpha", "0.5", "--hybrid-lambda", "0.3",
                         "--threads", "1", "4", "8", "--repeats", "1")
        with patch.object(BENCHMARK.EXISTING, "metadata_command", return_value={"returncode": 0, "stdout": "mock"}):
            metadata = BENCHMARK.build_metadata(args)
        protocol = metadata["scientific_protocol"]
        self.assertEqual((protocol["kabsch_method"], protocol["query_method"], protocol["rescoring_backend"]),
                         ("quaternion", "reference", "cuda"))
        self.assertEqual((protocol["vinardo_alpha"], protocol["hybrid_lambda"], protocol["fast_prefilter"]),
                         (0.5, 0.3, "none"))
        self.assertEqual(metadata["repetition_scope"], "single_repeat_exploratory")
        self.assertIn("does not", metadata["repetition_note"])
        for ranking, alpha in (("vinardo", "1.0"), ("vinardo-rep50", "0.5"), ("hybrid", "1.0")):
            argv = BENCHMARK.command(self.args("--ranking", ranking), "on", self.root / ranking)
            self.assertEqual(argv[argv.index("--vinardo-alpha") + 1], alpha)
            if ranking == "hybrid":
                self.assertEqual(argv[argv.index("--hybrid-lambda") + 1], "0.3")

    def test_actual_protocol_mismatches_fail_including_legacy_smoke(self):
        changes = (("kabsch_method_effective", "svd"), ("query_method_effective", "optimized"),
                   ("rescoring_backend", "cpu"), ("fast_prefilter", "500"),
                   ("vinardo_alpha", "1"), ("hybrid_lambda", "0.5"),
                   ("index_policy", "fallback"), ("index_cutoff_A", "20"),
                   ("target_clearance_enabled", "1"), ("chemical_area_crop_margin_A", "8"),
                   ("query_threads_requested", "8"), ("pose_output", "none"),
                   ("diagnostics_effective", "stages"), ("timings_enabled", "0"))
        for mode in ("off", "on", "legacy"):
            for key, value in changes:
                with self.subTest(mode=mode, key=key):
                    args = self.args("--ranking", "hybrid", "--vinardo-alpha", "0.5", "--limit", "3")
                    args.smoke_only = mode == "legacy"
                    output = self.root / f"mismatch-{mode}-{key}"
                    fixture(output, mode=mode, ranking="hybrid", alpha=0.5,
                            pose_output="all" if args.smoke_only else "topk")
                    fields, rows = BENCHMARK.EXISTING.read_table(output / "summary.csv")
                    next(row for row in rows if row["key"] == key)["value"] = value
                    write_table(output / "summary.csv", fields, rows)
                    result = BENCHMARK.inspect_output(args, output, 1, 10, 1, mode)
                    self.assertEqual(result["status"], "incomplete")
                    self.assertIsNone(result["successful_online_records_per_s"])

    def test_scientific_mismatch_stops_before_additional_configurations(self):
        args = self.args("--limit", "3")
        code, commands = self.mock_campaign(args, mutate=True)
        self.assertEqual(code, 1)
        self.assertEqual(commands, [(1, "off"), (1, "on")])
        _, rows = BENCHMARK.EXISTING.read_table(args.output / "runs.csv")
        self.assertEqual(rows[-1]["online_speedup_vs_same_thread_OFF"], "")
        self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "stopped_on_failure_or_mismatch")

    def test_smoke_includes_legacy_and_every_survivor_without_claiming_overlap(self):
        args = self.args("--smoke-only")
        code, commands = self.mock_campaign(args)
        self.assertEqual(code, 0)
        self.assertEqual(commands, [(1, "legacy"), (1, "off"), (1, "on"), (8, "off"), (8, "on")])
        status = json.loads((args.output / "status.json").read_text())
        self.assertFalse(status["host_overlap_demonstrated"])

    def test_multi_record_full_pool_smoke_preserves_errors_and_observes_host_overlap(self):
        args = self.args("--smoke-only", "--limit", "3")
        code, commands = self.mock_campaign(args)
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 5)
        _, results = BENCHMARK.EXISTING.read_table(args.output / "runs.csv")
        self.assertTrue(all(r["processed"] == "3" and r["errors"] == "1" for r in results))
        self.assertTrue(all(r["correct"] == "True" for r in results))
        self.assertTrue(json.loads((args.output / "status.json").read_text())["host_overlap_demonstrated"])

    def test_environment_disables_unrelated_timing_and_thread_policies(self):
        with patch.dict(os.environ, {"DOCKING_STAGE_TIMINGS": "0", "OMP_PLACES": "threads",
                                     "OPENBLAS_NUM_THREADS": "32"}, clear=True):
            env = BENCHMARK.run_environment(self.args(), 8)
        self.assertNotIn("DOCKING_STAGE_TIMINGS", env)
        self.assertNotIn("OMP_PLACES", env)
        self.assertEqual(env["DOCKING_QUERY_THREADS"], "8")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_scoring_options_keep_one_cuda_protocol_and_default_fast(self):
        for ranking in BENCHMARK.RANKINGS:
            with self.subTest(ranking=ranking):
                args = self.args("--ranking", ranking)
                argv = BENCHMARK.command(args, "on", self.root / ranking)
                self.assertEqual(argv[argv.index("--ranking") + 1], ranking)
                if ranking == "fast":
                    self.assertNotIn("--rescoring-backend", argv)
                    self.assertNotIn("--fast-prefilter", argv)
                else:
                    self.assertEqual(argv[argv.index("--rescoring-backend") + 1], "cuda")
                    self.assertEqual(argv[argv.index("--fast-prefilter") + 1], "none")
        args = self.args("--ranking", "hybrid", "--vinardo-alpha", "0.5",
                         "--hybrid-lambda", "0.3", "--fast-prefilter", "500")
        argv = BENCHMARK.command(args, "off", self.root / "hybrid")
        for flag, expected in (("--vinardo-alpha", "0.5"), ("--hybrid-lambda", "0.3"),
                               ("--fast-prefilter", "500"), ("--interaction-region", "pocket")):
            self.assertEqual(argv[argv.index(flag) + 1], expected)
        for extra in (("--backend", "cpu"), ("--ranking", "fast", "--fast-prefilter", "500"),
                      ("--ranking", "weighted-interaction", "--vinardo-alpha", "0.5"),
                      ("--ranking", "vinardo-rep50", "--vinardo-alpha", "1"),
                      ("--ranking", "vinardo", "--hybrid-lambda", "0.3"),
                      ("--ranking", "hybrid", "--hybrid-lambda", "1.1")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_every_cuda_ranking_compares_full_pool_legacy_off_on(self):
        for ranking in BENCHMARK.RANKINGS:
            with self.subTest(ranking=ranking):
                args = self.args("--ranking", ranking, "--smoke-only", "--limit", "3")
                args.output = self.root / f"campaign-{ranking}"
                code, commands = self.mock_campaign(args)
                self.assertEqual(code, 0)
                self.assertEqual(len(commands), 5)
                _, results = BENCHMARK.EXISTING.read_table(args.output / "runs.csv")
                self.assertTrue(all(r["correct"] == "True" and r["errors"] == "1" for r in results))
                if ranking != "fast":
                    self.assertTrue(all(r["ranking"] == ranking and r["rescoring_backend"] == "cuda" for r in results))

    def test_rescoring_overlap_counts_even_without_search_intersection(self):
        args = self.args("--ranking", "hybrid", "--limit", "3", "--vinardo-alpha", "0.5")
        output = self.root / "hybrid-rescoring"
        fixture(output, mode="on", ranking="hybrid", alpha=0.5, rescoring_only=True)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "on")
        self.assertNotEqual(result["status"], "incomplete", result["issues"])
        self.assertEqual(result["query_search_host_overlap_s"], 0)
        self.assertAlmostEqual(result["query_rescoring_host_overlap_s"], 0.1)
        self.assertTrue(result["host_overlap_observed"])
        self.assertAlmostEqual(result["timings"]["rescoring_completed_s"]["sum_s"], 0.2)

    def test_no_pose_input_is_complete_but_not_a_successful_smoke(self):
        args = self.args("--ranking", "weighted-interaction", "--limit", "3")
        output = self.root / "empty-pool"
        fixture(output, ranking=args.ranking, no_poses=True)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "off")
        self.assertEqual(result["status"], "complete_with_errors", result["issues"])
        self.assertEqual(result["records_with_poses"], 0)
        args.smoke_only = True
        fields, rows = BENCHMARK.EXISTING.read_table(output / "summary.csv")
        next(row for row in rows if row["key"] == "pose_output")["value"] = "all"
        write_table(output / "summary.csv", fields, rows)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "off")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("did not generate", result["issues"][-1])

    def test_extended_protocol_and_interval_errors_fail_closed(self):
        args = self.args("--ranking", "hybrid", "--limit", "3")
        output = self.root / "bad-hybrid"
        fixture(output, mode="on", ranking="hybrid", hybrid_lambda=0.5)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "on")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("lambda", result["issues"][-1])
        output = self.root / "bad-timeline"
        fixture(output, mode="on", ranking="hybrid")
        fields, rows = BENCHMARK.EXISTING.read_table(output / "overlap_timeline.csv")
        rows[0]["rescoring_start_s"] = "0.1"
        write_table(output / "overlap_timeline.csv", fields, rows)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "on")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("dependencies", result["issues"][-1])

    def test_extended_score_identity_and_order_mismatches_are_detected(self):
        left, right = self.root / "left", self.root / "right"
        fixture(left, ranking="hybrid")
        fixture(right, ranking="hybrid")
        fields, original = BENCHMARK.EXISTING.read_table(right / "poses.csv")
        for field, changed in (("weighted_interaction_score", "8"), ("vinardo_adjusted_score", "8"),
                               ("hybrid_score", "0"), ("pose_id", "99"), ("tx", "7")):
            with self.subTest(field=field):
                rows = [dict(row) for row in original]
                rows[0][field] = changed
                write_table(right / "poses.csv", fields, rows)
                self.assertFalse(BENCHMARK.EXISTING.compare_tables(left, right)["equal"])
        write_table(right / "poses.csv", fields, list(reversed(original)))
        self.assertFalse(BENCHMARK.EXISTING.compare_tables(left, right)["equal"])

    def test_selected_overlap_modes_preserve_baseline_without_claiming_overlap(self):
        args = self.args("--limit", "3", "--threads", "1", "--overlap", "off")
        code, commands = self.mock_campaign(args)
        self.assertEqual(code, 0)
        self.assertEqual(commands, [(1, "off")] * 3)
        self.assertFalse(json.loads((args.output / "status.json").read_text())["host_overlap_demonstrated"])

    def test_nonfast_without_observed_overlap_suppresses_on_speedup(self):
        args = self.args("--ranking", "vinardo-rep50", "--limit", "3", "--threads", "1")
        code, commands = self.mock_campaign(args, no_overlap=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 6)
        status = json.loads((args.output / "status.json").read_text())
        self.assertEqual(status["status"], "complete_no_host_overlap")
        _, rows = BENCHMARK.EXISTING.read_table(args.output / "runs.csv")
        self.assertTrue(all(row["correct"] == "True" for row in rows))
        self.assertTrue(all(row["online_speedup_vs_same_thread_OFF"] == "" for row in rows if row["overlap"] == "on"))

    def test_empty_or_short_input_cannot_be_reported_as_completed_campaign(self):
        args = self.args("--ranking", "hybrid", "--limit", "3")
        output = self.root / "empty-input"
        fixture(output, ranking="hybrid")
        for filename in ("records.csv", "poses.csv", "timings.csv", "overlap_timeline.csv"):
            fields, _ = BENCHMARK.EXISTING.read_table(output / filename)
            write_table(output / filename, fields, [])
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "off")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["completed_records"], 0)
        self.assertIsNone(result["successful_online_records_per_s"])

    def test_failed_completed_call_may_publish_only_its_dock_interval(self):
        args = self.args("--ranking", "hybrid", "--limit", "3")
        output = self.root / "failed-call"
        fixture(output, ranking="hybrid")
        fields, rows = BENCHMARK.EXISTING.read_table(output / "overlap_timeline.csv")
        rows[-1].update(dock_start_s="1.45", dock_end_s="1.46")
        write_table(output / "overlap_timeline.csv", fields, rows)
        result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "off")
        self.assertEqual(result["status"], "complete_with_errors", result["issues"])

    def test_on_only_keeps_wall_and_throughput_medians_without_speedup(self):
        for ranking in ("fast", "hybrid"):
            with self.subTest(ranking=ranking):
                args = self.args("--ranking", ranking, "--limit", "3", "--threads", "1", "--overlap", "on")
                args.output = self.root / f"on-only-{ranking}"
                code, commands = self.mock_campaign(args)
                self.assertEqual(code, 0)
                self.assertEqual(commands, [(1, "on")] * 3)
                _, rows = BENCHMARK.EXISTING.read_table(args.output / "summary.csv")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["valid_repeats"], "3")
                self.assertEqual(float(rows[0]["online_wall_s_median"]), 1.5)
                self.assertAlmostEqual(float(rows[0]["successful_ligands_per_s_median"]), 2 / 1.5)
                self.assertNotEqual(rows[0]["query_preparation_work_s_median"], "")
                self.assertEqual(rows[0]["online_speedup_vs_same_thread_OFF_median"], "")

    def test_successful_nonfast_rows_require_all_completed_spans(self):
        args = self.args("--ranking", "hybrid", "--limit", "3")
        for removed in (("dock", "search", "rescoring"), ("search",), ("rescoring",)):
            with self.subTest(removed=removed):
                output = self.root / ("missing-" + "-".join(removed))
                fixture(output, ranking="hybrid")
                fields, rows = BENCHMARK.EXISTING.read_table(output / "overlap_timeline.csv")
                for phase in removed:
                    rows[0][phase + "_start_s"] = rows[0][phase + "_end_s"] = ""
                write_table(output / "overlap_timeline.csv", fields, rows)
                result = BENCHMARK.inspect_output(args, output, 1, 10, 1, "off")
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("Missing completed", result["issues"][-1])


if __name__ == "__main__":
    unittest.main()
