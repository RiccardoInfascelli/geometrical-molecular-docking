#!/usr/bin/env python3
"""Opt-in real CUDA checks for prepared OFF/ON docking and CUDA rescoring.

The configured complex must produce clash survivors and support Vinardo typing.
Only a temporary three-record library is used; each subprocess has a timeout.
"""

import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import gpu_docking_library_test as library_test


RANKINGS = ("fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid")
TRANSFORMS = tuple(f"r{row}{column}" for row in range(3) for column in range(3)) + ("tx", "ty", "tz")
INTEGER_POSE_FIELDS = {"rank", "alignment_index", "occupied_atoms", "pose_id"}
TIMELINE_FIELDS = ("record_index", "status", "query_start_s", "query_end_s",
                   "search_start_s", "search_end_s", "consumer_wait_s")
EXTENDED_SPANS = ("dock_start_s", "dock_end_s", "rescoring_start_s", "rescoring_end_s")
require = library_test.require


def close(actual, expected, field):
    # Same device/backend and input protocol, not a CPU-vs-CUDA comparison.
    absolute = 1e-6 if field in TRANSFORMS else 1e-9
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute),
            f"{field} differs: {actual:.17g} versus {expected:.17g}")


def compare_pose_rows(actual, expected):
    require(len(actual) == len(expected), "Ranked pose counts differ")
    for index, (left, right) in enumerate(zip(actual, expected), 1):
        require(set(left) == set(right), f"Pose fields differ at rank {index}")
        for field, value in right.items():
            if field in INTEGER_POSE_FIELDS or not value:
                require(left[field] == value, f"Pose {field} differs at rank {index}")
            else:
                close(float(left[field]), float(value), field)


def invoke(executable, protein, pocket, library, output, environment, ranking,
           mode, *, recover=True, quantity="all", cutoff="24", expected_error=True, top_k="10"):
    command = [str(executable), str(protein), str(pocket), str(library), str(output),
               "--no-sdf", "--timings", "--kabsch", "quaternion", "--query-method", "reference",
               "--diagnostics", "off", "--limit", "3", "--cutoff", cutoff,
               "--ranking", ranking, "--top-k", top_k]
    if quantity is not None:
        command.extend(("--pose-output", quantity, "--pose-detail", "full"))
    if mode is not None:
        command.extend(("--cpu-gpu-overlap", mode))
    if recover:
        command.append("--continue-on-error")
    if ranking != "fast":
        command.extend(("--rescoring-backend", "cuda", "--fast-prefilter", "none"))
    if ranking in ("weighted-interaction", "hybrid"):
        command.extend(("--interaction-region", "pocket"))
    if ranking == "hybrid":
        command.extend(("--hybrid-lambda", "0.3", "--vinardo-alpha", "0.5"))
    completed = subprocess.run(command, env=environment, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    expected = 1 if expected_error else 0
    require(completed.returncode == expected,
            f"{ranking}/{mode}: expected exit {expected}, got {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    require(output.is_dir(), f"Missing output for {ranking}/{mode}: {completed.stderr}")
    require(not any(path.suffix.lower() == ".sdf" for path in output.rglob("*")),
            f"--no-sdf created structural output for {ranking}/{mode}")
    return completed


def check_ranking(poses, ranking):
    field = {"fast": "fast_weighted_score", "weighted-interaction": "weighted_interaction_score",
             "vinardo": "vinardo_adjusted_score", "vinardo-rep50": "vinardo_adjusted_score",
             "hybrid": "hybrid_score"}[ranking]
    values = [float(row[field]) for row in poses]
    identity = "alignment_index" if ranking == "fast" else "pose_id"
    for left, right, a, b in zip(poses, poses[1:], values, values[1:]):
        require(a <= b if ranking.startswith("vinardo") else a >= b,
                f"{ranking} score order differs at rank {left['rank']}")
        if a == b:
            require(int(left[identity]) < int(right[identity]),
                    f"{ranking} tie-breaking changed at rank {left['rank']}")
    if ranking != "fast":
        require({int(row["pose_id"]) for row in poses} == set(range(len(poses))),
                "Full export lost or duplicated original survivor IDs")
    if ranking.startswith("vinardo") or ranking == "hybrid":
        alpha = 1.0 if ranking == "vinardo" else 0.5
        for row in poses:
            adjusted = float(row["vinardo_like_score"]) + (alpha - 1.0) * float(row["vinardo_repulsion"])
            close(float(row["vinardo_adjusted_score"]), adjusted, "Vinardo alpha")
    if ranking == "hybrid":
        # Independent check on every survivor, before any Top-K truncation.
        def zscores(values):
            mean = statistics.fmean(values)
            variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
            deviation = math.sqrt(variance)
            return [(value - mean) / deviation if deviation else 0.0 for value in values]
        weighted = zscores([float(row["weighted_interaction_score"]) for row in poses])
        vinardo = zscores([-float(row["vinardo_adjusted_score"]) for row in poses])
        for row, first, second in zip(poses, weighted, vinardo):
            close(float(row["hybrid_score"]), 0.3 * first + 0.7 * second, "Hybrid normalization")


def check_timeline(output, ranking, mode):
    fields = TIMELINE_FIELDS + (EXTENDED_SPANS if ranking != "fast" else ())
    rows = library_test.table(output / "overlap_timeline.csv", fields)
    require([int(row["record_index"]) for row in rows] == [1, 2, 3], "Timeline record order differs")
    for row in rows:
        require(math.isfinite(float(row["consumer_wait_s"])) and float(row["consumer_wait_s"]) >= 0,
                "Invalid consumer wait")
        if row["status"] == "error":
            require(not row["search_start_s"] and not row["search_end_s"],
                    "A failed parse reached GPU search")
            continue
        query_end, search_start, search_end = map(float, (
            row["query_end_s"], row["search_start_s"], row["search_end_s"]))
        require(0 <= float(row["query_start_s"]) <= query_end <= search_start <= search_end,
                "Search began before its packet was prepared")
        if ranking != "fast":
            dock_start, dock_end, rescoring_start, rescoring_end = map(float, (row[name] for name in EXTENDED_SPANS))
            require(0 <= dock_start <= search_start <= search_end <= rescoring_start <= rescoring_end <= dock_end,
                    "Completed geometry/rescoring stages lie outside their docking span")
    if mode == "off":
        for query in rows:
            if not query["query_start_s"]:
                continue
            for other in rows:
                if other["record_index"] == query["record_index"] or not other["search_start_s"]:
                    continue
                end = other["dock_end_s"] if ranking != "fast" else other["search_end_s"]
                begin = other["dock_start_s"] if ranking != "fast" else other["search_start_s"]
                require(min(float(query["query_end_s"]), float(end)) <=
                        max(float(query["query_start_s"]), float(begin)),
                        "OFF unexpectedly overlapped a query with another docking call")


def inspect(output, ranking, mode):
    completed = library_test.summary(output / "summary.csv")
    for key, expected in {"status": "complete_with_errors", "processed": "3", "successful_records": "2",
                          "failed_records": "1", "cpu_gpu_overlap": mode,
                          "ranking_requested": ranking, "index_policy": "shared_index_only"}.items():
        require(completed.get(key) == expected, f"{ranking}/{mode}: summary {key} differs")
    if ranking != "fast":
        require(completed.get("rescoring_backend") == "cuda" and completed.get("fast_prefilter") == "none",
                "The integration test did not exercise full-pool CUDA rescoring")
    if ranking.startswith("vinardo") or ranking == "hybrid":
        close(float(completed["vinardo_alpha"]), 1.0 if ranking == "vinardo" else 0.5,
              "Effective Vinardo alpha")
    if ranking == "hybrid":
        close(float(completed["hybrid_lambda"]), 0.3, "Effective Hybrid lambda")
    records, poses = library_test.inspect(output, (1, 2, 3), recover=True)
    for index in (1, 3):
        record = records[index - 1]
        require(len(poses[index]) == int(record["survivors"]), "Full export omitted clash survivors")
        if ranking != "fast":
            require(int(record["candidate_pool"]) == int(record["survivors"]),
                    "No-prefilter rescoring changed pool size")
        check_ranking(poses[index], ranking)
    require(poses[1] == poses[3], f"{ranking}: repeated input changed after invalid record")
    check_timeline(output, ranking, mode)
    return records, poses


def compare_scientific_records(actual, expected, *, compare_exported=True):
    require(len(actual) == len(expected), "Processed record counts differ")
    for left, right in zip(actual, expected):
        require(set(left) == set(right), "Record fields differ")
        for field in right:
            if not compare_exported and field == "exported_poses":
                continue
            if field == "diameter_A":
                close(float(left[field]), float(right[field]), field)
            else:
                require(left[field] == right[field], f"Record {right['record_index']} field {field} differs")


def main():
    require(len(sys.argv) == 3, "Usage: gpu_docking_overlap_test.py <gpu_docking> <complex_directory>")
    executable = Path(sys.argv[1]).resolve(strict=True)
    complex_directory = Path(sys.argv[2]).resolve(strict=True)
    identifier = complex_directory.name
    protein, pocket, ligand = [complex_directory / f"{identifier}_{suffix}"
                               for suffix in ("protein.pdb", "pocket.pdb", "ligand.mol2")]
    originals = {path: library_test.digest(path) for path in (protein, pocket, ligand)}
    valid = library_test.first_record(ligand)
    environment = os.environ.copy()
    for key in ("DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS", "DOCKING_AREA_CROP_MARGIN_A"):
        environment.pop(key, None)
    environment.update(DOCKING_PROFILE_STAGES="0", DOCKING_QUERY_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="gpu-docking-overlap-") as temporary:
        root = Path(temporary)
        library = root / "library.mol2"
        library.write_bytes(library_test.renamed_record(valid, library_test.IDS[0]) +
                            b"@<TRIPOS>MOLECULE\ninvalid_middle\n" +
                            library_test.renamed_record(valid, library_test.IDS[2]))
        full = {}
        reference_survivors = None
        for ranking in RANKINGS:
            for mode in ("off", "on"):
                output = root / f"{ranking}-{mode}"
                invoke(executable, protein, pocket, library, output, environment, ranking, mode)
                full[ranking, mode] = inspect(output, ranking, mode)
            off, on = full[ranking, "off"], full[ranking, "on"]
            compare_scientific_records(on[0], off[0])
            for index in (1, 3):
                compare_pose_rows(on[1][index], off[1][index])
                compare_pose_rows(on[1][index][:10], off[1][index][:10])
            # Every ranking must preserve the same geometric survivor identities.
            survivors = {row["alignment_index"]: {field: row[field] for field in
                         ("alignment_index", "occupied_atoms", "clash_fraction", "fast_weighted_score") + TRANSFORMS}
                         for row in off[1][1]}
            require(len(survivors) == len(off[1][1]), "Duplicate alignment identity in survivor export")
            if reference_survivors is None:
                reference_survivors = survivors
            else:
                require(set(survivors) == set(reference_survivors), f"{ranking}: survivor identity changed")
                for identity in survivors:
                    compare_pose_rows([survivors[identity]], [reference_survivors[identity]])

        # Preserve the pre-prefetch Fast path as a separate reference.
        legacy = root / "fast-legacy"
        invoke(executable, protein, pocket, library, legacy, environment, "fast", None,
               quantity=None, top_k="20")
        records, poses = library_test.inspect(legacy, (1, 2, 3), recover=True)
        compare_scientific_records(records, full["fast", "off"][0], compare_exported=False)
        for index in (1, 3):
            compare_pose_rows(poses[index], full["fast", "off"][1][index][:20])

        # Exercise actual CUDA Top-K selection/download against a full-pool rank.
        prefix = root / "hybrid-on-top10"
        invoke(executable, protein, pocket, library, prefix, environment, "hybrid", "on", quantity="topk")
        records, poses = library_test.inspect(prefix, (1, 2, 3), recover=True)
        compare_scientific_records(records, full["hybrid", "off"][0], compare_exported=False)
        for index in (1, 3):
            compare_pose_rows(poses[index], full["hybrid", "off"][1][index][:10])

        # No producer thread may outlive an empty input, empty pool or fatal exit.
        empty = root / "empty.mol2"
        empty.write_bytes(b"")
        for ranking in ("fast", "hybrid"):
            output = root / f"empty-{ranking}"
            result = invoke(executable, protein, pocket, empty, output, environment, ranking, "on")
            require("Input contains no MOL2 records" in result.stderr, "Empty input failed for an unrelated reason")
            require(not (output / "summary.csv").exists(), "Empty input claimed a completed run")
            require(not library_test.table(output / "records.csv", library_test.RECORD_FIELDS),
                    "Empty input fabricated records")

        one = root / "one.mol2"
        one.write_bytes(library_test.renamed_record(valid, library_test.IDS[0]))
        no_pool = root / "hybrid-empty-index"
        invoke(executable, protein, pocket, one, no_pool, environment, "hybrid", "on",
               cutoff="0.0001", expected_error=False)
        records = library_test.table(no_pool / "records.csv", library_test.RECORD_FIELDS)
        require(len(records) == 1 and records[0]["status"] == "no_poses" and
                all(int(records[0][field]) == 0 for field in
                    ("candidate_matches", "alignments", "valid_transforms", "survivors", "candidate_pool")),
                "Empty index did not preserve an empty CUDA rescoring pool")
        require(not library_test.table(no_pool / "poses.csv", library_test.POSE_FIELDS),
                "Empty pool exported poses")

        failed = root / "hybrid-fail-fast"
        invoke(executable, protein, pocket, library, failed, environment, "hybrid", "on", recover=False)
        records, poses = library_test.inspect(failed, (1,), recover=False)
        require(not (failed / "summary.csv").exists(), "Fail-fast run claimed a completed summary")
        for field in records[0]:
            require(records[0][field] == full["hybrid", "off"][0][0][field], "Fail-fast changed completed prefix")
        compare_pose_rows(poses[1], full["hybrid", "off"][1][1])

    require(all(library_test.digest(path) == original for path, original in originals.items()),
            "An original molecular input changed")
    print("CUDA overlap OFF/ON, full survivors, all rankings, Hybrid normalization and termination passed")


if __name__ == "__main__":
    main()
