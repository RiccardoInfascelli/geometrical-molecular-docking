#!/usr/bin/env python3
"""Compare bounded batch prefetch and adaptive CUDA batches with serial Vinardo."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile

import gpu_docking_batch_test as batch
import gpu_docking_library_test as library


require = library.require
MIB = 1024 * 1024
RECORD_COUNT = 19
INVALID_RECORD = 17
ATTEMPT_FIELDS = ("batch_index", "attempt_index", "first_record", "last_record",
                  "prepared_records", "status", "required_bytes", "budget_bytes",
                  "dock_start_s", "dock_end_s")
TIMELINE_FIELDS = ("record_index", "status", "query_start_s", "query_end_s",
                   "dock_start_s", "dock_end_s")


def invoke(executable, protein, pocket, input_path, output, environment,
           batch_size=None, overlap="off", memory_mib=4096, recover=True,
           cutoff="24", expected_code=1, center_reference=None):
    command = [str(executable), str(protein), str(pocket), str(input_path), str(output),
               "--limit", str(RECORD_COUNT), "--cutoff", cutoff,
               "--kabsch", "quaternion", "--query-method", "reference",
               "--ranking", "vinardo", "--rescoring-backend", "cuda",
               "--vinardo-alpha", "1", "--fast-prefilter", "none",
               "--top-k", "10", "--pose-output", "topk", "--pose-detail", "full",
               "--no-sdf", "--timings", "--diagnostics", "off",
               "--gpu-batch", "off" if batch_size is None else "on",
               "--cpu-gpu-overlap", overlap]
    if batch_size is not None:
        command += ["--batch-size", str(batch_size), "--batch-memory-mib", str(memory_mib)]
    if recover:
        command.append("--continue-on-error")
    if center_reference is not None:
        command += ["--pose-center-distance", "on", "--center-reference-mol2", str(center_reference)]
    completed = subprocess.run(command, env=environment, text=True, encoding="utf-8",
        errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
    require(completed.returncode == expected_code,
            f"Batch {batch_size}/{overlap}/{memory_mib}MiB returned {completed.returncode}, "
            f"expected {expected_code}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    require(output.is_dir(), "The docking CLI did not create its output directory")
    require(not any(output.rglob("*.sdf")), "CSV-only prefetch created SDF output")
    return completed


def finite_nonnegative(value, description):
    result = float(value)
    require(math.isfinite(result) and result >= 0, f"Invalid {description}: {value!r}")
    return result


def check_timeline(output, mode, records, attempts):
    rows = library.table(output / "batch_overlap_timeline.csv", TIMELINE_FIELDS)
    require([int(row["record_index"]) for row in rows] == list(range(1, RECORD_COUNT + 1)),
            "Batch timeline lost, duplicated or reordered records")
    successful = [row for row in attempts if row["status"] == "ok"]
    spans = []
    for row, record in zip(rows, records):
        require(row["status"] == record["status"], "Timeline status differs from record output")
        if record["status"] == "error":
            require(row["dock_start_s"] == row["dock_end_s"] == "",
                    "A failed molecular record was attributed CUDA docking")
            continue
        begin, end = [finite_nonnegative(row[field], field) for field in ("dock_start_s", "dock_end_s")]
        query_begin, query_end = [finite_nonnegative(row[field], field)
                                  for field in ("query_start_s", "query_end_s")]
        require(query_begin <= query_end <= begin <= end,
                "A batch used a packet before query preparation finished")
        index = int(row["record_index"])
        eligible = [attempt for attempt in successful
                    if int(attempt["first_record"]) <= index <= int(attempt["last_record"])]
        require(len(eligible) == 1, "A successful record has no unique completed subbatch")
        expected = eligible[0]
        require(math.isclose(begin, float(expected["dock_start_s"]), rel_tol=0.0, abs_tol=1e-9) and
                math.isclose(end, float(expected["dock_end_s"]), rel_tol=0.0, abs_tol=1e-9),
                "Timeline docking span does not match the completed subbatch")
        spans.append((query_begin, query_end, begin, end))
    if mode == "off":
        for query_begin, query_end, _, _ in spans:
            require(all(min(query_end, end) <= max(query_begin, begin)
                        for _, _, begin, end in spans),
                    "OFF overlapped CPU query work with a docking wrapper")
    # ON need not overlap on every input or scheduler run. Host intervals are
    # scheduling evidence; only a device timeline establishes kernel overlap.


def inspect(output, names, batch_size=None, overlap="off", memory_mib=4096, with_center_distance=False):
    summary = library.summary(output / "summary.csv")
    for key, expected in {
        "status": "complete_with_errors", "processed": str(RECORD_COUNT),
        "successful_records": str(RECORD_COUNT - 1), "failed_records": "1",
        "ranking_requested": "vinardo", "ranking_method": "vinardo",
        "rescoring_backend": "cuda", "fast_prefilter": "none",
        "kabsch_method_effective": "quaternion", "query_method_effective": "reference",
        "query_threads_requested": "1", "cpu_gpu_overlap": overlap,
        "index_policy": "shared_index_only", "sdf_export": "0",
    }.items():
        require(summary.get(key) == expected, f"Incorrect effective {key}: {summary.get(key)!r}")
    batch.close(float(summary["vinardo_alpha"]), 1.0, "vinardo_alpha")
    if with_center_distance:
        require(summary.get("pose_center_distance") == "on" and
                summary.get("pose_center_reference_mode") == "shared_reference_mol2" and
                summary.get("rmsd") == "disabled", "Center-distance export changed RMSD/reference policy")
    else:
        require("pose_center_distance" not in summary, "Disabled center-distance export changed metadata")
    wall = finite_nonnegative(summary["online_wall_s"], "online wall")
    require(wall > 0, "The online wall interval is empty")
    throughput = finite_nonnegative(summary["online_successful_ligands_per_s"], "online throughput")
    require(math.isclose(throughput, (RECORD_COUNT - 1) / wall, rel_tol=1e-12, abs_tol=1e-12),
            "Online throughput does not use the completed successful-record count")
    for key in ("consumer_wait_s", "producer_queue_wait_s"):
        finite_nonnegative(summary[key], key)

    records = library.table(output / "records.csv", library.RECORD_FIELDS + ("candidate_pool", "error"))
    require([int(row["record_index"]) for row in records] == list(range(1, RECORD_COUNT + 1)),
            "The pipeline lost, duplicated or reordered records")
    require([row["molecule_id"] for row in records] == names, "Molecule IDs changed")
    poses = library.table(output / "poses.csv", library.POSE_FIELDS + batch.SCORE_FIELDS + ("pose_id",))
    for row in poses:
        require(("center_distance_A" in row) == with_center_distance,
                "Center-distance CSV field does not match the requested export mode")
        if with_center_distance:
            finite_nonnegative(row["center_distance_A"], "center_distance_A")
    grouped = {}
    for record in records:
        index = record["record_index"]
        selected = [row for row in poses if row["record_index"] == index]
        grouped[index] = selected
        if int(index) == INVALID_RECORD:
            require(record["status"] == "error" and record["error"] and not selected,
                    "A malformed record was not retained as an isolated failure")
            require(all(int(record[key]) == 0 for key in
                        ("candidate_matches", "alignments", "valid_transforms", "survivors", "candidate_pool")),
                    "A failed parse fabricated scientific counts")
            continue
        require(record["status"] == "ok" and not record["error"], "A valid record failed")
        require(all(int(record[key]) > 0 for key in
                    ("candidate_matches", "alignments", "valid_transforms", "survivors")),
                "The integration complex must exercise real geometry and clash survivors")
        require(record["candidate_pool"] == record["survivors"], "An implicit prefilter reduced Vinardo's pool")
        require(len(selected) == int(record["exported_poses"]) == min(10, int(record["survivors"])),
                "The selected Top-10 was lost or truncated")
        require([int(row["rank"]) for row in selected] == list(range(1, len(selected) + 1)),
                "Exported ranks are not contiguous")
        require(len({row["pose_id"] for row in selected}) == len(selected) and
                len({row["alignment_index"] for row in selected}) == len(selected),
                "A record exported duplicate pose identities")
        for row in selected:
            require(row["molecule_id"] == record["molecule_id"] and
                    0 <= int(row["pose_id"]) < int(record["survivors"]),
                    "A pose belongs to the wrong ligand or survivor pool")
            batch.close(float(row["vinardo_adjusted_score"]), float(row["vinardo_like_score"]),
                        "vinardo_adjusted_score")
        for left, right in zip(selected, selected[1:]):
            a, b = float(left["vinardo_adjusted_score"]), float(right["vinardo_adjusted_score"])
            require(a <= b and (a != b or int(left["pose_id"]) < int(right["pose_id"])),
                    "Vinardo order or tie-breaking changed")
    require(sum(len(rows) for rows in grouped.values()) == len(poses), "An unknown record exported poses")

    if batch_size is None:
        return records, grouped, summary, []
    require(summary.get("gpu_batch") == "on" and int(summary["batch_size"]) == batch_size,
            "The requested aggregate batch mode was not activated")
    require(int(summary["batch_memory_budget_bytes"]) == memory_mib * MIB,
            "The requested explicit memory budget was not recorded")
    require(int(summary["batch_prefetch_capacity_records"]) == 2 * batch_size and
            int(summary["batch_prefetch_budget_bytes"]) == 268435456,
            "The bounded CPU queue has an unexpected capacity")
    require(0 <= int(summary["batch_prefetch_high_water_records"]) <= 2 * batch_size and
            0 <= int(summary["batch_prefetch_high_water_bytes"]) <= 268435456,
            "CPU prefetch exceeded its bounded record/byte capacity")
    for key in ("batch_adaptive_splits", "batch_memory_retries"):
        require(int(summary[key]) >= 0, f"Invalid adaptive counter {key}")
    batches = library.table(output / "batches.csv", (
        "batch_index", "records", "prepared_records", "candidate_matches", "alignments",
        "valid_transforms", "survivors", "search_completed_s", "dock_completed_s", "explicit_device_bytes"))
    require(sum(int(row["records"]) for row in batches) == RECORD_COUNT and
            sum(int(row["prepared_records"]) for row in batches) == RECORD_COUNT - 1,
            "Batch accounting lost records")
    for row in batches:
        require(0 < int(row["records"]) <= batch_size and
                0 <= int(row["prepared_records"]) <= int(row["records"]),
                "A CPU batch exceeded the requested input capacity")
        require(0 <= int(row["explicit_device_bytes"]) <= memory_mib * MIB,
                "A completed subbatch exceeded the explicit device budget")
        for key in ("search_completed_s", "dock_completed_s"):
            finite_nonnegative(row[key], key)
    attempts = library.table(output / "batch_attempts.csv", ATTEMPT_FIELDS)
    completed = []
    previous_end = 0.0
    attempt_ids = set()
    for row in attempts:
        begin, end = [finite_nonnegative(row[key], key) for key in ("dock_start_s", "dock_end_s")]
        require(previous_end <= begin <= end, "CUDA consumer attempts ran concurrently or out of order")
        previous_end = end
        require(row["status"] in {"ok", "split", "retry_single", "budget_error"},
                "Unknown adaptive attempt status")
        identity = (int(row["batch_index"]), int(row["attempt_index"]))
        require(identity not in attempt_ids and identity[0] > 0 and identity[1] > 0,
                "Adaptive attempt identities were lost or duplicated")
        attempt_ids.add(identity)
        require(1 <= int(row["first_record"]) <= int(row["last_record"]) <= RECORD_COUNT and
                0 < int(row["prepared_records"]) <= batch_size,
                "An adaptive attempt contains an invalid record interval")
        require(int(row["budget_bytes"]) == memory_mib * MIB and int(row["required_bytes"]) >= 0,
                "An attempt used an unrecorded memory budget")
        if row["status"] == "ok":
            require(int(row["required_bytes"]) <= memory_mib * MIB,
                    "An accepted attempt exceeded the explicit budget")
            completed.append(row)
        else:
            require(int(row["required_bytes"]) > memory_mib * MIB,
                    "A rejected budget attempt did not explain its required capacity")
    require(sum(int(row["prepared_records"]) for row in completed) == RECORD_COUNT - 1,
            "Successful adaptive subbatches lost or duplicated prepared ligands")
    require(int(summary["batch_adaptive_splits"]) == sum(row["status"] == "split" for row in attempts) and
            int(summary["batch_memory_retries"]) == sum(row["status"] != "ok" for row in attempts),
            "Adaptive summary counters disagree with the actual attempt log")
    check_timeline(output, overlap, records, attempts)
    return records, grouped, summary, batches


def compare(actual, expected):
    batch.compare_records(actual[0], expected[0])
    for index in expected[1]:
        batch.compare_poses(actual[1][index], expected[1][index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("complex_directory", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    complex_directory = args.complex_directory.resolve(strict=True)
    identifier = complex_directory.name
    protein, pocket, ligand = [complex_directory / f"{identifier}_{suffix}"
                               for suffix in ("protein.pdb", "pocket.pdb", "ligand.mol2")]
    originals = {path: library.digest(path) for path in (protein, pocket, ligand)}
    original_record = library.first_record(ligand)
    environment = os.environ.copy()
    for key in ("DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS", "DOCKING_AREA_CROP_MARGIN_A"):
        environment.pop(key, None)
    environment.update(DOCKING_KABSCH_METHOD="quaternion", DOCKING_QUERY_METHOD="reference",
        DOCKING_QUERY_THREADS="1", DOCKING_PROFILE_STAGES="0", OMP_NUM_THREADS="1")
    names = [f"batch-overlap-{index}" for index in range(1, RECORD_COUNT + 1)]
    with tempfile.TemporaryDirectory(prefix="gpu-docking-batch-overlap-") as temporary:
        root = Path(temporary)
        input_path = root / "input.mol2"
        input_path.write_bytes(b"".join(
            b"@<TRIPOS>MOLECULE\n" + names[index - 1].encode() + b"\ninvalid\n"
            if index == INVALID_RECORD else
            batch.moved_record(original_record, names[index - 1], index, deform=index in (2, 6))
            for index in range(1, RECORD_COUNT + 1)))
        output = root / "serial"
        invoke(executable, protein, pocket, input_path, output, environment, center_reference=ligand)
        reference = inspect(output, names, with_center_distance=True)
        output = root / "serial-center-off"
        invoke(executable, protein, pocket, input_path, output, environment)
        disabled = inspect(output, names)
        batch.compare_records(reference[0], disabled[0])
        for index in disabled[1]:
            without_center = [{key: value for key, value in row.items() if key != "center_distance_A"}
                              for row in reference[1][index]]
            batch.compare_poses(without_center, disabled[1][index])
        results = {}
        for size in (1, 4, 8, 16):
            for mode in ("off", "on"):
                output = root / f"batch-{size}-{mode}"
                invoke(executable, protein, pocket, input_path, output, environment, size, mode,
                       center_reference=ligand)
                results[size, mode] = inspect(output, names, size, mode, with_center_distance=True)
                compare(results[size, mode], reference)
        single_peak = max(int(row["explicit_device_bytes"])
                          for mode in ("off", "on") for row in results[1, mode][3])
        small_budget = (single_peak + MIB - 1) // MIB + 1
        aggregate_peak = max(int(row["explicit_device_bytes"]) for row in results[16, "off"][3])
        require(small_budget * MIB < aggregate_peak,
                "The configured complex cannot exercise adaptive budget subdivision")
        for mode in ("off", "on"):
            output = root / f"adaptive-{mode}"
            invoke(executable, protein, pocket, input_path, output, environment, 16, mode, small_budget,
                   center_reference=ligand)
            result = inspect(output, names, 16, mode, small_budget, with_center_distance=True)
            compare(result, reference)
            require(int(result[2]["batch_adaptive_splits"]) > 0 and
                    int(result[2]["batch_memory_retries"]) > 0,
                    "The reduced budget did not exercise adaptive subdivision")

        fail_fast = root / "fail-fast"
        invoke(executable, protein, pocket, input_path, fail_fast, environment, 16, "on", recover=False,
               center_reference=ligand)
        records = library.table(fail_fast / "records.csv", library.RECORD_FIELDS)
        require([int(row["record_index"]) for row in records] == list(range(1, INVALID_RECORD)),
                "Fail-fast did not preserve exactly the completed record prefix")
        require(not (fail_fast / "summary.csv").exists(), "Fail-fast claimed a complete run")
        poses = library.table(fail_fast / "poses.csv", library.POSE_FIELDS)
        for index in range(1, INVALID_RECORD):
            batch.compare_poses([row for row in poses if int(row["record_index"]) == index], reference[1][str(index)])

        empty = root / "empty.mol2"
        empty.write_bytes(b"")
        output = root / "empty-output"
        completed = invoke(executable, protein, pocket, empty, output, environment, 4, "on")
        require("Input contains no MOL2 records" in completed.stderr and not (output / "summary.csv").exists(),
                "An empty producer failed incorrectly or claimed successful completion")

        one = root / "one.mol2"
        one.write_bytes(batch.moved_record(original_record, names[0], 1))
        output = root / "empty-index"
        invoke(executable, protein, pocket, one, output, environment, 4, "on", cutoff="0.0001", expected_code=0)
        rows = library.table(output / "records.csv", library.RECORD_FIELDS + ("candidate_pool",))
        require(len(rows) == 1 and rows[0]["status"] == "no_poses" and
                all(int(rows[0][key]) == 0 for key in
                    ("candidate_matches", "alignments", "valid_transforms", "survivors", "candidate_pool", "exported_poses")),
                "An empty shared index changed the completed empty-pool result")
        first_peak = int(results[1, "off"][3][0]["explicit_device_bytes"])
        if first_peak > MIB:
            output = root / "single-over-budget"
            insufficient_mib = max(1, (first_peak - 1) // MIB)
            completed = invoke(executable, protein, pocket, one, output, environment,
                16, "on", insufficient_mib)
            require("memory budget exceeded" in completed.stderr and
                    not (output / "summary.csv").exists(),
                    "An over-budget singleton did not fail clearly after its bounded retry")
            require(not library.table(output / "records.csv", library.RECORD_FIELDS) and
                    not library.table(output / "poses.csv", library.POSE_FIELDS),
                    "An over-budget singleton produced a partial scientific result")
    require(all(library.digest(path) == before for path, before in originals.items()),
            "An original molecular input was modified")
    print("Serial/batch Vinardo 1/4/8/16, bounded OFF/ON prefetch, adaptive budgets and recovery passed")


if __name__ == "__main__":
    main()
