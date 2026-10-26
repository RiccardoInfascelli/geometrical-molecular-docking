#!/usr/bin/env python3
"""Exercise the real CUDA CLI on a temporary valid/invalid/valid MOL2 library."""

import csv
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


RECORD_FIELDS = (
    "record_index", "molecule_id", "status", "diameter_A", "diameter_gt_cutoff",
    "ligand_triplets", "candidate_matches", "alignments", "valid_transforms",
    "survivors", "exported_poses", "sdf",
)
COUNT_FIELDS = RECORD_FIELDS[4:11]
POSE_FIELDS = (
    "record_index", "molecule_id", "rank", "alignment_index", "fast_weighted_score",
    "clash_fraction", "occupied_atoms", "r00", "r01", "r02", "r10", "r11", "r12",
    "r20", "r21", "r22", "tx", "ty", "tz",
)
TIMING_FIELDS = (
    "record_index", "read_parse_s", "ligand_preparation_s", "search_completed_s",
    "top_k_selection_s", "reference_rmsd_export_s", "query_preparation_s", "gpu_search_s",
    "query_threads_requested", "query_threads_used",
)
IDS = ("valid_first", "invalid_middle", "valid_last")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def table(path, required):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames
        require(fields is not None and len(set(fields)) == len(fields), f"Invalid CSV header: {path}")
        require(set(required) <= set(fields), f"Missing CSV columns in {path}: {set(required) - set(fields)}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"CSV row width differs from its header: {path}")
    return rows


def summary(path):
    rows = table(path, ("key", "value"))
    result = {row["key"]: row["value"] for row in rows}
    require(len(result) == len(rows), f"Duplicate summary keys: {path}")
    return result


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.digest()


def first_record(path):
    lines = path.read_bytes().splitlines(keepends=True)
    markers = [index for index, line in enumerate(lines) if line.startswith(b"@<TRIPOS>MOLECULE")]
    require(markers, f"No MOL2 record in {path}")
    block = lines[markers[0]:markers[1] if len(markers) > 1 else len(lines)]
    require(len(block) >= 3, f"The first ligand MOL2 record is truncated: {path}")
    return block


def renamed_record(lines, name):
    copied = list(lines)
    ending = b"\r\n" if copied[1].endswith(b"\r\n") else b"\n"
    copied[1] = name.encode("ascii") + ending
    block = b"".join(copied)
    # Keep every coordinate, bond, atom index and other input field intact.
    return block if block.endswith(b"\n") else block + b"\n"


def run(executable, protein, pocket, library, output, environment, recover):
    command = [str(executable), str(protein), str(pocket), str(library), str(output),
               "--no-sdf", "--timings", "--kabsch", "quaternion", "--query-method", "reference",
               "--ranking", "fast", "--limit", "3"]
    if recover:
        command.append("--continue-on-error")
    completed = subprocess.run(command, env=environment, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, check=False)
    require(completed.returncode == 1,
            f"Expected exit 1 for recover={recover}, got {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    require(output.is_dir(), f"Docking did not create its output directory: {completed.stderr}")
    require(not any(path.suffix.lower() == ".sdf" for path in output.rglob("*")),
            f"--no-sdf created an SDF file in {output}")


def inspect(output, expected_indices, recover):
    records = table(output / "records.csv", RECORD_FIELDS + (("error",) if recover else ()))
    require([row["record_index"] for row in records] == [str(index) for index in expected_indices],
            f"Wrong record order or missing rows in {output}")
    for row, index in zip(records, expected_indices):
        require(row["molecule_id"] == IDS[index - 1], f"Wrong molecule ID for record {index}")
        require(row["sdf"] == "", f"Record {index} advertises an SDF with --no-sdf")
        require(math.isfinite(float(row["diameter_A"])), f"Nonfinite diameter for record {index}")
        require(all(int(row[field]) >= 0 for field in COUNT_FIELDS), f"Invalid counts for record {index}")
        if index == 2:
            require(row["status"] == "error", "Malformed MOL2 record was not marked as an error")
            stage, separator, reason = row["error"].partition(":")
            require(separator and stage == "parsing/sanitization" and reason.strip(),
                    "Record error lacks its parsing/sanitization stage or reason")
            require(float(row["diameter_A"]) == 0 and all(int(row[field]) == 0 for field in COUNT_FIELDS),
                    "Parse failure fabricated scientific counts")
        else:
            require(row["status"] == "ok" and not row.get("error", ""), f"Valid record {index} failed")
            require(all(int(row[field]) > 0 for field in
                        ("ligand_triplets", "candidate_matches", "valid_transforms", "exported_poses")),
                    "The configured integration complex must produce candidate matches and exported poses")

    poses = table(output / "poses.csv", POSE_FIELDS)
    successful = [index for index in expected_indices if index != 2]
    require(all(int(row["record_index"]) in successful for row in poses), "Failed record exported poses")
    grouped = {}
    for index in successful:
        selected = [row for row in poses if int(row["record_index"]) == index]
        record = next(row for row in records if int(row["record_index"]) == index)
        require(len(selected) == int(record["exported_poses"]), f"Pose count differs for record {index}")
        require([int(row["rank"]) for row in selected] == list(range(1, len(selected) + 1)),
                f"Pose ranks differ for record {index}")
        for row in selected:
            require(row["molecule_id"] == IDS[index - 1], f"Pose molecule ID differs for record {index}")
            require(all(math.isfinite(float(row[field])) for field in POSE_FIELDS[4:]),
                    f"Nonfinite pose score or transform for record {index}")
        grouped[index] = [{key: value for key, value in row.items()
                           if key not in ("record_index", "molecule_id")} for row in selected]

    timings = table(output / "timings.csv", TIMING_FIELDS)
    require([int(row["record_index"]) for row in timings] == successful,
            "Timing rows must contain only the successful records in source order")
    for row in timings:
        require(all(math.isfinite(float(value)) and float(value) >= 0
                    for field, value in row.items() if field.endswith("_s")), "Invalid timing interval")
        require(row["query_threads_requested"] == row["query_threads_used"] == "1",
                "Timing rows did not preserve the fixed serial query preparation")
    return records, grouped


def main():
    require(len(sys.argv) == 3, "Usage: gpu_docking_library_test.py <gpu_docking> <complex_directory>")
    executable = Path(sys.argv[1]).resolve(strict=True)
    complex_directory = Path(sys.argv[2]).resolve(strict=True)
    complex_id = complex_directory.name
    protein = complex_directory / f"{complex_id}_protein.pdb"
    pocket = complex_directory / f"{complex_id}_pocket.pdb"
    ligand = complex_directory / f"{complex_id}_ligand.mol2"
    originals = {path: digest(path) for path in (protein, pocket, ligand)}
    valid = first_record(ligand)
    environment = os.environ.copy()
    for key in ("DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS", "DOCKING_AREA_CROP_MARGIN_A"):
        environment.pop(key, None)
    environment.update(DOCKING_PROFILE_STAGES="0", DOCKING_QUERY_THREADS="1")
    # TemporaryDirectory honors TMPDIR. All generated inputs and output directories live here.
    with tempfile.TemporaryDirectory(prefix="gpu-docking-library-") as temporary:
        root = Path(temporary)
        library = root / "library.mol2"
        library.write_bytes(renamed_record(valid, IDS[0]) +
                            b"@<TRIPOS>MOLECULE\ninvalid_middle\n" + renamed_record(valid, IDS[2]))
        recovered = root / "recovered"
        run(executable, protein, pocket, library, recovered, environment, recover=True)
        completed = summary(recovered / "summary.csv")
        for key, expected in {"status": "complete_with_errors", "processed": "3", "successful_records": "2",
                              "failed_records": "1", "continue_on_error": "1", "sdf_export": "0"}.items():
            require(completed.get(key) == expected, f"Summary {key}: expected {expected}, got {completed.get(key)}")
        records, poses = inspect(recovered, (1, 2, 3), recover=True)
        comparison_fields = set(records[0]) - {"record_index", "molecule_id"}
        require(all(records[0][field] == records[2][field] for field in comparison_fields),
                "Repeated valid ligand changed scientific counts or status after a parse failure")
        require(poses[1] == poses[3], "Repeated valid ligand changed pose order, scores or transforms")

        fail_fast = root / "fail-fast"
        run(executable, protein, pocket, library, fail_fast, environment, recover=False)
        partial_records, partial_poses = inspect(fail_fast, (1,), recover=False)
        if (fail_fast / "summary.csv").exists():
            require(summary(fail_fast / "summary.csv").get("status") not in ("complete", "complete_with_errors"),
                    "Fail-fast run claimed a completed summary")
        require(all(partial_records[0][field] == records[0][field] for field in RECORD_FIELDS),
                "Fail-fast changed the first completed record")
        require(partial_poses[1] == poses[1], "Fail-fast changed the first record's scores or transforms")
    require(all(digest(path) == original for path, original in originals.items()), "An original input file changed")
    print("GPU library recovery, fail-fast prefix and --no-sdf integration passed")


if __name__ == "__main__":
    main()
