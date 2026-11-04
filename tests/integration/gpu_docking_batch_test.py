#!/usr/bin/env python3
"""Compare aggregate CUDA docking with the serial CLI on one real complex."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import gpu_docking_library_test as library


TRANSFORMS = tuple(f"r{i}{j}" for i in range(3) for j in range(3)) + ("tx", "ty", "tz")
INTEGER_FIELDS = {"record_index", "rank", "alignment_index", "occupied_atoms", "pose_id"}
RANKINGS = ("fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid")
SCORE_FIELDS = ("weighted_interaction_score", "vinardo_like_score", "vinardo_gaussian",
                "vinardo_repulsion", "vinardo_hydrophobic", "vinardo_hydrogen_bond",
                "vinardo_adjusted_score", "hybrid_score")
require = library.require


def moved_record(lines, name, offset, deform=False):
    # Distinct coordinate frames expose missing point/atom offsets even when
    # adjacent records contain the same molecular graph.
    copied = library.renamed_record(lines, name).decode("utf-8").splitlines(keepends=True)
    in_atoms = False
    for index, line in enumerate(copied):
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
        elif in_atoms and line.strip():
            fields = line.split()
            require(len(fields) >= 6, "Truncated atom row in the integration fixture")
            x, y, z = map(float, fields[2:5])
            if deform:
                # Small anisotropic changes give a distinct conformer/pool,
                # rather than another rigid frame of the same coordinates.
                x, y, z = 1.025 * x, 0.985 * y, 1.01 * z
            fields[2:5] = [f"{coordinate:.10f}" for coordinate in
                           (-y + offset * 2.75, x - offset * 1.25, z + offset * 0.625)]
            copied[index] = " ".join(fields) + "\n"
    return "".join(copied).encode("utf-8")


def close(a, b, field):
    absolute, relative = ((1e-6, 0.0) if field in TRANSFORMS else
                          (1e-11, 1e-9) if field == "hybrid_score" else
                          (1e-10, 1e-8) if field.startswith("vinardo_") else (1e-9, 0.0))
    require(math.isfinite(a) and math.isfinite(b) and
            abs(a - b) <= absolute + relative * abs(b),
            f"Batch changed {field}: {a:.17g} versus {b:.17g}")


def compare_poses(actual, expected):
    require(len(actual) == len(expected), "Batch changed the number of returned poses")
    for left, right in zip(actual, expected):
        require(set(left) == set(right), "Batch changed pose CSV fields")
        for key, value in right.items():
            if key == "molecule_id" or key in INTEGER_FIELDS or value == "":
                require(left[key] == value, f"Batch changed pose {key}")
            else:
                close(float(left[key]), float(value), key)


def alpha_for(ranking):
    return 0.73 if ranking == "vinardo" else 0.5


def hybrid_values(poses):
    def zscores(values):
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        deviation = math.sqrt(variance)
        return [(value - mean) / deviation if deviation else 0.0 for value in values]
    weighted = zscores([float(row["weighted_interaction_score"]) for row in poses])
    vinardo = zscores([-float(row["vinardo_adjusted_score"]) for row in poses])
    return [0.3 * first + 0.7 * second for first, second in zip(weighted, vinardo)]


def check_ranking(poses, ranking, complete=True):
    score = {"fast": "fast_weighted_score", "weighted-interaction": "weighted_interaction_score",
             "vinardo": "vinardo_adjusted_score", "vinardo-rep50": "vinardo_adjusted_score",
             "hybrid": "hybrid_score"}[ranking]
    identity = "alignment_index" if ranking == "fast" else "pose_id"
    for left, right in zip(poses, poses[1:]):
        a, b = float(left[score]), float(right[score])
        require(a <= b if ranking.startswith("vinardo") else a >= b,
                f"{ranking} score order changed")
        if a == b:
            require(int(left[identity]) < int(right[identity]), f"{ranking} tie-breaking changed")
    if ranking.startswith("vinardo") or ranking == "hybrid":
        for row in poses:
            adjusted = float(row["vinardo_like_score"]) + (alpha_for(ranking) - 1.0) * float(row["vinardo_repulsion"])
            close(float(row["vinardo_adjusted_score"]), adjusted, "vinardo_adjusted_score")
    if ranking == "hybrid" and poses and complete:
        for row, expected in zip(poses, hybrid_values(poses)):
            close(float(row["hybrid_score"]), expected, "hybrid_score")


def run(executable, protein, pocket, input_path, output, environment, batch_size=None,
        quantity="all", expected_code=1, cutoff="24", recover=True,
        ranking="fast", prefilter="none"):
    command = [str(executable), str(protein), str(pocket), str(input_path), str(output),
               "--limit", "10", "--cutoff", cutoff, "--kabsch", "quaternion",
               "--query-method", "reference", "--ranking", ranking, "--top-k", "10",
               "--pose-output", quantity, "--pose-detail", "full", "--no-sdf",
               "--timings", "--diagnostics", "off",
               "--gpu-batch", "off" if batch_size is None else "on"]
    if recover:
        command.append("--continue-on-error")
    if batch_size is not None:
        command += ["--batch-size", str(batch_size)]
    if ranking != "fast":
        command += ["--rescoring-backend", "cuda", "--fast-prefilter", prefilter]
    if ranking in ("weighted-interaction", "hybrid"):
        command += ["--interaction-region", "pocket"]
    if ranking.startswith("vinardo") or ranking == "hybrid":
        command += ["--vinardo-alpha", str(alpha_for(ranking))]
    if ranking == "hybrid":
        command += ["--hybrid-lambda", "0.3"]
    completed = subprocess.run(command, env=environment, text=True, encoding="utf-8",
        errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, check=False)
    require(completed.returncode == expected_code,
            f"Batch {batch_size}/{quantity}: expected exit {expected_code}, got {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    require(output.is_dir(), "Docking did not create its output directory")
    require(not any(output.rglob("*.sdf")), "CSV-only batch created SDF output")
    return completed


def inspect(output, names, batch_size=None, ranking="fast", prefilter="none", quantity="all"):
    summary = library.summary(output / "summary.csv")
    for key, expected in {"status": "complete_with_errors", "processed": "10",
                          "successful_records": "9", "failed_records": "1",
                          "ranking_method": "vinardo" if ranking.startswith("vinardo") else ranking,
                          "ranking_requested": ranking, "fast_prefilter": prefilter,
                          "index_policy": "shared_index_only", "sdf_export": "0"}.items():
        require(summary.get(key) == expected, f"Batch summary {key} differs: {summary.get(key)!r}")
    require(summary.get("gpu_batch", "off") == ("off" if batch_size is None else "on"),
            "CLI did not select the requested GPU batch mode")
    require(summary.get("kabsch_method_effective") == "quaternion" and
            summary.get("query_method_effective") == "reference" and
            summary.get("query_threads_requested") == "1",
            "The requested scientific/query protocol was not recorded")
    if ranking != "fast":
        require(summary.get("rescoring_backend") == "cuda", "Rescoring did not use CUDA")
    if ranking in ("weighted-interaction", "hybrid"):
        require(summary.get("interaction_region") == "pocket", "Wrong WeightedInteraction region")
    if ranking.startswith("vinardo") or ranking == "hybrid":
        close(float(summary["vinardo_alpha"]), alpha_for(ranking), "vinardo_alpha")
    if ranking == "hybrid":
        close(float(summary["hybrid_lambda"]), 0.3, "hybrid_score")
    if batch_size is not None:
        require(int(summary["batch_size"]) == batch_size, "CLI ignored batch size")
        budget = int(summary["batch_memory_budget_bytes"])
        batches = library.table(output / "batches.csv", (
            "batch_index", "records", "prepared_records", "candidate_matches", "alignments",
            "valid_transforms", "survivors", "search_completed_s", "dock_completed_s", "explicit_device_bytes"))
        require(sum(int(row["records"]) for row in batches) == 10 and
                sum(int(row["prepared_records"]) for row in batches) == 9,
                "Batch table lost records or included an invalid molecule")
        for row in batches:
            require(0 < int(row["records"]) <= batch_size and
                    0 <= int(row["prepared_records"]) <= int(row["records"]),
                    "Batch exceeds its configured record capacity")
            require(0 <= int(row["explicit_device_bytes"]) <= budget,
                    "Batch memory estimate exceeds its declared budget")
            for key in ("search_completed_s", "dock_completed_s"):
                require(math.isfinite(float(row[key])) and float(row[key]) >= 0,
                        "Batch timing is nonfinite or negative")
            if ranking != "fast":
                for key in ("rescoring_completed_s", "fast_prefilter_s", "weighted_interaction_scoring_s",
                            "vinardo_scoring_s", "rescoring_chemical_preparation_s",
                            "rescoring_h2d_s", "rescoring_score_preparation_s", "hybrid_normalization_s",
                            "rescoring_top_k_d2h_s"):
                    require(key in row and math.isfinite(float(row[key])) and float(row[key]) >= 0,
                            f"Missing or invalid aggregate {key}")
        timings = library.table(output / "timings.csv", ("record_index", "search_completed_s", "gpu_search_s"))
        require(all(row["search_completed_s"] == row["gpu_search_s"] == "" for row in timings),
                "Aggregate GPU work was reported as a measured per-ligand interval")
        if ranking != "fast":
            require(all(row[field] == "" for row in timings for field in
                        ("weighted_interaction_scoring_s", "vinardo_scoring_s", "rescoring_h2d_s")),
                    "Shared scoring/upload time was attributed to individual ligands")
            require(all(math.isfinite(float(row[field])) and float(row[field]) >= 0
                        for row in timings for field in ("fast_prefilter_s",
                            "rescoring_chemical_preparation_s", "rescoring_score_preparation_s",
                            "hybrid_normalization_s", "rescoring_top_k_d2h_s")),
                    "A per-ligand rescoring interval is missing or invalid")
    records = library.table(output / "records.csv", library.RECORD_FIELDS + ("error",))
    require([int(row["record_index"]) for row in records] == list(range(1, 11)),
            "Batch lost, duplicated or reordered input records")
    require([row["molecule_id"] for row in records] == names, "Batch changed molecule IDs")
    for row in records:
        if row["record_index"] == "9":
            require(row["status"] == "error" and row["error"], "Malformed record was not retained as an error")
            require(all(int(row[key]) == 0 for key in
                        ("candidate_matches", "alignments", "valid_transforms", "survivors", "exported_poses")),
                    "Malformed record fabricated scientific counts")
        else:
            require(row["status"] == "ok" and not row["error"], "A valid record failed")
            require(all(int(row[key]) > 0 for key in
                        ("candidate_matches", "alignments", "valid_transforms", "survivors", "exported_poses")),
                    "The configured complex must exercise actual pose generation")
            if ranking != "fast":
                pool = int(row["candidate_pool"])
                require(pool == (int(row["survivors"]) if prefilter == "none" else
                                 min(int(prefilter), int(row["survivors"]))),
                        "Rescoring changed the per-ligand selected pool size")
    rows = library.table(output / "poses.csv", library.POSE_FIELDS)
    grouped = {}
    for record in records:
        index = record["record_index"]
        selected = [row for row in rows if row["record_index"] == index]
        require(len(selected) == int(record["exported_poses"]), "Batch export count differs")
        require([int(row["rank"]) for row in selected] == list(range(1, len(selected) + 1)),
                "Batch ranks are not a contiguous prefix")
        require(all(row["molecule_id"] == record["molecule_id"] for row in selected),
                "A pose belongs to the wrong ligand")
        require(len({row["alignment_index"] for row in selected}) == len(selected),
                "Batch duplicated a ligand's alignment ID")
        if ranking != "fast":
            require(all(field in row for row in selected for field in SCORE_FIELDS),
                    "Rescoring export omitted score/component fields")
            require(len({row["pose_id"] for row in selected}) == len(selected) and
                    all(0 <= int(row["pose_id"]) < int(record["survivors"]) for row in selected),
                    "Rescoring changed local survivor IDs")
            expected = int(record["candidate_pool"])
            require(len(selected) == (expected if quantity == "all" else min(10, expected)),
                    "Export omitted part of the selected rescoring pool")
            if prefilter == "none" and quantity == "all":
                require({int(row["pose_id"]) for row in selected} == set(range(expected)),
                        "Full rescoring export lost local survivor identities")
        check_ranking(selected, ranking, complete=quantity == "all")
        grouped[index] = selected
    require(sum(len(group) for group in grouped.values()) == len(rows), "An unknown record exported a pose")
    return records, grouped


def compare_records(actual, expected, compare_exported=True):
    require(len(actual) == len(expected), "Batch changed the number of record rows")
    for left, right in zip(actual, expected):
        require(set(left) == set(right), "Batch changed record CSV fields")
        for key in right:
            if key == "error":
                require(bool(left[key]) == bool(right[key]), "Batch changed error handling")
            elif key != "exported_poses" or compare_exported:
                require(left[key] == right[key], f"Batch changed record {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("complex_directory", type=Path)
    parser.add_argument("--ranking", choices=RANKINGS, default="fast")
    options = parser.parse_args()
    executable = options.executable.resolve(strict=True)
    complex_directory = options.complex_directory.resolve(strict=True)
    ranking = options.ranking
    identifier = complex_directory.name
    protein, pocket, ligand = [complex_directory / f"{identifier}_{suffix}"
                               for suffix in ("protein.pdb", "pocket.pdb", "ligand.mol2")]
    originals = {path: library.digest(path) for path in (protein, pocket, ligand)}
    valid = library.first_record(ligand)
    environment = os.environ.copy()
    for key in ("DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS", "DOCKING_AREA_CROP_MARGIN_A"):
        environment.pop(key, None)
    environment.update(DOCKING_KABSCH_METHOD="quaternion", DOCKING_QUERY_METHOD="reference",
        DOCKING_QUERY_THREADS="1", DOCKING_PROFILE_STAGES="0", OMP_NUM_THREADS="1")
    names = [f"ligand-{index}" for index in range(1, 11)]
    with tempfile.TemporaryDirectory(prefix="gpu-docking-batch-") as temporary:
        root = Path(temporary)
        input_path = root / "input.mol2"
        input_path.write_bytes(b"".join(
            (b"@<TRIPOS>MOLECULE\n" + names[index - 1].encode() + b"\ninvalid\n")
            if index == 9 else moved_record(valid, names[index - 1], index,
                deform=ranking != "fast" and index in (2, 6))
            for index in range(1, 11)))
        complete_pool = None
        for prefilter in (("none",) if ranking == "fast" else ("none", "7")):
            serial = root / f"serial-{prefilter}"
            run(executable, protein, pocket, input_path, serial, environment,
                ranking=ranking, prefilter=prefilter)
            reference_records, reference_poses = inspect(serial, names, ranking=ranking, prefilter=prefilter)
            if prefilter == "none":
                require(all(len(reference_poses[row["record_index"]]) == int(row["survivors"])
                            for row in reference_records), "Serial full export omitted survivors")
                complete_pool = reference_poses
                if ranking != "fast":
                    require(len(reference_poses["1"]) != len(reference_poses["2"]) or
                            {(row["alignment_index"], row["weighted_interaction_score"], row["vinardo_like_score"])
                             for row in reference_poses["1"]} !=
                            {(row["alignment_index"], row["weighted_interaction_score"], row["vinardo_like_score"])
                             for row in reference_poses["2"]},
                            "The deformed fixture must create distinct pose/scoring pools")
            else:
                require(any(len(rows) > 7 for rows in complete_pool.values()),
                        "The configured complex must exercise actual Fast prefilter truncation")
                for index, selected in reference_poses.items():
                    expected = sorted(complete_pool[index],
                        key=lambda row: (-float(row["fast_weighted_score"]), int(row["pose_id"])))[:7]
                    require({row["pose_id"] for row in selected} == {row["pose_id"] for row in expected},
                            "Fast prefilter did not preserve its per-ligand selection and tie-breaking")
            if ranking == "hybrid":
                pooled = [row for rows in reference_poses.values() for row in rows]
                require(any(not math.isclose(float(row["hybrid_score"]), value, abs_tol=1e-8, rel_tol=1e-8)
                            for row, value in zip(pooled, hybrid_values(pooled))),
                        "The fixture cannot distinguish per-ligand from pooled Hybrid normalization")
            for batch_size in (1, 2, 4, 8):
                output = root / f"batch-{batch_size}-{prefilter}"
                run(executable, protein, pocket, input_path, output, environment, batch_size,
                    ranking=ranking, prefilter=prefilter)
                records, poses = inspect(output, names, batch_size, ranking, prefilter)
                compare_records(records, reference_records)
                for index in reference_poses:
                    compare_poses(poses[index], reference_poses[index])
                if ranking != "fast" or batch_size == 8:
                    output = root / f"batch-{batch_size}-{prefilter}-top10"
                    run(executable, protein, pocket, input_path, output, environment, batch_size, "topk",
                        ranking=ranking, prefilter=prefilter)
                    records, poses = inspect(output, names, batch_size, ranking, prefilter, "topk")
                    compare_records(records, reference_records, compare_exported=False)
                    for index in reference_poses:
                        compare_poses(poses[index], reference_poses[index][:10])

        # Error and empty-pool paths use the unfiltered protocol, including CUDA
        # rescoring for the optional non-Fast test invocations.
        reference_records, reference_poses = inspect(root / "serial-none", names, ranking=ranking)

        output = root / "batch-fail-fast"
        fail_input = root / "fail-fast.mol2"
        fail_input.write_bytes(moved_record(valid, names[0], 1) +
            b"@<TRIPOS>MOLECULE\ninvalid_middle\ninvalid\n" + moved_record(valid, names[2], 3))
        run(executable, protein, pocket, fail_input, output, environment, 8, recover=False, ranking=ranking)
        partial = library.table(output / "records.csv", library.RECORD_FIELDS)
        require([int(row["record_index"]) for row in partial] == [1],
                "Fail-fast lost its completed prefix or processed records after the error")
        for actual, expected in zip(partial, reference_records):
            require(all(actual[key] == expected[key] for key in actual),
                    "Batch fail-fast changed a completed record")
        require(not (output / "summary.csv").exists(), "Fail-fast batch claimed a completed run")
        exported = library.table(output / "poses.csv", library.POSE_FIELDS)
        compare_poses(exported, reference_poses["1"])

        empty = root / "empty.mol2"
        empty.write_bytes(b"")
        output = root / "empty-output"
        completed = run(executable, protein, pocket, empty, output, environment, 4, ranking=ranking)
        require("Input contains no MOL2 records" in completed.stderr,
                "Empty batch input failed for an unrelated reason")
        require(not (output / "summary.csv").exists(), "Empty input claimed a completed run")

        one = root / "one.mol2"
        one.write_bytes(library.renamed_record(valid, names[0]))
        output = root / "empty-index"
        run(executable, protein, pocket, one, output, environment, 4,
            expected_code=0, cutoff="0.0001", ranking=ranking)
        rows = library.table(output / "records.csv", library.RECORD_FIELDS)
        require(len(rows) == 1 and rows[0]["status"] == "no_poses" and
                all(int(rows[0][key]) == 0 for key in
                    ("candidate_matches", "alignments", "valid_transforms", "survivors", "exported_poses")),
                "An empty shared index changed batch results or skipped the ligand")
    require(all(library.digest(path) == before for path, before in originals.items()),
            "A molecular input was modified")
    print(f"Aggregate {ranking} batches 1/2/4/8, full pools, Top-K, partial tail and record recovery passed")


if __name__ == "__main__":
    main()
