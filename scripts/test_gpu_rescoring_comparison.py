#!/usr/bin/env python3
"""Synthetic export checks; no scientific inputs or CUDA execution."""

import csv
from pathlib import Path
import tempfile
import unittest

from compare_gpu_rescoring import POSE_FIELDS, RECORD_FIELDS, compare_runs


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = [Path(self.temporary.name) / name for name in ("cpu", "cuda")]
        for path in self.paths:
            path.mkdir()
        self.summary = dict(status="complete", processed="1", top_k="2", ranking_method="hybrid",
            pose_pool="all_clash_survivors", fast_prefilter="none", index_policy="shared_index_only",
            index_cutoff_A="24", tie_break="original alignment/survivor order ascending")
        self.record = dict.fromkeys(RECORD_FIELDS, "2")
        self.record.update(record_index="1", molecule_id="synthetic", status="ok", diameter_A="3.5",
            diameter_gt_cutoff="0", candidate_matches="6", alignments="8", valid_transforms="8",
            survivors="2", exported_poses="2", candidate_pool="2", sdf="record-1.sdf")
        self.poses = []
        for pose_id in range(2):
            row = dict.fromkeys(POSE_FIELDS, "0")
            row.update(record_index="1", molecule_id="synthetic", rank=str(pose_id + 1),
                pose_id=str(pose_id), alignment_index=str(pose_id), fast_weighted_score="5",
                weighted_interaction_score="2", hybrid_score=str(1 - pose_id), vinardo_like_score="-3",
                vinardo_adjusted_score="-3")
            self.poses.append(row)
        self.save()

    def save(self, gpu_poses=None, gpu_record=None, **gpu_summary):
        for number, path in enumerate(self.paths):
            summary = dict(self.summary, rescoring_backend=("cpu", "cuda")[number])
            summary.update(gpu_summary if number else {})
            write_csv(path / "summary.csv", ["key", "value"],
                      [dict(key=key, value=value) for key, value in summary.items()])
            write_csv(path / "records.csv", RECORD_FIELDS,
                      [gpu_record if number and gpu_record is not None else self.record])
            rows = gpu_poses if number and gpu_poses is not None else self.poses
            write_csv(path / "poses.csv", POSE_FIELDS, rows)

    def compare(self):
        return compare_runs(*self.paths)

    def test_equal_and_full_pool(self):
        result = self.compare()
        self.assertTrue(result["ok"])
        self.assertTrue(result["full_pool_exported"])
        self.assertEqual(result["compared_poses"], 2)

    def test_tolerated_roundoff_and_report(self):
        for field, change in (("vinardo_like_score", 1e-9), ("hybrid_score", 1e-10)):
            with self.subTest(field=field):
                rows = [dict(row) for row in self.poses]
                rows[0][field] = str(float(rows[0][field]) + change)
                self.save(gpu_poses=rows)
                result = self.compare()
                self.assertTrue(result["ok"])
                self.assertGreater(result["max_errors"][field]["absolute"], 0)

    def test_tolerance_combines_absolute_and_cpu_relative_parts(self):
        self.poses[0]["vinardo_gaussian"] = "0.01"
        rows = [dict(row) for row in self.poses]
        rows[0]["vinardo_gaussian"] = str(0.01 + 1.5e-10)
        self.save(gpu_poses=rows)
        self.assertTrue(self.compare()["ok"])
        rows[0]["vinardo_gaussian"] = str(0.01 + 2.5e-10)
        self.save(gpu_poses=rows)
        self.assertFalse(self.compare()["ok"])

    def test_fast_weighted_interaction_clash_and_transforms_require_exact_equality(self):
        for field in ("fast_weighted_score", "weighted_interaction_score", "clash_fraction", "tx"):
            with self.subTest(field=field):
                rows = [dict(row) for row in self.poses]
                rows[0][field] = str(float(rows[0][field]) + 1e-12)
                self.save(gpu_poses=rows)
                self.assertFalse(self.compare()["ok"])

    def test_order_changes_are_reported_even_when_scores_match(self):
        rows = [dict(row, rank=str(i + 1)) for i, row in enumerate(reversed(self.poses))]
        self.save(gpu_poses=rows)
        result = self.compare()
        self.assertEqual(result["top1_mismatches"], [1])
        self.assertEqual(result["order_mismatches"], [1])
        self.assertEqual(result["topk_set_mismatches"], [])
        self.assertFalse(result["ok"])

    def test_protocol_or_counts_mismatch(self):
        self.save(index_cutoff_A="20")
        self.assertEqual(self.compare()["protocol_mismatches"], ["index_cutoff_A"])
        self.save(gpu_record=dict(self.record, candidate_matches="7"))
        self.assertEqual(self.compare()["record_mismatches"], [1])

    def test_topk_rowsets_differ_without_hiding_the_common_scores(self):
        self.record.update(survivors="3", candidate_pool="3")
        rows = [dict(self.poses[0]), dict(self.poses[1], pose_id="2")]
        self.save(gpu_poses=rows)
        result = self.compare()
        self.assertFalse(result["full_pool_exported"])
        self.assertEqual(result["compared_poses"], 1)
        self.assertEqual(result["topk_set_mismatches"], [1])
        self.assertEqual(result["pose_set_mismatches"], [[1, 1], [1, 2]])

    def test_rmsd_does_not_affect_comparison(self):
        for number, path in enumerate(self.paths):
            rows = [dict(row, rmsd_heavy_A=str(number + 0.5)) for row in self.poses]
            write_csv(path / "poses.csv", POSE_FIELDS + ["rmsd_heavy_A"], rows)
        self.assertTrue(self.compare()["ok"])

    def test_incomplete_duplicate_missing_or_nonfinite_export(self):
        cases = [dict(status="partial"), dict(processed="2"), dict(index_cutoff_A="nan")]
        for changes in cases:
            with self.subTest(changes=changes):
                self.save(**changes)
                with self.assertRaises(ValueError):
                    self.compare()
        for rows in ([], self.poses + self.poses[:1], [dict(self.poses[0], weighted_interaction_score="nan"), self.poses[1]]):
            self.save(gpu_poses=rows)
            with self.assertRaises(ValueError):
                self.compare()
        self.save()
        (self.paths[1] / "summary.csv").unlink()
        with self.assertRaises(ValueError):
            self.compare()

    def test_empty_survivor_pool_is_valid(self):
        self.record.update(status="no_poses", survivors="0", candidate_pool="0", exported_poses="0")
        self.poses = []
        self.save()
        self.assertTrue(self.compare()["ok"])

    def test_unfiltered_protocol_must_score_every_survivor(self):
        self.record["survivors"] = "3"
        self.save()
        with self.assertRaises(ValueError):
            self.compare()


if __name__ == "__main__":
    unittest.main()
