"""Tests for optimization.evaluate (table building, no invented values).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.test_evaluate -v
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimization.evaluate import (
    BRUSH_LOAD_STATUS, COLUMNS, QUALITY, collect_rows, coverage_proxy,
    main, roundtrip_max_abs_err, write_csv, write_markdown,
)
from optimization.test_pipeline import make_vertex_array, write_ply


class EvaluateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.orig_v = make_vertex_array(200, seed=5, sh_degree=3)
        self.orig = write_ply(self.dir / "orig.ply", self.orig_v,
                              sh_degree=3)


class TestRoundtrip(EvaluateTestCase):
    def test_roundtrip_max_abs_err_known_difference(self):
        other = self.orig_v.copy()
        step = 0.0078125  # 2^-7, exactly representable in float32
        other["x"] = other["x"] + step
        decoded = write_ply(self.dir / "decoded.ply", other, sh_degree=3)
        err = roundtrip_max_abs_err(self.orig, decoded)
        self.assertAlmostEqual(err, step, places=5)

    def test_roundtrip_identical_is_zero(self):
        same = write_ply(self.dir / "same.ply", self.orig_v.copy(),
                         sh_degree=3)
        self.assertEqual(roundtrip_max_abs_err(self.orig, same), 0.0)

    def test_roundtrip_count_mismatch_raises(self):
        shorter = write_ply(self.dir / "short.ply",
                            make_vertex_array(50, seed=6), sh_degree=3)
        with self.assertRaises(ValueError):
            roundtrip_max_abs_err(self.orig, shorter)


class TestCollectRows(EvaluateTestCase):
    def test_original_row_and_pruned_row(self):
        subset = write_ply(self.dir / "sub.ply", self.orig_v[:150],
                           sh_degree=3)
        rows = collect_rows(self.orig, {"pruned": str(subset)},
                            times={"pruned": 1.5})
        self.assertEqual(len(rows), 2)
        original, pruned = rows
        self.assertEqual(original["variant"], "original")
        self.assertEqual(original["size_ratio"], 1.0)
        self.assertEqual(original["processing_time_s"], "n/a")
        self.assertEqual(original["psnr"], QUALITY)
        self.assertEqual(original["brush_load_status"],
                         BRUSH_LOAD_STATUS["original"])

        self.assertEqual(pruned["gaussians"], 150)
        self.assertGreater(pruned["size_ratio"], 1.0)
        self.assertEqual(pruned["processing_time_s"], 1.5)
        self.assertEqual(pruned["psnr"], QUALITY)
        self.assertEqual(pruned["ssim"], QUALITY)
        self.assertEqual(pruned["fps"], QUALITY)
        self.assertEqual(pruned["brush_load_status"], "not tested")
        self.assertTrue(pruned["coverage_proxy"].startswith("voxels emptied"),
                        pruned["coverage_proxy"])
        self.assertEqual(pruned["route_b_roundtrip_max_abs_err"], "n/a")

    def test_time_without_flag_is_not_measured(self):
        same = write_ply(self.dir / "same2.ply", self.orig_v.copy(),
                         sh_degree=3)
        rows = collect_rows(self.orig, {"rewritten": str(same)})
        row = rows[1]
        self.assertEqual(row["processing_time_s"], QUALITY)
        # same count as the original -> nothing was removed
        self.assertEqual(row["coverage_proxy"], "n/a (no removals)")
        self.assertEqual(row["brush_load_status"], "not tested")

    def test_sh_truncated_layout_is_not_comparable(self):
        v = make_vertex_array(200, seed=5, sh_degree=2)
        # different count (rows also differ) and different property layout
        shd = write_ply(self.dir / "shd2.ply", v[:150], sh_degree=2)
        rows = collect_rows(self.orig, {"smart_shd2": str(shd)})
        self.assertEqual(rows[1]["coverage_proxy"],
                         "n/a (layout differs from original)")

    def test_unknown_time_label_rejected(self):
        with self.assertRaises(ValueError):
            collect_rows(self.orig, {}, times={"ghost": 1.0})

    def test_missing_original_rejected(self):
        with self.assertRaises(FileNotFoundError):
            collect_rows(self.dir / "nope.ply", {})

    def test_coverage_of_untouched_row_count(self):
        v = make_vertex_array(40, seed=8)
        self.assertEqual(coverage_proxy(v, v.copy(), 5.0),
                         "n/a (no removals)")


class TestOutputs(EvaluateTestCase):
    def test_csv_and_markdown_written(self):
        subset = write_ply(self.dir / "sub.ply", self.orig_v[:150],
                           sh_degree=3)
        rows = collect_rows(self.orig, {"smart_default": str(subset)},
                            times={"smart_default": 2.0})
        csv_path = write_csv(rows, self.dir / "out.csv")
        md_path = write_markdown(rows, str(self.orig), self.dir / "out.md")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, list(COLUMNS))
            data = list(reader)
        self.assertEqual(len(data), 2)
        for col in ("psnr", "ssim", "fps"):
            self.assertEqual({r[col] for r in data}, {QUALITY})
        self.assertEqual(data[1]["brush_load_status"], "not tested")

        md = md_path.read_text(encoding="utf-8")
        self.assertIn("| variant |", md)
        self.assertIn("not measured", md)
        # no raw pipe inside table headers (would break the table)
        header_line = next(line for line in md.splitlines()
                           if line.startswith("| variant |"))
        self.assertEqual(header_line.count("|"), 11 + 1)

    def test_main_writes_reports_and_validates_paths(self):
        subset = write_ply(self.dir / "sub.ply", self.orig_v[:150],
                           sh_degree=3)
        rc = main(["--original", str(self.orig),
                   "--variant", f"smart_default={subset}",
                   "--time", "smart_default=2.5",
                   "--out-csv", str(self.dir / "t.csv"),
                   "--out-md", str(self.dir / "t.md")])
        self.assertEqual(rc, 0)
        self.assertTrue((self.dir / "t.csv").is_file())
        self.assertTrue((self.dir / "t.md").is_file())

        rc = main(["--original", str(self.dir / "missing.ply"),
                   "--out-csv", str(self.dir / "x.csv"),
                   "--out-md", str(self.dir / "x.md")])
        self.assertEqual(rc, 2)
        self.assertFalse((self.dir / "x.csv").exists())

        rc = main(["--original", str(self.orig),
                   "--time", "ghost=1.0",
                   "--out-csv", str(self.dir / "y.csv"),
                   "--out-md", str(self.dir / "y.md")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
