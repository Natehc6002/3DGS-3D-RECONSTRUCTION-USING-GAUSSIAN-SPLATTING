"""Unit tests for mirror_clip (pure plan functions + CLI exit codes).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.pruning.test_mirror_clip -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.pruning import mirror_clip as mc
from optimization.pruning.test_smart_prune import make_vertex_array


def make_rep(normal=(0.0, 1.0, 0.0), point=(0.0, 0.0, 0.0),
             confidence=0.5, subject_centre=(0.0, 5.0, 0.0),
             subject=True, extent=10.0):
    return {
        "extent": {"robust_extent": extent},
        "plane": {"normal": list(normal), "point": list(point),
                  "confidence": confidence, "method": "test plane"},
        "subject": ({"centre": list(subject_centre), "radius_p90": 1.0,
                     "confidence": 1.0, "low_confidence": False}
                    if subject else {"radius_p90": None, "confidence": 0.0}),
    }


def make_scene(n_above=50, n_below=50, seed=0):
    rng = np.random.default_rng(seed)
    above = np.column_stack([
        rng.uniform(-4.0, 4.0, n_above),
        rng.uniform(0.5, 4.0, n_above),
        rng.uniform(-4.0, 4.0, n_above)])
    below = np.column_stack([
        rng.uniform(-4.0, 4.0, n_below),
        rng.uniform(-4.0, -0.5, n_below),
        rng.uniform(-4.0, 4.0, n_below)])
    xyz = np.vstack([above, below])
    return {"xyz": xyz, "weights": np.ones(len(xyz)), "count": len(xyz)}


class TestBuildPlan(unittest.TestCase):
    def test_clips_only_beyond_margin(self):
        rep = make_rep()
        scene = make_scene()
        plan = mc.build_plan(rep, scene)
        # margin = 0.02 * 10 = 0.2; every below point has y <= -0.5
        self.assertTrue(plan["remove"][50:].all())
        self.assertFalse(plan["remove"][:50].any())
        self.assertEqual(plan["removed_count"], 50)
        self.assertEqual(plan["kept_count"], 50)
        self.assertAlmostEqual(plan["removed_weight_share"], 0.5)
        self.assertTrue(plan["subject_checked"])
        self.assertFalse(plan["manual_plane"])
        self.assertAlmostEqual(plan["margin"], 0.2)

    def test_subject_on_removed_side_refused(self):
        rep = make_rep(subject_centre=(0.0, -5.0, 0.0))
        scene = make_scene()
        with self.assertRaises(ValueError) as ctx:
            mc.build_plan(rep, scene)
        self.assertIn("subject centre", str(ctx.exception))

    def test_unreliable_plane_refused(self):
        rep = make_rep(confidence=0.2)
        scene = make_scene()
        with self.assertRaises(ValueError) as ctx:
            mc.build_plan(rep, scene)
        self.assertIn("--plane", str(ctx.exception))

    def test_manual_plane_overrides_unreliable_detection(self):
        rep = make_rep(confidence=0.1)
        scene = make_scene()
        plan = mc.build_plan(rep, scene, plane=((0.0, 1.0, 0.0), 0.0))
        self.assertTrue(plan["manual_plane"])
        self.assertEqual(plan["plane_confidence"], 1.0)
        self.assertEqual(plan["removed_count"], 50)

    def test_manual_plane_sign_flips_removed_side(self):
        rep = make_rep(subject=False)
        scene = make_scene()
        plan = mc.build_plan(rep, scene, plane=((0.0, -1.0, 0.0), 0.0))
        # removed side is -y < -margin -> y > margin -> the above half
        self.assertTrue(plan["remove"][:50].all())
        self.assertFalse(plan["remove"][50:].any())

    def test_weight_fraction_guard(self):
        rep = make_rep()
        scene = make_scene()
        with self.assertRaises(ValueError) as ctx:
            mc.build_plan(rep, scene, max_weight_fraction=0.4)
        self.assertIn("weight", str(ctx.exception))

    def test_no_subject_falls_back_to_weight_guard_only(self):
        rep = make_rep(subject=False)
        scene = make_scene()
        plan = mc.build_plan(rep, scene)
        self.assertFalse(plan["subject_checked"])
        self.assertEqual(plan["removed_count"], 50)

    def test_zero_margin_clips_plane_touchers(self):
        rep = make_rep()
        scene = {"xyz": np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
                 "weights": np.array([1.0, 1.0]), "count": 2}
        plan = mc.build_plan(rep, scene, margin_ratio=0.0)
        np.testing.assert_array_equal(plan["remove"], [False, True])


class TestWriteAndCli(unittest.TestCase):
    def test_write_preserves_comments_and_property_order(self):
        v = make_vertex_array(40, seed=9, scrambled=True)
        comments = ["Exported from Brush", "Vertical axis: y"]
        remove = np.zeros(40, dtype=bool)
        remove[:10] = True
        plan = {"remove": remove}
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(v, "vertex")],
                    text=False, byte_order="<", comments=comments).write(str(src))
            kept = mc.write_clipped(src, dst, plan)
            out = PlyData.read(str(dst), mmap=None)
            self.assertEqual(kept, 30)
            self.assertEqual(list(out["vertex"].data.dtype.names),
                             list(v.dtype.names))
            self.assertEqual(list(out.comments), comments)
            self.assertEqual(len(out["vertex"].data), 30)

    def test_write_refuses_removing_everything(self):
        v = make_vertex_array(5, seed=1)
        plan = {"remove": np.ones(5, dtype=bool)}
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(v, "vertex")],
                    text=False, byte_order="<").write(str(src))
            with self.assertRaises(ValueError):
                mc.write_clipped(src, dst, plan)
            self.assertFalse(dst.exists())

    def test_main_missing_input_is_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            rc = mc.main(["--input", str(Path(td) / "nope.ply"),
                          "--output", str(Path(td) / "out.ply")])
        self.assertEqual(rc, 2)

    def test_main_output_exists_is_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(make_vertex_array(4), "vertex")],
                    text=False, byte_order="<").write(str(src))
            dst.write_bytes(b"x")
            rc = mc.main(["--input", str(src), "--output", str(dst)])
        self.assertEqual(rc, 2)

    def test_main_bad_arguments_are_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            PlyData([PlyElement.describe(make_vertex_array(4), "vertex")],
                    text=False, byte_order="<").write(str(src))
            rc = mc.main(["--input", str(src),
                          "--output", str(Path(td) / "o.ply"),
                          "--max-weight-fraction", "1.5"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
