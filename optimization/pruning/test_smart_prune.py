"""Unit tests for the pure functions in smart_prune.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.pruning.test_smart_prune -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.pruning import smart_prune as sp


def make_vertex_array(n, seed=0, scrambled=False):
    """Small synthetic vertex array. scrambled=True puts columns in an
    odd order (like Brush's f_rest text-sort) to prove name-based access."""
    rng = np.random.default_rng(seed)
    fields = [
        ("opacity", "<f4"), ("scale_2", "<f4"), ("f_rest_10", "<f4"),
        ("x", "<f4"), ("rot_1", "<f4"), ("scale_0", "<f4"),
        ("f_rest_2", "<f4"), ("y", "<f4"), ("rot_0", "<f4"),
        ("scale_1", "<f4"), ("f_rest_0", "<f4"), ("z", "<f4"),
        ("rot_2", "<f4"), ("f_dc_0", "<f4"), ("rot_3", "<f4"),
        ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("f_rest_1", "<f4"),
    ]
    if not scrambled:
        fields = sorted(fields, key=lambda t: t[0])
    dt = np.dtype(fields)
    v = np.zeros(n, dtype=dt)
    for name in dt.names:
        v[name] = rng.normal(size=n).astype("<f4")
    return v


class TestSigmoidAndImportance(unittest.TestCase):
    def test_sigmoid_known_values(self):
        x = np.array([-1000.0, 0.0, 1000.0])
        s = sp.sigmoid(x)
        self.assertAlmostEqual(s[0], 0.0, places=12)
        self.assertAlmostEqual(s[1], 0.5, places=12)
        self.assertAlmostEqual(s[2], 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(s)))

    def test_importance_by_name_not_position(self):
        # Values chosen so a wrong column would give a wrong answer:
        # opacity = logit(0.5) = 0, scales = 0 -> volume = 4/3*pi
        v = make_vertex_array(3, scrambled=True)
        v["opacity"] = 0.0
        v["scale_0"] = 0.0
        v["scale_1"] = 0.0
        v["scale_2"] = 0.0
        imp = sp.importance_scores(v)
        # sigmoid(0) = 0.5, so importance = 0.5 * 4/3 * pi
        np.testing.assert_allclose(imp, 0.5 * 4.0 / 3.0 * np.pi, rtol=1e-6)

    def test_importance_scales_with_opacity(self):
        v = make_vertex_array(2, scrambled=True)
        v["scale_0"] = v["scale_1"] = v["scale_2"] = 0.0
        v["opacity"] = np.array([0.0, 2.0], dtype="<f4")  # sigmoid 0.5 vs ~0.88
        imp = sp.importance_scores(v)
        self.assertGreater(imp[1], imp[0])

    def test_require_properties_missing(self):
        v = make_vertex_array(2)
        names = [n for n in v.dtype.names if n != "opacity"]
        bad = v[names]
        with self.assertRaises(ValueError) as ctx:
            sp.require_properties(bad)
        self.assertIn("opacity", str(ctx.exception))

    def test_require_properties_ok(self):
        sp.require_properties(make_vertex_array(2))  # must not raise


class TestQuaternionRenormalize(unittest.TestCase):
    def test_unit_quat_unchanged(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.70710678, 0.0, 0.70710678]])
        out = sp.renormalize_quaternions(q)
        np.testing.assert_allclose(out, q, atol=1e-9)

    def test_non_unit_becomes_unit(self):
        q = np.array([[2.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 5.0],
                      [1.0, 1.0, 1.0, 1.0]])
        out = sp.renormalize_quaternions(q)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)
        # direction preserved
        np.testing.assert_allclose(out[0], [1.0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(out[2], [0.5, 0.5, 0.5, 0.5], atol=1e-12)

    def test_zero_quat_becomes_identity(self):
        out = sp.renormalize_quaternions(np.zeros((1, 4)))
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_batch_shape_preserved(self):
        q = np.random.default_rng(1).normal(size=(7, 4))
        self.assertEqual(sp.renormalize_quaternions(q).shape, (7, 4))


class TestVoxelSafety(unittest.TestCase):
    def test_never_empty_and_fraction_respected(self):
        counts = np.array([1, 2, 3, 4, 10, 100, 26581])
        cap = sp.voxel_capacities(counts, 0.3)
        self.assertTrue(np.all(cap >= 0))
        self.assertTrue(np.all(cap < counts))            # never empty
        self.assertTrue(np.all(cap <= np.floor(counts * 0.3)))  # fraction

    def test_known_values(self):
        counts = np.array([1, 3, 4, 10])
        cap = sp.voxel_capacities(counts, 0.3)
        np.testing.assert_array_equal(cap, [0, 0, 1, 3])

    def test_fraction_one_allows_all_but_one(self):
        counts = np.array([1, 5])
        cap = sp.voxel_capacities(counts, 1.0)
        np.testing.assert_array_equal(cap, [0, 4])

    def test_zero_fraction_removes_nothing(self):
        cap = sp.voxel_capacities(np.array([5, 9]), 0.0)
        np.testing.assert_array_equal(cap, [0, 0])


class TestFlagsAndSelection(unittest.TestCase):
    def _grid(self, spacing=1.0, shape=(6, 6, 6)):
        xs = np.arange(shape[0]) * spacing
        ys = np.arange(shape[1]) * spacing
        zs = np.arange(shape[2]) * spacing
        g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        return g

    def test_isolated_point_flagged(self):
        xyz = self._grid()
        xyz = np.vstack([xyz, [[100.0, 100.0, 100.0]]])  # far floater
        v = make_vertex_array(len(xyz), scrambled=True)
        flags, thr = sp.build_flags(xyz, v, knn_k=3, iso_percentile=95.0,
                                    size_percentile=99.9,
                                    body_percentile_low=0.5,
                                    body_percentile_high=99.5,
                                    body_margin=5.0)
        self.assertTrue(flags["isolated"][-1])
        self.assertFalse(flags["isolated"][:-1].any())
        self.assertGreater(thr["isolation_distance"], 0.0)

    def test_body_outlier_flagged(self):
        xyz = self._grid()
        xyz = np.vstack([xyz, [[500.0, 500.0, 500.0]]])
        v = make_vertex_array(len(xyz), scrambled=True)
        flags, _ = sp.build_flags(xyz, v, knn_k=3, iso_percentile=99.0,
                                  size_percentile=99.9,
                                  body_percentile_low=0.5,
                                  body_percentile_high=99.5,
                                  body_margin=5.0)
        self.assertTrue(flags["outside_body"][-1])
        self.assertFalse(flags["outside_body"][:-1].any())

    def test_no_voxel_loses_more_than_fraction_and_none_emptied(self):
        rng = np.random.default_rng(3)
        xyz = rng.uniform(0, 50, size=(5000, 3))
        geo = np.zeros(len(xyz), dtype=bool)
        geo[rng.choice(len(xyz), 300, replace=False)] = True
        low = np.zeros(len(xyz), dtype=bool)
        low[rng.choice(len(xyz), 2000, replace=False)] = True
        imp = rng.uniform(0, 1, size=len(xyz))
        sel = sp.select_removals(xyz, geo, low, imp, max_fraction=0.3,
                                 voxel_size=5.0)
        counts, rem = sel["voxel_counts"], sel["voxel_removed"]
        self.assertTrue(np.all(rem <= np.floor(counts * 0.3) + 1e-9))
        self.assertTrue(np.all(rem < counts))  # never empty
        self.assertEqual(int(rem.sum()), int(sel["remove"].sum()))

    def test_geo_candidates_removed_before_low_importance(self):
        # One voxel, two Gaussians: geo-flagged has HIGH importance,
        # low-importance candidate has LOW importance. Capacity is 1,
        # so the geo-flagged one must win (geo priority), showing that
        # floater detection is not blocked by importance rank.
        xyz = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        geo = np.array([True, False])
        low = np.array([False, True])
        imp = np.array([10.0, 0.001])
        sel = sp.select_removals(xyz, geo, low, imp, max_fraction=0.5,
                                 voxel_size=10.0)
        np.testing.assert_array_equal(sel["remove"], [True, False])

    def test_importance_rank_within_voxel(self):
        # Same voxel, both geo-flagged: lower importance removed first.
        xyz = np.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]])
        geo = np.array([True, True, True])
        low = np.zeros(3, dtype=bool)
        imp = np.array([5.0, 0.1, 2.0])
        sel = sp.select_removals(xyz, geo, low, imp, max_fraction=0.34,
                                 voxel_size=10.0)  # floor(3*0.34)=1
        np.testing.assert_array_equal(sel["remove"], [False, True, False])

    def test_singleton_kept_by_default(self):
        xyz = np.array([[0.0, 0.0, 0.0]])
        geo = np.array([True])
        low = np.array([False])
        imp = np.array([1.0])
        sel = sp.select_removals(xyz, geo, low, imp, max_fraction=0.9,
                                 voxel_size=1.0)
        self.assertFalse(sel["remove"][0])

    def test_singleton_removed_with_exception_flag(self):
        xyz = np.array([[0.0, 0.0, 0.0]])
        sel = sp.select_removals(xyz, np.array([True]), np.array([False]),
                                 np.array([1.0]), max_fraction=0.9,
                                 voxel_size=1.0, allow_singleton_floaters=True)
        self.assertTrue(sel["remove"][0])

    def test_primary_reason_precedence(self):
        flags = {"outside_body": np.array([True, True, False, False]),
                 "isolated": np.array([True, False, True, False]),
                 "size_outlier": np.array([True, False, False, True])}
        low = np.array([True, True, True, False])
        r = sp.primary_reason(flags, low)
        np.testing.assert_array_equal(
            r, [sp.REASON_OUTSIDE, sp.REASON_OUTSIDE,
                sp.REASON_ISOLATED, sp.REASON_SIZE])


class TestNeedleCleanup(unittest.TestCase):
    """Needle flag: scene-relative, identical criteria to analyze_scene."""

    N_NORM = 100
    N_NEEDLE = 3

    def _scene(self):
        rng = np.random.default_rng(11)
        n = self.N_NORM + self.N_NEEDLE
        v = make_vertex_array(n, seed=5, scrambled=True)
        xyz = rng.uniform(10.0, 40.0, size=(n, 3))
        # 10 normals share the needles' voxel [0, 10)^3 so the per-voxel
        # budget (floor(0.3 * count) >= 3) can actually spend all needles
        xyz[:10] = rng.uniform(2.0, 8.0, size=(10, 3))
        xyz[self.N_NORM:] = [[5.0, 5.0, 5.0],
                             [6.0, 6.0, 6.0],
                             [7.0, 7.0, 7.0]]
        for i, k in enumerate(("x", "y", "z")):
            v[k] = xyz[:, i].astype("<f4")
        # normal Gaussians: near-isotropic axes around exp(0)=1
        for k in sp.SCALE_KEYS:
            v[k][:self.N_NORM] = rng.normal(0.0, 0.3, self.N_NORM)
        # needles: max axis 30, min axis 0.1 -> anisotropy 300
        v["scale_0"][self.N_NORM:] = np.log(30.0)
        v["scale_1"][self.N_NORM:] = np.log(0.1)
        v["scale_2"][self.N_NORM:] = np.log(0.1)
        return xyz, v

    def _flags(self, xyz, v, needle_cleanup):
        return sp.build_flags(xyz, v, knn_k=5, iso_percentile=99.0,
                              size_percentile=99.9,
                              body_percentile_low=0.5,
                              body_percentile_high=99.5,
                              body_margin=5.0,
                              needle_cleanup=needle_cleanup)

    def test_needles_flagged_when_enabled(self):
        xyz, v = self._scene()
        flags, thr = self._flags(xyz, v, True)
        self.assertTrue(thr["needle_enabled"])
        self.assertTrue(flags["needle"][self.N_NORM:].all())
        self.assertGreaterEqual(thr["needle_aniso_threshold"],
                                sp.NEEDLE_ANISO_RATIO)
        self.assertGreater(thr["needle_size_threshold"], 0.0)

    def test_disabled_by_default(self):
        xyz, v = self._scene()
        flags, thr = sp.build_flags(xyz, v, knn_k=5, iso_percentile=99.0,
                                    size_percentile=99.9,
                                    body_percentile_low=0.5,
                                    body_percentile_high=99.5,
                                    body_margin=5.0)
        self.assertFalse(thr["needle_enabled"])
        self.assertIsNone(thr["needle_aniso_threshold"])
        self.assertFalse(flags["needle"].any())

    def test_needle_only_exact_match(self):
        xyz, v = self._scene()
        flags, _ = self._flags(xyz, v, True)
        # no near-isotropic normal may be mistaken for a needle
        self.assertEqual(int(flags["needle"].sum()), self.N_NEEDLE)

    def test_primary_reason_needle_beats_size(self):
        flags = {"outside_body": np.array([False, False, False]),
                 "isolated": np.array([False, False, True]),
                 "size_outlier": np.array([True, True, True]),
                 "needle": np.array([True, False, False])}
        low = np.array([False, False, False])
        r = sp.primary_reason(flags, low)
        np.testing.assert_array_equal(
            r, [sp.REASON_NEEDLE, sp.REASON_SIZE, sp.REASON_ISOLATED])

    def test_prune_with_needle_cleanup_reports_counts(self):
        xyz, v = self._scene()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(v, "vertex")],
                    text=False, byte_order="<",
                    comments=["Brush export test"]).write(str(src))
            opts = {"max_fraction": 0.3, "voxel_size": 10.0,
                    "importance_threshold": 1e-4, "knn_k": 5,
                    "iso_percentile": 99.0, "size_percentile": 99.9,
                    "body_low": 0.5, "body_high": 99.5, "body_margin": 5.0,
                    "singleton": False, "needle": True}
            info = sp.prune(src, dst, opts)
        self.assertEqual(info["flagged_counts"]["needle"], self.N_NEEDLE)
        self.assertTrue(info["thresholds"]["needle_enabled"])
        removed = info["removed_by_reason"]["needle"]
        blocked = info["blocked_by_safety"]["needle"]
        self.assertGreaterEqual(removed, 1)       # voxel budget allows >= 1
        self.assertEqual(removed + blocked, self.N_NEEDLE)
        self.assertEqual(info["voxels_emptied"], 0)

    def test_prune_without_needle_key_still_works(self):
        # pipeline.py builds prune_opts without "needle" (Task 5 wires it);
        # the key must stay optional.
        xyz, v = self._scene()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(v, "vertex")],
                    text=False, byte_order="<").write(str(src))
            opts = {"max_fraction": 0.3, "voxel_size": 10.0,
                    "importance_threshold": 1e-4, "knn_k": 5,
                    "iso_percentile": 99.0, "size_percentile": 99.9,
                    "body_low": 0.5, "body_high": 99.5, "body_margin": 5.0,
                    "singleton": False}
            info = sp.prune(src, dst, opts)
        self.assertFalse(info["thresholds"]["needle_enabled"])
        self.assertEqual(info["flagged_counts"]["needle"], 0)


class TestOutputFormat(unittest.TestCase):
    def test_output_preserves_property_order_and_comments(self):
        v = make_vertex_array(50, seed=7, scrambled=True)
        comments = ["Brush export test", "comment two"]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            dst = Path(td) / "out.ply"
            PlyData([PlyElement.describe(v, "vertex")],
                    text=False, byte_order="<", comments=comments).write(str(src))

            opts = {"max_fraction": 0.3, "voxel_size": 5.0,
                    "importance_threshold": 1e-4, "knn_k": 3,
                    "iso_percentile": 99.0, "size_percentile": 99.9,
                    "body_low": 0.5, "body_high": 99.5, "body_margin": 20.0,
                    "singleton": False}
            info = sp.prune(src, dst, opts)

            # mmap=None: plyfile's default mmap='c' keeps the file locked
            # on Windows, which would block TemporaryDirectory cleanup.
            out = PlyData.read(str(dst), mmap=None)
            self.assertEqual(list(out["vertex"].data.dtype.names),
                             list(v.dtype.names))
            self.assertEqual(list(out.comments), comments)
            self.assertEqual(info["pruned_count"], len(out["vertex"].data))
            self.assertGreater(info["original_count"], 0)
            self.assertEqual(info["voxels_emptied"], 0)
            self.assertLessEqual(info["removed_count"], info["original_count"])


if __name__ == "__main__":
    unittest.main()
