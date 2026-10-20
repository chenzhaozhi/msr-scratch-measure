"""纯 numpy 合成数据回归测试；mmind 模块仅替代脚本已使用的字段。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class Plane:
    def __init__(self, a=0, b=0, c=1, d=-10):
        self.a, self.b, self.c, self.d = a, b, c, d


def load_measurement():
    surface = ModuleType("mmind_surface")
    surface.PySurfaceData = object
    feature = ModuleType("mmind_feature")
    feature.PyPlane3D = Plane
    modules = {"mmind_surface": surface, "mmind_feature": feature,
               "mmind_util": ModuleType("mmind_util")}
    path = Path(__file__).with_name("measure_scratch_v1.1.py")
    spec = importlib.util.spec_from_file_location("scratch_v11", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


measurement = load_measurement()


def surface(depths=None, y_resolution=0.25):
    if depths is None:
        depths = np.tile([2., 3., 4., 3., 2.], (41, 1))
    return SimpleNamespace(depth=-np.asarray(depths, dtype=float),
                           xResolution=0.1, yResolution=y_resolution,
                           xOffset=0., yOffset=0., zOffset=10.)


class ScratchTests(unittest.TestCase):
    def measure(self, surf=None, plane=None, **kwargs):
        options = dict(trim_ratio=0, depth_limit=20, width_limit=20)
        options.update(kwargs)
        return measurement.measure_scratch(
            surface() if surf is None else surf,
            Plane() if plane is None else plane, **options)

    def test_ports_and_known_straight_groove(self):
        result = self.measure()
        self.assertEqual(len(result), 7)
        np.testing.assert_allclose(result[:5], [4, 4, .4, .4, 10])
        self.assertIs(result[5], True)

    def test_plane_scaling_and_normal_reversal(self):
        expected = self.measure()[:6]
        for plane in [Plane(c=7, d=-70), Plane(c=-3, d=30)]:
            np.testing.assert_allclose(self.measure(plane=plane)[:6], expected)

    def test_projection_removes_depth_from_axes(self):
        depths = np.tile((2 + 5 * (np.arange(41) % 2))[:, None], (1, 5))
        result = self.measure(surface(depths))
        np.testing.assert_allclose(result[2:5], [.4, .4, 10], atol=1e-10)

    def test_last_endpoint_included(self):
        depths = np.full((11, 3), 2.)
        depths[-1] = 9
        result = self.measure(surface(depths, .1), section_step=1)
        self.assertEqual(result[0], 9)

    def test_depth_survives_missing_width(self):
        depths = np.tile([1., 1., 6., 1., 1.], (41, 1))
        with self.assertRaisesRegex(ValueError, "宽度缺测.*深度 max=6.0000mm"):
            self.measure(surface(depths))

    def test_min_points_one_does_not_allow_single_transverse_location(self):
        depths = np.tile([1., 1., 6., 1., 1.], (41, 1))
        with self.assertRaisesRegex(ValueError, "宽度缺测"):
            self.measure(surface(depths), min_points=1)

    def test_deep_narrow_section_not_dropped_when_other_widths_exist(self):
        depths = np.full((41, 5), 2.)
        depths[20:24] = [1, 1, 8, 1, 1]
        result = self.measure(surface(depths))
        self.assertEqual(result[0], 8)
        self.assertTrue(np.isfinite(result[2]))
        self.assertIs(result[5], False)

    def test_trim_preserves_contract_and_exposes_raw_extreme(self):
        depths = np.full((41, 5), 2.)
        depths[0] = 9
        result = self.measure(surface(depths), trim_ratio=.1)
        self.assertEqual(result[0], 2)
        self.assertAlmostEqual(result[4], 8)
        self.assertIn("可能隐藏真实缺陷", result[6])
        self.assertIn("全带投影跨度=10.00mm", result[6])
        self.assertIn("全带单点最大距离=9.0000mm", result[6])

    def test_trim_fallback_reports_actual_point_count(self):
        result = self.measure(surface(np.full((4, 3), 2.)), trim_ratio=.1)
        self.assertIn("已回退不剔除", result[6])
        self.assertIn("有效点=12(端部剔除0%)", result[6])

    def test_invalid_parameters_rejected(self):
        for option in [dict(section_step=0), dict(section_step=np.nan),
                       dict(min_points=1.5), dict(min_points=np.inf),
                       dict(trim_ratio=np.nan), dict(trim_ratio=.5),
                       dict(depth_limit=-1), dict(width_limit=np.inf)]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.measure(**option)

    def test_invalid_planes_rejected(self):
        for plane in [Plane(c=0), Plane(c=np.nan), Plane(d=np.inf)]:
            with self.subTest(plane=plane), self.assertRaises(ValueError):
                self.measure(plane=plane)

    def test_xyz_filter_retains_existing_z_sentinel_semantics(self):
        surf = surface([[10, 2, np.nan, np.inf]])
        xyz = measurement._surface_to_xyz(surf)
        np.testing.assert_allclose(xyz, [[.1, 0, 8]])
        surf.xResolution = np.nan
        self.assertEqual(len(measurement._surface_to_xyz(surf)), 0)

    def test_nearly_round_cloud_warns(self):
        result = self.measure(surface(np.full((11, 11), 2.), .1))
        self.assertIn("PCA长宽比小于3", result[6])

    def test_equal_section_mean_not_point_weighted(self):
        depths = np.full((4, 3), 2.)
        depths[2:] = 4
        depths[3, [0, 2]] = np.nan
        result = self.measure(surface(depths), section_step=.5, min_points=2)
        self.assertEqual(result[1], 3)


if __name__ == "__main__":
    unittest.main()
