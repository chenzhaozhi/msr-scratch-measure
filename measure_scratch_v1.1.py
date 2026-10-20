# -*- coding: utf-8 -*-
"""
划痕深度/宽度测量（Mech-MSR V2.2.1「使用Python计算结果」步骤）
v1.1  2026-10-20  补智（chenzhaozhi 的 AI 助手）
  基于 v1.0 用户实测反馈：
  - 最深出现在 s=0 端部，疑端部塌边/起弧假深度 → 新增 trim_ratio 端部剔除开关
  - report 截图看不清剖面数 → 新增逐剖面深度列表，端部异常一眼可见

== MSR 端口配置 ==
输入（顺序必须一致）：
  1. Surface  scratch_surface   划痕带表面（表面预处理"平面距离过滤"的输出）
  2. Shape3D  reference_plane   拟合平面（表面拟合平面步骤的"拟合平面"输出）
  3. Number   section_step      剖面间距 mm（默认 0.5）
  4. Number   min_points        单剖面最少点数（默认 3）
  5. Number   depth_limit       深度上限 mm，超了判 NG（默认 0.1）
  6. Number   width_limit       宽度上限 mm，超了判 NG（默认 1.0）
  7. Number   trim_ratio        端部剔除比例 0~0.4（默认 0.1，即首尾各剔除 10% 走向长度）
输出（顺序必须一致）：
  1. Number depth_max    最大深度 mm（剔除端部后）
  2. Number depth_mean   平均深度 mm
  3. Number width_max    最大宽度 mm（半深全宽 FWHM）
  4. Number width_mean   平均宽度 mm
  5. Number length_mm    划痕长度 mm（剔除端部后）
  6. Bool   is_ok        深度、宽度都不超上限为 True
  7. String report       测量报告（含逐剖面深度列表与警告）
调用函数名：measure_scratch
"""

import numpy as np
from mmind_surface import PySurfaceData
from mmind_util import *
from mmind_feature import *


def _surface_to_xyz(surf):
    """Surface 深度图 → XYZ 点云（mm），只保留有效点。"""
    depth = np.asarray(surf.depth, dtype=np.float64)
    rows, cols = depth.shape
    col_grid, row_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    x = col_grid * surf.xResolution + surf.xOffset
    y = row_grid * surf.yResolution + surf.yOffset
    z = depth + surf.zOffset
    valid = np.isfinite(z) & (z != 0.0)
    return np.column_stack((x[valid], y[valid], z[valid]))


def _extract_plane(shape3d_input):
    """Shape3D 输入里取 PyPlane3D（Shape3D 可能是混合列表，逐个找）。"""
    items = shape3d_input if isinstance(shape3d_input, (list, tuple)) else [shape3d_input]
    for s in items:
        if isinstance(s, PyPlane3D):
            return s
    raise ValueError("参考平面端口里没有 PyPlane3D，实际收到: %s"
                     % [type(s).__name__ for s in items])


def measure_scratch(scratch_surface, reference_plane,
                    section_step=0.5, min_points=3,
                    depth_limit=0.1, width_limit=1.0,
                    trim_ratio=0.1):
    # ---------- 0. 端部剔除参数检查 ----------
    trim_ratio = float(trim_ratio)
    if trim_ratio < 0.0:
        trim_ratio = 0.0
    if trim_ratio > 0.4:
        trim_ratio = 0.4
    if section_step <= 0:
        section_step = 0.5
    if min_points < 1:
        min_points = 1

    # ---------- 1. 基准平面：ax+by+cz+d=0，法向量单位化 ----------
    plane = _extract_plane(reference_plane)
    n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
    n /= np.linalg.norm(n)

    # ---------- 2. Surface → 划痕点云 ----------
    pts = _surface_to_xyz(scratch_surface)
    if pts.shape[0] < 10:
        raise ValueError("划痕表面有效点太少 (%d)，检查平面距离过滤窗口" % pts.shape[0])

    # ---------- 3. 相对高度 h：平面为零，划痕一侧为正 ----------
    p0 = -plane.d * n                     # 平面上一点
    h = (pts - p0) @ n                    # 带符号距离
    if np.median(h) < 0:                  # 统一翻到划痕在正侧
        h = -h
    warn = ""
    bad_ratio = float(np.mean(h <= 0))
    if bad_ratio > 0.2:
        warn = "警告:%.0f%%的点不在划痕一侧,检查距离过滤窗口 | " % (bad_ratio * 100)

    # ---------- 4. PCA 定划痕走向/横向 ----------
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis_len = vt[0]                      # 第一主方向 = 划痕走向
    axis_wid = vt[1]                      # 第二主方向 = 横向
    s_along = (pts - c) @ axis_len
    t_later = (pts - c) @ axis_wid

    # ---------- 5. 端部剔除（去塌边/起弧假深度） ----------
    lo_raw, hi_raw = float(s_along.min()), float(s_along.max())
    span_raw = hi_raw - lo_raw
    lo = lo_raw + span_raw * trim_ratio
    hi = hi_raw - span_raw * trim_ratio
    in_mid = (s_along >= lo) & (s_along <= hi)
    pts_mid = int(in_mid.sum())
    if pts_mid < 20:
        # 剔除后点太少，自动回退到不剔除
        trim_ratio = 0.0
        lo, hi = lo_raw, hi_raw
        in_mid = np.ones_like(s_along, dtype=bool)
        warn += "警告:端部剔除后点数不足,已回退不剔除 | "

    # ---------- 6. 分剖面测深度与半深全宽 ----------
    bins = np.arange(lo, hi + section_step, section_step)
    depths, widths, centers = [], [], []
    for i in range(len(bins) - 1):
        m = in_mid & (s_along >= bins[i]) & (s_along < bins[i + 1])
        if int(m.sum()) < int(min_points):
            continue
        hh = h[m]
        tt = t_later[m]
        d_sec = float(np.percentile(hh, 98))      # 98 分位抗单点飞点
        if d_sec <= 0:
            continue
        deep = hh >= 0.5 * d_sec                   # 半深以上的点 → FWHM 宽度
        if int(deep.sum()) < int(min_points):
            continue
        w_sec = float(tt[deep].max() - tt[deep].min())
        depths.append(d_sec)
        widths.append(w_sec)
        centers.append((bins[i] + bins[i + 1]) / 2.0 - lo)   # 从 0 计,便于对读

    if not depths:
        raise ValueError("没有有效剖面：调大剖面间距或减小单剖面最少点数")

    depths = np.array(depths)
    widths = np.array(widths)
    depth_max = float(depths.max())
    depth_mean = float(depths.mean())
    width_max = float(widths.max())
    width_mean = float(widths.mean())
    length_mm = float(hi - lo)
    is_ok = bool(depth_max <= depth_limit and width_max <= width_limit)

    imax = int(np.argmax(depths))
    # 逐剖面深度列表：s:深度，紧凑串，端部异常一眼可见
    sec_list = ",".join("%.1f:%.3f" % (centers[i], depths[i]) for i in range(len(depths)))
    if len(sec_list) > 480:
        sec_list = sec_list[:480] + "..."

    report = (warn
              + "剖面数=%d | 有效点=%d(端部剔除%.0f%%) | 深度 max=%.4fmm(均值%.4f) @s=%.2fmm | "
                "宽度 max=%.4fmm(均值%.4f,半深全宽) | 长度=%.2fmm | %s | 逐剖面深度: %s"
              % (len(depths), pts_mid, trim_ratio * 100,
                 depth_max, depth_mean, centers[imax],
                 width_max, width_mean, length_mm,
                 "OK" if is_ok else "NG", sec_list))
    return depth_max, depth_mean, width_max, width_mean, length_mm, is_ok, report
