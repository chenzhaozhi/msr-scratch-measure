# -*- coding: utf-8 -*-
"""
划痕深度/宽度测量（Mech-MSR V2.2.1「使用Python计算结果」步骤）
v1.1  2026-10-20  补智（chenzhaozhi 的 AI 助手）
  基于 v1.0 用户实测反馈：
  - 最深出现在 s=0 端部，疑端部塌边/起弧假深度 → 新增 trim_ratio 端部剔除开关
  - report 截图看不清剖面数 → 新增逐剖面深度列表，端部异常一眼可见
  - 平面系数同步归一化；在基准平面内 PCA，避免深度混入宽度
  - 深度不再因半深点不足而丢弃；部分宽度缺测时 is_ok=False，全部缺测则报错

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
  3. Number width_max    最大宽度 mm（采样半深全宽 FWHM）
  4. Number width_mean   平均宽度 mm
  5. Number length_mm    划痕长度 mm（剔除端部后）
  6. Bool   is_ok        有效深度剖面均有宽度且中央区域深度、宽度不超上限为 True
  7. String report       测量报告（含逐剖面深度列表与警告）
调用函数名：measure_scratch

适用范围：单条近似直线槽；宽度为采样点半深跨度，不是槽口宽度。
前置 [-5,-2]mm 距离窗口若切掉半深边界，本脚本无法恢复真实 FWHM。
98 分位在少点剖面上接近最大值，不保证剔除飞点；不自动用 MAD
删除深点，以免把真实 V 型槽底当成异常。均值为有效剖面等权平均，
不是点云均值；缺失剖面及末段不足一个步长会影响长度平均的近似。
trim_ratio 是测量区域约定，不是端部伪影检测；长度为裁剪后的平面内
主轴跨度，不是曲线弧长。完整划痕的验收需结合未裁剪结果与原始数据。
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
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z != 0.0)
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
    # ---------- 0. 参数检查 ----------
    section_step = float(section_step)
    min_points = float(min_points)
    depth_limit = float(depth_limit)
    width_limit = float(width_limit)
    trim_ratio = float(trim_ratio)
    if not np.all(np.isfinite([section_step, min_points, depth_limit,
                               width_limit, trim_ratio])):
        raise ValueError("测量参数必须为有限数值")
    if section_step <= 0 or min_points < 1 or min_points != int(min_points):
        raise ValueError("剖面间距必须大于零，单剖面最少点数必须为正整数")
    if depth_limit < 0 or width_limit < 0 or not 0 <= trim_ratio <= 0.4:
        raise ValueError("深度/宽度上限不能为负，端部剔除比例必须在 0~0.4")
    min_points = int(min_points)

    # ---------- 1. 基准平面：ax+by+cz+d=0，法向量单位化 ----------
    plane = _extract_plane(reference_plane)
    n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
    norm = np.linalg.norm(n)
    if not np.all(np.isfinite(n)) or not np.isfinite(plane.d) or not np.isfinite(norm) or norm == 0:
        raise ValueError("参考平面系数必须有限且法向量非零")
    n /= norm
    d = plane.d / norm
    if not np.isfinite(d):
        raise ValueError("参考平面归一化后的常数项必须有限")

    # ---------- 2. Surface → 划痕点云 ----------
    pts = _surface_to_xyz(scratch_surface)
    if pts.shape[0] < 10:
        raise ValueError("划痕表面有效点太少 (%d)，检查平面距离过滤窗口" % pts.shape[0])

    # ---------- 3. 相对高度 h：平面为零，划痕一侧为正 ----------
    h = pts @ n + d                       # 带符号距离，d 与法向量同步归一化
    if not np.all(np.isfinite(h)):
        raise ValueError("平面距离非有限值，检查输入坐标及平面系数")
    if np.median(h) < 0:                  # 统一翻到划痕在正侧
        h = -h
    warn = ""
    bad_ratio = float(np.mean(h <= 0))
    if bad_ratio > 0.2:
        warn = "警告:%.0f%%的点不在划痕一侧,检查距离过滤窗口 | " % (bad_ratio * 100)

    # ---------- 4. 在基准平面内 PCA 定划痕走向/横向 ----------
    c = pts.mean(axis=0)
    centered = pts - c
    projected = centered - np.outer(centered @ n, n)
    _, singular, vt = np.linalg.svd(projected, full_matrices=False)
    if singular[0] <= 0:
        raise ValueError("划痕在基准平面内没有可测跨度")
    axis_len = vt[0]                      # 第一主方向 = 划痕走向
    axis_wid = np.cross(n, axis_len)       # 横向严格位于基准平面内
    axis_wid /= np.linalg.norm(axis_wid)
    s_along = projected @ axis_len
    t_later = projected @ axis_wid
    if singular[1] > 0 and singular[0] / singular[1] < 3:
        warn += "警告:PCA长宽比小于3(经验阈值),走向可能不稳定;仅适用单条近直线槽 | "

    # ---------- 5. 端部剔除（测量区域约定，不自动认定端部为伪影） ----------
    lo_raw, hi_raw = float(s_along.min()), float(s_along.max())
    span_raw = hi_raw - lo_raw
    lo = lo_raw + span_raw * trim_ratio
    hi = hi_raw - span_raw * trim_ratio
    in_mid = (s_along >= lo) & (s_along <= hi)
    pts_mid = int(in_mid.sum())
    if trim_ratio > 0 and pts_mid < 20:
        # 剔除后点太少，自动回退到不剔除
        trim_ratio = 0.0
        lo, hi = lo_raw, hi_raw
        in_mid = np.ones_like(s_along, dtype=bool)
        pts_mid = int(in_mid.sum())
        warn += "警告:端部剔除后点数不足,已回退不剔除 | "
    if trim_ratio > 0:
        warn += "警告:端部剔除可能隐藏真实缺陷,OK仅代表中央区域 | "

    # ---------- 6. 分剖面测深度与半深全宽 ----------
    section_count = max(1, int(np.ceil((hi - lo) / section_step)))
    depths, widths, centers = [], [], []
    missing_widths, sparse_sections, small_sections = 0, 0, 0
    for i in range(section_count):
        left = lo + i * section_step
        right = min(lo + (i + 1) * section_step, hi)
        upper = s_along <= hi if i == section_count - 1 else s_along < right
        m = in_mid & (s_along >= left) & upper
        count = int(m.sum())
        if count < min_points:
            sparse_sections += 1
            continue
        hh = h[m]
        tt = t_later[m]
        d_sec = float(np.percentile(hh, 98))      # 少点剖面仍接近最大值
        if d_sec <= 0:
            continue
        depths.append(d_sec)
        centers.append((left + right) / 2.0 - lo)
        if count < 50:
            small_sections += 1
        deep = hh >= 0.5 * d_sec                   # 半深以上的点 → FWHM 宽度
        if int(deep.sum()) < max(2, min_points):
            missing_widths += 1
            continue
        w_sec = float(tt[deep].max() - tt[deep].min())
        # 同一横向位置的多个纵向采样点不能证明宽度为零。
        if w_sec <= 8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(tt)))):
            missing_widths += 1
            continue
        widths.append(w_sec)

    if not depths:
        raise ValueError("没有有效剖面：调大剖面间距或减小单剖面最少点数")

    depths = np.array(depths)
    widths = np.array(widths)
    depth_max = float(depths.max())
    depth_mean = float(depths.mean())
    if not widths.size:
        raise ValueError("全部深度剖面宽度缺测，不能判OK：深度 max=%.4fmm(均值%.4f)，"
                         "检查横向采样及半深点数" % (depth_max, depth_mean))
    width_max = float(widths.max())
    width_mean = float(widths.mean())
    length_mm = float(hi - lo)
    is_ok = bool(missing_widths == 0 and depth_max <= depth_limit and width_max <= width_limit)
    if missing_widths:
        warn += "警告:%d个深度剖面宽度缺测,判NG;宽度仅统计可测剖面 | " % missing_widths
    if sparse_sections:
        warn += "警告:%d个剖面点数不足,未参与统计 | " % sparse_sections
    if small_sections:
        warn += "警告:%d个剖面少于50点,98分位不能可靠抗单点飞点 | " % small_sections

    imax = int(np.argmax(depths))
    # 逐剖面深度列表：s:深度，紧凑串，端部异常一眼可见
    sec_list = ",".join("%.1f:%.3f" % (centers[i], depths[i]) for i in range(len(depths)))
    if len(sec_list) > 480:
        sec_list = sec_list[:480] + "..."

    report = (warn
              + "剖面数=%d | 有效点=%d(端部剔除%.0f%%) | 深度 max=%.4fmm(均值%.4f) @s=%.2fmm | "
                "宽度 max=%.4fmm(均值%.4f,采样半深跨度,非槽口宽度) | 长度=%.2fmm | "
                "全带投影跨度=%.2fmm,全带单点最大距离=%.4fmm(非剖面98分位) | "
                "%s | 逐剖面深度(s自裁剪起点): %s"
              % (len(depths), pts_mid, trim_ratio * 100,
                 depth_max, depth_mean, centers[imax],
                 width_max, width_mean, length_mm, span_raw, float(h.max()),
                 "OK" if is_ok else "NG", sec_list))
    return depth_max, depth_mean, width_max, width_mean, length_mm, is_ok, report
