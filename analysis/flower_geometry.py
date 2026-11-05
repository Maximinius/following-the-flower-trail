"""Flower trail geometry and arc-length segmentation.

Reconstructs the inner and outer boundary of the flower trail from the SVG path
used by the experiment (scaled by 2.5 and centred on a 1920 x 1080 display) and
measures the curvature of arc-length segments of the inner boundary.
"""
import re

import numpy as np
from shapely.geometry import Polygon

SCREEN_RES = (1920, 1080)
SCALE_FACTOR = 2.5
CORRIDOR_OFFSET = 10
N_SEGMENTS = 20

_FLOWER_INNER_PATH_DATA = (
    "M304.1,555.6c0-7.1,0-14.3,0-21.6c21.1-3.7,33.5-15.7,35.1-37.7c5.2-2.5,10.7-4.3,15.7-7.1"
    "c10.3-5.6,19.2-12.9,26.7-21.8c9.5-11.3,15.6-24.3,18.6-38.6c1.6-7.6,1.8-15.3,1.4-23.1"
    "c-0.5-11.3-3.7-21.8-8.3-32c-2.4-5.3-5.4-10.4-9.2-14.9c-2.4-2.8-3-5.7-2.7-9.2"
    "c1.2-12.9,3.1-39.6,3-40.8c-14.4,9-27.8,18.7-42.4,28c-4.8-15.7-8-31.4-13-47.5"
    "c-9.9,13.6-19.4,26.6-29.2,40.1c-9.6-13.4-18.9-26.1-28.5-39.5c-4.7,15.9-7.9,31.3-13,47"
    "c-14.2-9.2-27.2-19.3-41.1-28.5c0.1,2.3,3,37,3.4,42.2c0.2,2.1-0.4,3.8-1.8,5.4"
    "c-8.1,9.8-13.9,20.7-17,32.4c-1.9,7-2.9,14.4-2.9,22c0,10.2,1.4,19.8,4.4,28.9"
    "c3.9,11.7,10.2,22.4,19.2,32c8.3,9,18.1,16,29.1,21.2c3.5,1.6,7.1,2.8,10.9,4.2"
    "c2.2,21.7,14.6,33.8,36,37.2c0,7.1,0,14.2,0,21.2"
)


def _extract_path_data(path_data):
    """Sample the SVG path (relative move, line and cubic Bezier commands) into vertices."""
    commands = re.findall(r'([MmZzLlHhVvCcSsQqTtAa])|(-?[0-9]*\.?[0-9]+)', path_data)
    coords = []
    pos = (0.0, 0.0)
    i = 0
    while i < len(commands):
        cmd = commands[i][0]
        if cmd in 'Zz':
            i += 1
        elif cmd in 'Mm':
            pos = (float(commands[i + 1][1]), float(commands[i + 2][1]))
            coords.append(pos)
            i += 3
        elif cmd in 'Ll':
            pos = (pos[0] + float(commands[i + 1][1]), pos[1] + float(commands[i + 2][1]))
            coords.append(pos)
            i += 3
        elif cmd in 'Cc':
            c1 = (pos[0] + float(commands[i + 1][1]), pos[1] + float(commands[i + 2][1]))
            c2 = (pos[0] + float(commands[i + 3][1]), pos[1] + float(commands[i + 4][1]))
            ep = (pos[0] + float(commands[i + 5][1]), pos[1] + float(commands[i + 6][1]))
            for t in np.linspace(0, 1, 20):
                x = ((1 - t) ** 3 * pos[0] + 3 * (1 - t) ** 2 * t * c1[0]
                     + 3 * (1 - t) * t ** 2 * c2[0] + t ** 3 * ep[0])
                y = ((1 - t) ** 3 * pos[1] + 3 * (1 - t) ** 2 * t * c1[1]
                     + 3 * (1 - t) * t ** 2 * c2[1] + t ** 3 * ep[1])
                coords.append((x, y))
            pos = ep
            i += 7
        else:
            i += 1
    return coords


def _reduce_redundant_points(vertices, threshold=0.1):
    reduced = []
    for v in vertices:
        if (not reduced
                or abs(v[0] - reduced[-1][0]) > threshold
                or abs(v[1] - reduced[-1][1]) > threshold):
            reduced.append(v)
    return reduced


def _scale_vertices(vertices, scale_factor, centroid):
    cx, cy = centroid
    hw, hh = SCREEN_RES[0] // 2, SCREEN_RES[1] // 2
    return [((v[0] - cx) * scale_factor + hw, (v[1] - cy) * scale_factor + hh)
            for v in vertices]


def build_flower_geometry():
    """Return (inner, outer) boundary vertex lists in screen coordinates."""
    verts = _extract_path_data(_FLOWER_INNER_PATH_DATA)
    verts.append((304.1, 555.0999999999999))
    verts = _reduce_redundant_points(verts)
    centroid = (sum(v[0] for v in verts) / len(verts), sum(v[1] for v in verts) / len(verts))
    inner_scaled = _scale_vertices(verts, SCALE_FACTOR, centroid)
    outer_raw = list(Polygon(verts).buffer(CORRIDOR_OFFSET, join_style=2).exterior.coords)
    outer_scaled = _scale_vertices(outer_raw, SCALE_FACTOR, centroid)
    stalk_y = max(v[1] for v in inner_scaled)
    outer_scaled = [(x, min(y, stalk_y)) for x, y in outer_scaled]
    return inner_scaled, outer_scaled


def compute_arc_lengths(path):
    """Cumulative arc length at each vertex."""
    pts = np.array(path, dtype=float)
    seg_len = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def segment_curvature(path_arr, cumlen, n_segments):
    """Mean absolute turning angle per unit arc length in each segment."""
    boundaries = np.linspace(0.0, cumlen[-1], n_segments + 1)
    d = np.diff(path_arr, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.abs(np.diff(ang))
    dang = np.minimum(dang, 2 * np.pi - dang)
    vertex_arc = cumlen[1:-1]
    seg_of_vertex = np.clip(np.searchsorted(boundaries[1:], vertex_arc), 0, n_segments - 1)
    seg_len = np.diff(boundaries)
    return np.array([dang[seg_of_vertex == s].sum() / seg_len[s] for s in range(n_segments)])
