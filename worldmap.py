"""Static map model: rasterized ground/buildings, line of sight, and a nav graph with Dijkstra flow fields.

Everything is precomputed once (build time if possible, else at hello) and cached as .npz in /tmp or next to this file.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

R = 2.0  # grid resolution (m)
HEAD = 9.0
STEP_UP = 1.2


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _poly_mask(xs, ys, ring):
    """Even-odd point in polygon for arrays of points xs, ys (same shape). ring is flat [x0,y0,...]."""
    px = ring[0::2]
    py = ring[1::2]
    n = len(px)
    inside = np.zeros(xs.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        if yi != yj:
            cond = ((yi > ys) != (yj > ys))
            xint = (xj - xi) * (ys - yi) / (yj - yi) + xi
            inside ^= cond & (xs < xint)
        j = i
    return inside


def build(arena_dir: str):
    t0 = time.time()
    w = json.load(open(os.path.join(arena_dir, "v2", "world.json")))
    tr = w["terrain"]
    nx, ny = tr["nx"], tr["ny"]
    raw = np.fromfile(os.path.join(arena_dir, "v2", tr["file"]), dtype="<i2").reshape(ny, nx).astype(np.float32) * tr.get("scale", 0.01)
    minx, miny, step = tr["minX"], tr["minY"], tr["step"]
    b = w["bounds"]
    gx0, gy0 = b["minX"], b["minY"]
    gnx = int(math.ceil((b["maxX"] - gx0) / R))
    gny = int(math.ceil((b["maxY"] - gy0) / R))
    cx = gx0 + (np.arange(gnx) + 0.5) * R
    cy = gy0 + (np.arange(gny) + 0.5) * R
    XX, YY = np.meshgrid(cx, cy)  # shape (gny, gnx)
    # terrain sample (bilinear)
    fi = np.clip((XX - minx) / step, 0, nx - 1.001)
    fj = np.clip((YY - miny) / step, 0, ny - 1.001)
    i0 = fi.astype(int); j0 = fj.astype(int)
    di = fi - i0; dj = fj - j0
    g = (raw[j0, i0] * (1 - di) * (1 - dj) + raw[j0, i0 + 1] * di * (1 - dj)
         + raw[j0 + 1, i0] * (1 - di) * dj + raw[j0 + 1, i0 + 1] * di * dj).astype(np.float32)
    # solids: for each cell, the lowest-base solid that is relevant to a ground walker / LOS
    sbase = np.full((gny, gnx), 1e4, np.float32)   # base of solid column at cell
    stop = np.full((gny, gnx), -1e4, np.float32)   # top of solid column at cell
    roof = np.full((gny, gnx), -1e4, np.float32)   # highest top
    for p in w["prisms"]:
        ring = p["ring"]
        xs = ring[0::2]; ys = ring[1::2]
        i_lo = max(0, int((min(xs) - gx0) / R) - 1); i_hi = min(gnx, int((max(xs) - gx0) / R) + 2)
        j_lo = max(0, int((min(ys) - gy0) / R) - 1); j_hi = min(gny, int((max(ys) - gy0) / R) + 2)
        if i_hi <= i_lo or j_hi <= j_lo:
            continue
        sx = XX[j_lo:j_hi, i_lo:i_hi]; sy = YY[j_lo:j_hi, i_lo:i_hi]
        m = _poly_mask(sx, sy, ring)
        for h in p.get("holes") or []:
            if len(h) >= 6:
                m &= ~_poly_mask(sx, sy, h)
        if not m.any():
            continue
        base, top = float(p["base"]), float(p["top"])
        sub_b = sbase[j_lo:j_hi, i_lo:i_hi]; sub_t = stop[j_lo:j_hi, i_lo:i_hi]; sub_r = roof[j_lo:j_hi, i_lo:i_hi]
        sub_b[m] = np.minimum(sub_b[m], base)
        sub_t[m] = np.maximum(sub_t[m], top)
        sub_r[m] = np.maximum(sub_r[m], top)
    # walk blocked: a solid between ground+step_up and ground+head
    gnd = g
    blocked = (stop > gnd + STEP_UP) & (sbase < gnd + HEAD)
    # slope check: terrain gradient
    gy_, gx_ = np.gradient(gnd, R)
    slope = np.sqrt(gx_ ** 2 + gy_ ** 2)
    steep = slope > math.tan(math.radians(46))
    # surface height for walkers (ground, or top of low solid that we step on)
    surf = np.where((stop > -1e3) & (stop <= gnd + STEP_UP) & (sbase < gnd + HEAD), np.maximum(gnd, stop), gnd)
    # edge margin
    hard = blocked | steep
    edge = np.zeros_like(hard)
    edge[:, :8] = True; edge[:, -8:] = True; edge[:8, :] = True; edge[-8:, :] = True
    hard |= edge
    # clearance: distance (cells) to nearest hard cell
    from scipy import ndimage
    dist_hard = ndimage.distance_transform_edt(~hard) * R
    # trees: crown blocking (for vision) as an extra opaque height band
    tf = w["trees"]["fields"]; td = w["trees"]["data"]; nf = len(tf)
    tree = np.zeros((gny, gnx), np.float32)  # canopy density factor
    ix = tf.index("x"); iy = tf.index("y"); icr = tf.index("crownR")
    icx = tf.index("crownX") if "crownX" in tf else ix
    icy = tf.index("crownY") if "crownY" in tf else iy
    for k in range(0, len(td), nf):
        x = td[k + icx] if td[k + icx] is not None else td[k + ix]
        y = td[k + icy] if td[k + icy] is not None else td[k + iy]
        r = td[k + icr] or 3
        i = int((x - gx0) / R); j = int((y - gy0) / R); rr = int(r / R) + 1
        for jj in range(max(0, j - rr), min(gny, j + rr + 1)):
            for ii in range(max(0, i - rr), min(gnx, i + rr + 1)):
                if (cx[ii] - x) ** 2 + (cy[jj] - y) ** 2 <= r * r:
                    tree[jj, ii] = 1
    log(f"map built in {time.time() - t0:.1f}s grid {gnx}x{gny}")
    return dict(gx0=np.float64(gx0), gy0=np.float64(gy0), gnx=np.int64(gnx), gny=np.int64(gny), gnd=gnd, surf=surf.astype(np.float32),
                sbase=sbase, stop=stop, roof=roof, hard=hard, steep=steep, clear=dist_hard.astype(np.float32), tree=tree)


class WorldMap:
    def __init__(self, arena_dir: str):
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [os.path.join(here, "mapcache.npz"), "/tmp/mapcache.npz"]
        d = None
        for c in cands:
            if os.path.exists(c):
                try:
                    z = np.load(c)
                    d = {k: z[k] for k in z.files}
                    break
                except Exception as e:  # noqa
                    log("cache load failed", c, e)
        if d is None:
            d = build(arena_dir)
            try:
                np.savez("/tmp/mapcache.npz", **d)
            except Exception:
                pass
        self.gx0 = float(d["gx0"]); self.gy0 = float(d["gy0"])
        self.nx = int(d["gnx"]); self.ny = int(d["gny"])
        self.gnd = d["gnd"]; self.surf = d["surf"]; self.sbase = d["sbase"]; self.stop = d["stop"]
        self.hard = d["hard"].astype(bool); self.clear = d["clear"]; self.tree = d["tree"]
        self.cover = None
        self._graph = None
        self._build_graph()

    # --- coordinates
    def cell(self, x, y):
        i = int((x - self.gx0) / R); j = int((y - self.gy0) / R)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def center(self, i, j):
        return self.gx0 + (i + 0.5) * R, self.gy0 + (j + 0.5) * R

    def ground(self, x, y):
        i, j = self.cell(x, y)
        return float(self.surf[j, i])

    def solid_at(self, x, y, z):
        i, j = self.cell(x, y)
        if z < self.gnd[j, i] - 0.3:
            return True
        return self.sbase[j, i] <= z <= self.stop[j, i]

    def los(self, a, b, trees=True, stepm=2.0):
        """Line of sight between 3D points a, b. Buildings/terrain block; dense canopy optionally blocks."""
        dx = b[0] - a[0]; dy = b[1] - a[1]; dz = b[2] - a[2]
        L = math.sqrt(dx * dx + dy * dy)
        n = int(L / stepm)
        if n <= 1:
            return True
        gx0 = self.gx0; gy0 = self.gy0; nx = self.nx; ny = self.ny
        gnd = self.gnd; sb = self.sbase; st = self.stop; tree = self.tree
        treecount = 0
        for k in range(1, n):
            f = k / n
            x = a[0] + dx * f; y = a[1] + dy * f; z = a[2] + dz * f
            i = int((x - gx0) / R); j = int((y - gy0) / R)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            if z < gnd[j, i]:
                return False
            if sb[j, i] <= z <= st[j, i]:
                return False
            if trees and tree[j, i] and z < gnd[j, i] + 14:
                treecount += 1
                if treecount > 4:
                    return False
        return True

    # --- navigation (coarse 4 m grid; node = the best-clearance 2 m subcell)
    def _build_graph(self):
        from scipy.sparse import csr_matrix
        self.ok = ~self.hard
        cl = np.where(self.hard, 0.0, self.clear).astype(np.float32)
        ny2, nx2 = (self.ny // 2) * 2, (self.nx // 2) * 2
        cny, cnx = ny2 // 2, nx2 // 2
        blk = cl[:ny2, :nx2].reshape(cny, 2, cnx, 2).transpose(0, 2, 1, 3).reshape(cny, cnx, 4)
        arg = blk.argmax(axis=2)
        best = blk.max(axis=2)
        sj = np.arange(cny)[:, None] * 2 + arg // 2
        si = np.arange(cnx)[None, :] * 2 + arg % 2
        self.cnx, self.cny = cnx, cny
        self.cpx = (self.gx0 + (si + 0.5) * R).astype(np.float32)
        self.cpy = (self.gy0 + (sj + 0.5) * R).astype(np.float32)
        self.csi, self.csj = si, sj
        cok = best >= 2.6
        self.cok = cok
        pen = np.where(best < 3.5, 2.5, np.where(best < 5.0, 1.3, 1.0)).astype(np.float32)
        surf = self.surf[sj, si]
        idx = np.arange(cnx * cny).reshape(cny, cnx)
        rows = []; cols = []; vals = []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            if dj >= 0:
                a = (slice(0, cny - dj), slice(0, cnx - di)); b = (slice(dj, cny), slice(di, cnx))
            else:
                a = (slice(-dj, cny), slice(0, cnx - di)); b = (slice(0, cny + dj), slice(di, cnx))
            m = cok[a] & cok[b]
            ax, ay = self.cpx[a], self.cpy[a]; bx, by = self.cpx[b], self.cpy[b]
            L = np.hypot(bx - ax, by - ay)
            # midpoint clearance check (2 m grid)
            mi = np.clip(((ax + bx) * 0.5 - self.gx0) / R, 0, self.nx - 1).astype(int)
            mj = np.clip(((ay + by) * 0.5 - self.gy0) / R, 0, self.ny - 1).astype(int)
            m &= cl[mj, mi] >= 2.4
            m &= np.abs(surf[a] - surf[b]) < L * 1.1 + 0.3
            c = np.maximum(L, 0.5) * 0.5 * (pen[a] + pen[b])
            ia = idx[a][m]; ib = idx[b][m]; cv = c[m]
            rows += [ia, ib]; cols += [ib, ia]; vals += [cv, cv]
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        self._graph = csr_matrix((vals, (rows, cols)), shape=(cnx * cny, cnx * cny))

    def ccell(self, x, y):
        i = int((x - self.gx0) / (2 * R)); j = int((y - self.gy0) / (2 * R))
        return min(max(i, 0), self.cnx - 1), min(max(j, 0), self.cny - 1)

    def cpos(self, i, j):
        return float(self.cpx[j, i]), float(self.cpy[j, i])

    def nearest_cok(self, i, j, maxr=8):
        if self.cok[j, i]:
            return i, j
        for r in range(1, maxr):
            best = None; bd = 1e9
            for jj in range(max(0, j - r), min(self.cny, j + r + 1)):
                for ii in range(max(0, i - r), min(self.cnx, i + r + 1)):
                    if self.cok[jj, ii]:
                        d = (ii - i) ** 2 + (jj - j) ** 2
                        if d < bd:
                            bd = d; best = (ii, jj)
            if best:
                return best
        return i, j

    def nearest_ok(self, i, j, maxr=12):
        """2 m grid: nearest cell with clearance."""
        if self.ok[j, i] and self.clear[j, i] >= 2.6:
            return i, j
        for r in range(1, maxr):
            best = None; bd = 1e9
            for jj in range(max(0, j - r), min(self.ny, j + r + 1)):
                for ii in range(max(0, i - r), min(self.nx, i + r + 1)):
                    if self.ok[jj, ii] and self.clear[jj, ii] >= 3:
                        d = (ii - i) ** 2 + (jj - j) ** 2
                        if d < bd:
                            bd = d; best = (ii, jj)
            if best:
                return best
        return i, j

    def flow(self, sources, limit=np.inf):
        """Dijkstra distance field (coarse grid) from (x, y) sources. Returns (cny, cnx) array."""
        from scipy.sparse.csgraph import dijkstra
        ids = []
        for s in sources:
            i, j = self.nearest_cok(*self.ccell(s[0], s[1]))
            ids.append(j * self.cnx + i)
        d = dijkstra(self._graph, directed=False, indices=ids, min_only=True, limit=limit)
        return d.reshape(self.cny, self.cnx)


if __name__ == "__main__":
    arena = sys.argv[1] if len(sys.argv) > 1 else "/arena"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapcache.npz")
    d = build(arena)
    np.savez_compressed(out, **d)
    log("wrote", out, os.path.getsize(out))
