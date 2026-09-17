"""Evaluate an Ultra-Fast WBPP master against the PixInsight WBPP master of the same data.

Implements the master evaluation standard (docs/master-evaluation-standard.md):
one matched clean-star list on both native grids, a photometric model
``P = a*O + b`` from star fluxes and background tiles, and per-family metrics
with a uniform PASS/WARN/FAIL rule (bootstrap confidence intervals where the
statistic is star- or tile-based).  Everything dimensional is reported in P
units; background amplitudes are measured against ``sigma_ref``, PixInsight's
own per-pixel noise.  Noise verdicts use binned effective noise, which is
insensitive to interpolation, never the per-pixel sigma.

    .venv/bin/python benchmarks/evaluate_masters.py \\
        --pair L ours_L.fits pi_L.xisf --pair R ours_R.fits pi_R.xisf \\
        --out build/eval-report

Outputs ``<out>/<filter>.json`` with every value and ``<out>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sep
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import theilslopes

SEED = 0
BOOTSTRAP = 300


# --------------------------------------------------------------------------- io
def load_image(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".xisf"):
        from lightframeqc.xisf import XISF

        data = np.squeeze(XISF(path).read_image(0)).astype(np.float32)
    else:
        data = np.squeeze(fits.getdata(path)).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"{path}: expected a mono image, got {data.shape}")
    return np.ascontiguousarray(data)


# ----------------------------------------------------------------- robust helpers
def madn(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator, n: int = BOOTSTRAP) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 8:
        return float("nan"), float("nan")
    draws = np.empty(n)
    for i in range(n):
        sample = values[rng.integers(0, values.size, values.size)]
        draws[i] = statistic(sample)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def status_from(d: float, tau: float, lo: float = float("nan"), hi: float = float("nan")) -> str:
    if not math.isfinite(d):
        return "N/A"
    if math.isfinite(lo) and math.isfinite(hi):
        if hi <= tau:
            return "PASS"
        if lo > tau:
            return "FAIL"
        return "WARN"
    return "PASS" if d <= tau else "FAIL"


# ------------------------------------------------------------- image analysis
class Analysis:
    """Per-image native-grid detection, masks and photometry."""

    def __init__(self, name: str, image: np.ndarray) -> None:
        self.name = name
        self.image = image
        self.height, self.width = image.shape
        finite = np.isfinite(image)
        self.valid = ndimage.binary_erosion(finite, iterations=8, border_value=0)
        self.nan_count = int(np.count_nonzero(~finite))
        self.zero_count = int(np.count_nonzero(image == 0))
        work = np.where(finite, image, np.float32(np.nanmedian(image)))
        self.work = np.ascontiguousarray(work, dtype=np.float32)
        bkg0 = sep.Background(self.work, bw=64, bh=64, fw=3, fh=3)
        sub0 = self.work - bkg0.back()
        try:
            _, seg0 = sep.extract(sub0, 10.0, err=bkg0.globalrms, segmentation_map=True)
        except Exception:
            seg0 = np.zeros(image.shape, dtype=np.int32)
        mask0 = ndimage.binary_dilation(seg0 > 0, iterations=3)
        self.bkg = sep.Background(self.work, mask=mask0, bw=64, bh=64, fw=3, fh=3)
        self.back = self.bkg.back()
        self.sub = np.ascontiguousarray(self.work - self.back, dtype=np.float32)
        self.globalrms = float(self.bkg.globalrms)
        objects, seg = sep.extract(
            self.sub, 5.0, err=self.bkg.rms(), minarea=5, filter_type="matched",
            deblend_nthresh=32, deblend_cont=0.005, clean=True, segmentation_map=True,
        )
        self.objects = objects
        star_mask = ndimage.binary_dilation(seg > 0, iterations=3)
        # Saturation level: max of the 3x3 median-filtered image.
        med3 = ndimage.median_filter(self.work, size=3)
        self.saturation = float(np.nanmax(np.where(self.valid, med3, -np.inf)))
        x, y = objects["x"].astype(np.float64), objects["y"].astype(np.float64)
        n = len(objects)
        peak = np.zeros(n)
        xi = np.clip(np.rint(x).astype(int), 1, self.width - 2)
        yi = np.clip(np.rint(y).astype(int), 1, self.height - 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                peak = np.maximum(peak, self.sub[yi + dy, xi + dx])
        self.peak = peak
        # Flat-top test on the 5x5 core.
        flat_top = np.zeros(n, dtype=bool)
        for k in range(n):
            core = self.sub[max(0, yi[k] - 2): yi[k] + 3, max(0, xi[k] - 2): xi[k] + 3]
            flat_top[k] = np.count_nonzero(core >= 0.98 * peak[k]) >= 3
        self.flat_top = flat_top
        r50_first, _ = sep.flux_radius(self.sub, x, y, 12.0 * np.ones(n), 0.5, subpix=5)
        bright = np.argsort(-objects["flux"])[:200]
        r50_med = float(np.nanmedian(r50_first[bright])) if n else 2.0
        r50_med = min(max(r50_med, 1.0), 6.0)
        xw, yw, wflag = sep.winpos(self.sub, x, y, 0.85 * r50_med * np.ones(n))
        good = (wflag == 0) & np.isfinite(xw) & np.isfinite(yw)
        xw = np.where(good, xw, x)
        yw = np.where(good, yw, y)
        self.x, self.y = xw, yw
        self.flux = {}
        self.flux_err = {}
        for radius in (1.5, 4.0, 8.0, 10.0):
            f, e, _ = sep.sum_circle(
                self.sub, xw, yw, radius, err=self.globalrms, bkgann=(15.0, 22.0), subpix=5
            )
            self.flux[radius] = f
            self.flux_err[radius] = np.maximum(e, 1e-9)
        f12, _, _ = sep.sum_circle(self.sub, xw, yw, 12.0, bkgann=(15.0, 22.0), subpix=5)
        r50, _ = sep.flux_radius(self.sub, xw, yw, 12.0 * np.ones(n), 0.5, normflux=np.maximum(f12, 1e-6), subpix=5)
        self.r50 = r50
        self.aper_flag = self.flux[10.0] * 0  # sep.sum_circle flags are not retained; keep placeholder
        # Star mask for background/noise work: segmentation plus bright halos.
        snr4 = self.flux[4.0] / np.maximum(self.flux_err[4.0], 1e-9)
        halo = np.zeros(image.shape, dtype=bool)
        for k in np.flatnonzero((snr4 >= 1000) | flat_top | (peak >= 0.5 * self.saturation)):
            radius = 60 if (flat_top[k] or peak[k] >= 0.5 * self.saturation) else 25
            x0, x1 = max(0, int(x[k]) - radius), min(self.width, int(x[k]) + radius + 1)
            y0, y1 = max(0, int(y[k]) - radius), min(self.height, int(y[k]) + radius + 1)
            sy, sx = np.ogrid[y0:y1, x0:x1]
            halo[y0:y1, x0:x1] |= (sx - x[k]) ** 2 + (sy - y[k]) ** 2 <= radius**2
        self.star_mask = star_mask | halo
        # Isolation and clean flags (image-side part).
        tree = cKDTree(np.c_[xw, yw])
        pairs = tree.query_pairs(20.0, output_type="ndarray")
        crowded = np.zeros(n, dtype=bool)
        if pairs.size:
            d = np.hypot(xw[pairs[:, 0]] - xw[pairs[:, 1]], yw[pairs[:, 0]] - yw[pairs[:, 1]])
            f10 = self.flux[10.0]
            for (i, j), dist in zip(pairs, d):
                if dist < 12.0:
                    crowded[i] = crowded[j] = True
                else:
                    if f10[j] > 0.1 * f10[i]:
                        crowded[i] = True
                    if f10[i] > 0.1 * f10[j]:
                        crowded[j] = True
        ab_ok = (objects["b"] > 0) & (objects["a"] / np.maximum(objects["b"], 1e-9) < 2.0)
        inside = (xw >= 32) & (yw >= 32) & (xw < self.width - 32) & (yw < self.height - 32)
        unsat = (peak < 0.5 * self.saturation) & ~flat_top
        flags_ok = (objects["flag"] & 0x03) == 0
        self.clean_local = unsat & ~crowded & flags_ok & ab_ok & inside & good & np.isfinite(r50) & (r50 > 0)
        self.n_flat_top = int(np.count_nonzero(flat_top))

    # ---- noise tiles (native grid) ----
    def tile_noise(self, tile: int = 64, max_mask: float = 0.30):
        h = (self.height // tile) * tile
        w = (self.width // tile) * tile
        img = self.sub[:h, :w].reshape(h // tile, tile, w // tile, tile).transpose(0, 2, 1, 3)
        mask = (self.star_mask | ~self.valid)[:h, :w].reshape(h // tile, tile, w // tile, tile).transpose(0, 2, 1, 3)
        yy, xx = np.mgrid[:tile, :tile]
        design = np.c_[xx.ravel(), yy.ravel(), np.ones(tile * tile)].astype(np.float64)
        sig = np.full((h // tile, w // tile), np.nan)
        rho = {k: np.full((h // tile, w // tile), np.nan) for k in ("x1", "x2", "x3", "y1", "y2", "y3")}
        residuals = np.full((h // tile, w // tile, tile, tile), np.nan, dtype=np.float32)
        level = np.full((h // tile, w // tile), np.nan)
        for i in range(h // tile):
            for j in range(w // tile):
                m = ~mask[i, j].ravel()
                if m.mean() < 1.0 - max_mask:
                    continue
                t = img[i, j].ravel().astype(np.float64)
                coef, *_ = np.linalg.lstsq(design[m], t[m], rcond=None)
                r = (t - design @ coef).reshape(tile, tile)
                r[mask[i, j]] = np.nan
                residuals[i, j] = r
                level[i, j] = coef[2] + coef[0] * tile / 2 + coef[1] * tile / 2 + float(
                    np.median(self.back[i * tile: (i + 1) * tile, j * tile: (j + 1) * tile])
                )
                s = madn(r[~mask[i, j]])
                sig[i, j] = s
                if s > 0:
                    for k in (1, 2, 3):
                        dx = (r[:, k:] - r[:, :-k]).ravel()
                        dy = (r[k:, :] - r[:-k, :]).ravel()
                        rho[f"x{k}"][i, j] = 1.0 - madn(dx) ** 2 / (2 * s * s)
                        rho[f"y{k}"][i, j] = 1.0 - madn(dy) ** 2 / (2 * s * s)
        return sig, rho, residuals, level

    def binned_noise(self, residuals: np.ndarray, sig_tiles: np.ndarray, b: int) -> float:
        """Effective noise b*sigma_b from block averages of the tile residuals."""

        valid = np.isfinite(sig_tiles)
        res = residuals[valid]  # (n, tile, tile)
        if res.size == 0:
            return float("nan")
        tile = res.shape[1]
        nb = tile // b
        blocks = np.nanmean(res[:, : nb * b, : nb * b].reshape(res.shape[0], nb, b, nb, b), axis=(2, 4))
        frac = np.isfinite(res[:, : nb * b, : nb * b].reshape(res.shape[0], nb, b, nb, b)).mean(axis=(2, 4))
        blocks = np.where(frac >= 0.75, blocks, np.nan)
        per_tile = np.array([madn(t) for t in blocks.reshape(blocks.shape[0], -1)])
        return float(b * np.nanmedian(per_tile))


# ------------------------------------------------------------- matching
def match_and_align(ours: Analysis, pi: Analysis) -> dict[str, Any]:
    """Affine map from the O grid to the P grid from matched stars."""

    def brightest(a: Analysis, n: int = 400):
        ok = a.clean_local & (a.peak > 0)
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-a.flux[10.0][idx])][:n]
        return a.x[idx], a.y[idx]

    ox, oy = brightest(ours)
    px, py = brightest(pi)
    dx = (px[:, None] - ox[None, :]).ravel()
    dy = (py[:, None] - oy[None, :]).ravel()
    hist, xe, ye = np.histogram2d(dx, dy, bins=400, range=[[-600, 600], [-600, 600]])
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    sx, sy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    tree = cKDTree(np.c_[pi.x, pi.y])
    model = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]])
    matched = None
    for radius in (6.0, 2.0, 1.0):
        px_pred = model[0, 0] * ours.x + model[0, 1] * ours.y + model[0, 2]
        py_pred = model[1, 0] * ours.x + model[1, 1] * ours.y + model[1, 2]
        d, k = tree.query(np.c_[px_pred, py_pred], distance_upper_bound=radius)
        m = np.isfinite(d)
        # mutual nearest neighbour
        back = cKDTree(np.c_[px_pred[m], py_pred[m]])
        db, kb = back.query(np.c_[pi.x[k[m]], pi.y[k[m]]], distance_upper_bound=radius)
        mutual = np.flatnonzero(m)[np.flatnonzero(np.isfinite(db) & (kb == np.arange(m.sum())))]
        if mutual.size < 20:
            break
        A = np.c_[ours.x[mutual], ours.y[mutual], np.ones(mutual.size)]
        cx, *_ = np.linalg.lstsq(A, pi.x[k[mutual]], rcond=None)
        cy, *_ = np.linalg.lstsq(A, pi.y[k[mutual]], rcond=None)
        model = np.array([cx, cy])
        matched = (mutual, k[mutual])
    if matched is None:
        raise RuntimeError("star matching failed")
    oi, pj = matched
    A = np.c_[ours.x[oi], ours.y[oi], np.ones(oi.size)]
    rx = A @ model[0] - pi.x[pj]
    ry = A @ model[1] - pi.y[pj]
    rms = float(np.sqrt(np.mean(rx**2 + ry**2)))
    return {"model": model, "ours_index": oi, "pi_index": pj, "rms": rms, "matched": int(oi.size),
            "coarse_shift": [float(sx), float(sy)]}


def map_o_to_p(model: np.ndarray, x: np.ndarray, y: np.ndarray):
    return model[0, 0] * x + model[0, 1] * y + model[0, 2], model[1, 0] * x + model[1, 1] * y + model[1, 2]


# ------------------------------------------------------------- evaluation
def evaluate_pair(filter_name: str, ours_path: str, pi_path: str, out_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    ours = Analysis("ours", load_image(ours_path))
    pi = Analysis("pi", load_image(pi_path))
    report: dict[str, Any] = {"filter": filter_name, "ours": ours_path, "pi": pi_path, "metrics": [],
                              "info": {}}

    def metric(family: str, name: str, p: float | None, o: float | None, d: float, tau: float,
               lo: float = float("nan"), hi: float = float("nan"), note: str = "", verdict: bool = True,
               forced: str | None = None) -> str:
        st = forced or status_from(d, tau, lo, hi)
        report["metrics"].append({"family": family, "metric": name, "pi": p, "ours": o, "d": d, "tau": tau,
                                  "ci": [lo, hi], "status": st, "note": note, "verdict": verdict})
        return st

    align = match_and_align(ours, pi)
    model = align["model"]
    oi, pj = align["ours_index"], align["pi_index"]
    report["info"]["alignment"] = {"rms": align["rms"], "matched": align["matched"],
                                   "model_o_to_p": model.tolist(), "coarseShift": align["coarse_shift"]}
    valid_alignment = align["rms"] <= 0.10 and align["matched"] >= 200

    # ---- photometric scale a and pedestal b ----
    clean = ours.clean_local[oi] & pi.clean_local[pj]
    f10_o, f10_p = ours.flux[10.0][oi], pi.flux[10.0][pj]
    snr_p = pi.flux[4.0][pj] / np.maximum(pi.flux_err[4.0][pj], 1e-9)
    scale_sel = clean & (snr_p >= 100) & (f10_o > 0) & (f10_p > 0)
    ln_ratio = np.log(f10_p[scale_sel]) - np.log(f10_o[scale_sel])
    a = float(np.exp(np.median(ln_ratio)))
    # pedestal from star-free 256-px background tiles
    bo = sep.Background(ours.work, mask=ours.star_mask, bw=256, bh=256, fw=5, fh=5).back()
    bp = sep.Background(pi.work, mask=pi.star_mask, bw=256, bh=256, fw=5, fh=5).back()
    lattice_y, lattice_x = np.mgrid[64: ours.height - 64: 32, 64: ours.width - 64: 32]
    lx, ly = map_o_to_p(model, lattice_x.astype(np.float64), lattice_y.astype(np.float64))
    inside = (lx >= 64) & (ly >= 64) & (lx < pi.width - 64) & (ly < pi.height - 64)
    bo_l = bo[lattice_y[inside], lattice_x[inside]].astype(np.float64)
    bp_l = ndimage.map_coordinates(bp, [ly[inside], lx[inside]], order=1)
    b = float(np.median(bp_l - a * bo_l))
    report["info"]["photometric_model"] = {"a": a, "b": b, "lnRatioMadn": madn(ln_ratio), "stars": int(scale_sel.sum())}
    sigma_ref = None

    # ---- family 2: noise ----
    sig_o, rho_o, res_o, _ = ours.tile_noise()
    sig_p, rho_p, res_p, _ = pi.tile_noise()
    sigma1_o, sigma1_p = float(np.nanmedian(sig_o)), float(np.nanmedian(sig_p))
    sigma_ref = sigma1_p
    report["info"]["noise"] = {
        "sigma1_pi": sigma1_p, "sigma1_ours_Punits": a * sigma1_o, "sigma1_ours": sigma1_o,
        "rho_pi": {k: float(np.nanmedian(v)) for k, v in rho_p.items()},
        "rho_ours": {k: float(np.nanmedian(v)) for k, v in rho_o.items()},
        "tiles_pi": int(np.isfinite(sig_p).sum()), "tiles_ours": int(np.isfinite(sig_o).sum()),
    }
    gains = {}
    for bsize in (2, 4, 8, 16):
        eo = ours.binned_noise(res_o, sig_o, bsize)
        ep = pi.binned_noise(res_p, sig_p, bsize)
        g = ep / (a * eo) if eo > 0 else float("nan")
        # tile bootstrap of the gain
        vo = np.isfinite(sig_o)
        vp = np.isfinite(sig_p)
        draws = []
        ro = res_o[vo]
        rp = res_p[vp]
        for _ in range(60):
            so = rng.integers(0, ro.shape[0], ro.shape[0])
            sp = rng.integers(0, rp.shape[0], rp.shape[0])
            eo_b = ours.binned_noise(ro[so], np.ones(so.size), bsize)
            ep_b = pi.binned_noise(rp[sp], np.ones(sp.size), bsize)
            draws.append(ep_b / (a * eo_b))
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        gains[bsize] = (g, lo, hi, ep, a * eo)
        report["info"]["noise"][f"sigma_eff_{bsize}_pi"] = ep
        report["info"]["noise"][f"sigma_eff_{bsize}_ours_Punits"] = a * eo
    for bsize in (4, 8):
        g, lo, hi, ep, eo = gains[bsize]
        metric("noise", f"G_{bsize} (effective-noise gain, >1 = ours better)", ep, eo, 1 - g, 0.02,
               1 - hi, 1 - lo, note=f"G={g:.4f} CI[{lo:.4f},{hi:.4f}]", verdict=(bsize == 8))
    report["info"]["noise"]["G"] = {str(k): [v[0], v[1], v[2]] for k, v in gains.items()}

    # empty-aperture depth
    def empty_apertures(an: Analysis, positions: np.ndarray) -> np.ndarray:
        x, y = positions[:, 0], positions[:, 1]
        f, _, _ = sep.sum_circle(an.sub, x, y, 4.0, bkgann=(8.0, 12.0), subpix=5)
        return f

    pos_p = np.c_[rng.uniform(64, pi.width - 64, 12000), rng.uniform(64, pi.height - 64, 12000)]
    keep = ~pi.star_mask[pos_p[:, 1].astype(int), pos_p[:, 0].astype(int)]
    pos_p = pos_p[keep][:3000]
    inv = np.linalg.inv(np.vstack([model, [0, 0, 1]]))
    ox_e = inv[0, 0] * pos_p[:, 0] + inv[0, 1] * pos_p[:, 1] + inv[0, 2]
    oy_e = inv[1, 0] * pos_p[:, 0] + inv[1, 1] * pos_p[:, 1] + inv[1, 2]
    ok = (ox_e >= 64) & (oy_e >= 64) & (ox_e < ours.width - 64) & (oy_e < ours.height - 64)
    ok &= ~ours.star_mask[np.clip(oy_e.astype(int), 0, ours.height - 1), np.clip(ox_e.astype(int), 0, ours.width - 1)]
    fe_p = empty_apertures(pi, pos_p[ok])
    fe_o = empty_apertures(ours, np.c_[ox_e[ok], oy_e[ok]])
    sf_p, sf_o = madn(fe_p), a * madn(fe_o)
    dm = -2.5 * math.log10(sf_o / sf_p)
    draws = []
    for _ in range(BOOTSTRAP):
        s1 = fe_p[rng.integers(0, fe_p.size, fe_p.size)]
        s2 = fe_o[rng.integers(0, fe_o.size, fe_o.size)]
        draws.append(-2.5 * math.log10(a * madn(s2) / madn(s1)))
    dm_lo, dm_hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    metric("noise", "depth Δm (mag, >0 = ours deeper)", 5 * sf_p, 5 * sf_o, -dm, 0.02, -dm_hi, -dm_lo,
           note=f"Δm={dm:+.3f} CI[{dm_lo:+.3f},{dm_hi:+.3f}] apertures={int(ok.sum())}")
    report["info"]["depth"] = {"dm": dm, "ci": [dm_lo, dm_hi], "flim5_pi": 5 * sf_p, "flim5_ours_Punits": 5 * sf_o}

    # faint-star flux preservation (B5: SNR 10-30) and SNR ratio
    bins = {"B1": snr_p >= 1000, "B2": (snr_p >= 300) & (snr_p < 1000), "B3": (snr_p >= 100) & (snr_p < 300),
            "B4": (snr_p >= 30) & (snr_p < 100), "B5": (snr_p >= 10) & (snr_p < 30)}
    sel5 = clean & bins["B5"] & (f10_o > 0) & (f10_p > 0)
    q_faint = a * f10_o[sel5] / f10_p[sel5]
    qf = float(np.median(q_faint)) if sel5.sum() >= 20 else float("nan")
    lo, hi = bootstrap_ci(q_faint, np.median, rng)
    metric("noise", "q_faint (B5 flux ratio ours/pi)", 1.0, qf, abs(qf - 1) if math.isfinite(qf) else float("nan"), 0.02,
           max(abs(lo - 1), 0) if math.isfinite(lo) else float("nan"), max(abs(lo - 1), abs(hi - 1)) if math.isfinite(hi) else float("nan"),
           note=f"n={int(sel5.sum())}")

    # ---- family 1: PSF on matched clean stars B1-B3 ----
    psf_sel = clean & (bins["B1"] | bins["B2"] | bins["B3"]) & np.isfinite(ours.r50[oi]) & np.isfinite(pi.r50[pj])
    fw_o, fw_p = 2 * ours.r50[oi][psf_sel], 2 * pi.r50[pj][psf_sel]
    rel = (fw_o - fw_p) / fw_p
    d = float(np.median(rel))
    lo, hi = bootstrap_ci(rel, np.median, rng)
    metric("psf", "FWHM_hlr relative (ours-pi)/pi", float(np.median(fw_p)), float(np.median(fw_o)), d, 0.02, lo, hi,
           note=f"n={int(psf_sel.sum())} px: pi {np.median(fw_p):.3f} ours {np.median(fw_o):.3f}")
    # windowed moments (deconvolved) on stamps
    def moments(an: Analysis, idx: np.ndarray, sigma_w: float):
        out = np.full((idx.size, 3), np.nan)
        for n_, k in enumerate(idx):
            cx, cy = an.x[k], an.y[k]
            x0, y0 = int(round(cx)) - 12, int(round(cy)) - 12
            if x0 < 0 or y0 < 0 or x0 + 25 > an.width or y0 + 25 > an.height:
                continue
            stamp = an.sub[y0: y0 + 25, x0: x0 + 25].astype(np.float64)
            yy, xx = np.mgrid[y0: y0 + 25, x0: x0 + 25].astype(np.float64)
            mx, my = cx, cy
            for _ in range(5):
                w = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * sigma_w**2)) * np.maximum(stamp, 0)
                tot = w.sum()
                if tot <= 0:
                    break
                mx, my = (w * xx).sum() / tot, (w * yy).sum() / tot
            w = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * sigma_w**2)) * np.maximum(stamp, 0)
            tot = w.sum()
            if tot <= 0:
                continue
            x2 = (w * (xx - mx) ** 2).sum() / tot
            y2 = (w * (yy - my) ** 2).sum() / tot
            xy = (w * (xx - mx) * (yy - my)).sum() / tot
            M = np.array([[x2, xy], [xy, y2]])
            try:
                Mt = np.linalg.inv(np.linalg.inv(M) - np.eye(2) / sigma_w**2)
            except np.linalg.LinAlgError:
                continue
            lam = np.linalg.eigvalsh(Mt)
            if lam.min() <= 0:
                continue
            out[n_] = (2.3548 * math.sqrt(lam.mean()), (lam[1] - lam[0]) / (lam[1] + lam[0]), tot)
        return out

    sigma_w = 2.0 * float(np.median(pi.r50[pj][psf_sel]))
    mo = moments(ours, oi[psf_sel], sigma_w)
    mp = moments(pi, pj[psf_sel], sigma_w)
    okm = np.isfinite(mo[:, 0]) & np.isfinite(mp[:, 0])
    rel_m = (mo[okm, 0] - mp[okm, 0]) / mp[okm, 0]
    lo, hi = bootstrap_ci(rel_m, np.median, rng)
    metric("psf", "FWHM_mom relative (ours-pi)/pi", float(np.median(mp[okm, 0])), float(np.median(mo[okm, 0])),
           float(np.median(rel_m)), 0.02, lo, hi, note=f"n={int(okm.sum())}")
    de = mo[okm, 1] - mp[okm, 1]
    lo, hi = bootstrap_ci(de, np.median, rng)
    metric("psf", "ellipticity (ours-pi)", float(np.median(mp[okm, 1])), float(np.median(mo[okm, 1])),
           float(np.median(de)), 0.01, lo, hi)
    c_o = ours.flux[1.5][oi][psf_sel] / np.maximum(ours.flux[10.0][oi][psf_sel], 1e-9)
    c_p = pi.flux[1.5][pj][psf_sel] / np.maximum(pi.flux[10.0][pj][psf_sel], 1e-9)
    rel_c = (c_p - c_o) / c_p
    lo, hi = bootstrap_ci(rel_c, np.median, rng)
    metric("psf", "core concentration C=F1.5/F10 relative loss (pi-ours)/pi", float(np.median(c_p)),
           float(np.median(c_o)), float(np.median(rel_c)), 0.03, lo, hi)
    # per-zone worst FWHM
    zx = np.clip((pi.x[pj][psf_sel] / pi.width * 3).astype(int), 0, 2)
    zy = np.clip((pi.y[pj][psf_sel] / pi.height * 3).astype(int), 0, 2)
    zone_fwhm = {}
    for zi in range(3):
        for zj in range(3):
            s = (zy == zi) & (zx == zj)
            if s.sum() >= 15:
                zone_fwhm[f"{zi}{zj}"] = float(np.median(rel[s]))
    worst = max(zone_fwhm.values()) if zone_fwhm else float("nan")
    metric("psf", "FWHM_hlr worst zone relative", None, None, worst, 0.04, note=json.dumps({k: round(v, 4) for k, v in zone_fwhm.items()}))

    # ---- family 3: background flatness (P units, sigma_ref) ----
    def p2_fit(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xs, ys = x / x.max(), y / y.max()
        A = np.c_[np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys]
        keep = np.isfinite(values)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[keep], values[keep], rcond=None)
            r = values - A @ coef
            s = madn(r[keep])
            keep = np.isfinite(values) & (np.abs(r) <= 3 * s + 1e-12)
        return A @ coef

    bo_p = a * bo_l + b  # ours background in P units on the P lattice
    bp_p = bp_l
    lxv, lyv = lx[inside], ly[inside]
    H_o = bo_p - p2_fit(bo_p, lxv, lyv)
    H_p = bp_p - p2_fit(bp_p, lxv, lyv)
    dH = H_p - H_o
    def ptp(v): return float(np.nanpercentile(v, 99) - np.nanpercentile(v, 1))
    rms_h_o, rms_h_p = float(np.nanstd(H_o)) / sigma_ref, float(np.nanstd(H_p)) / sigma_ref
    metric("background", "RMS_H high-order background (σ_ref)", rms_h_p, rms_h_o, rms_h_o - rms_h_p, 0.05)
    ptp_o, ptp_p = ptp(H_o) / sigma_ref, ptp(H_p) / sigma_ref
    metric("background", "PtP_H high-order p1-p99 (σ_ref)", ptp_p, ptp_o, ptp_o - max(ptp_p + 0.10, 0.30), 0.0,
           note="PASS if ours <= max(pi+0.10, 0.30)")
    ptp_dh = ptp(dH) / sigma_ref
    metric("background", "PtP_ΔH difference of high-order structure (σ_ref)", None, ptp_dh, ptp_dh, 0.30,
           note="<0.30 => visually indistinguishable backgrounds", forced=("PASS" if ptp_dh < 0.30 else "WARN"))
    # axis-aligned residual power and edge bands (own footprint, own units -> sigma_ref)
    def axis_and_edges(back: np.ndarray, an: Analysis, scale: float):
        fin = an.valid
        H, W = back.shape
        rows = np.array([np.nanmedian(np.where(fin[i], back[i], np.nan)) for i in range(0, H, 8)])
        cols = np.array([np.nanmedian(np.where(fin[:, j], back[:, j], np.nan)) for j in range(0, W, 8)])
        def detrend(p):
            t = np.arange(p.size, dtype=float)
            k = np.isfinite(p)
            coef = np.polyfit(t[k], p[k], 2)
            return p - np.polyval(coef, t)
        py_, px_ = detrend(rows), detrend(cols)
        rms_axis = math.sqrt(np.nanmean(py_**2) + np.nanmean(px_**2)) * scale / sigma_ref
        max_axis = float(np.nanmax(np.abs(np.r_[py_, px_]))) * scale / sigma_ref
        # edge roll-off: median over band of (B - P2_interior)
        yy, xx = np.mgrid[0:H:16, 0:W:16]
        vals = back[::16, ::16].astype(np.float64)
        interior = (yy > 0.05 * H) & (yy < 0.95 * H) & (xx > 0.05 * W) & (xx < 0.95 * W)
        xs, ys = xx / W, yy / H
        A = np.c_[np.ones(xs.size), xs.ravel(), ys.ravel(), (xs * xs).ravel(), (xs * ys).ravel(), (ys * ys).ravel()]
        coef, *_ = np.linalg.lstsq(A[interior.ravel()], vals.ravel()[interior.ravel()], rcond=None)
        resid = (vals.ravel() - A @ coef).reshape(vals.shape)
        bands = {
            "top": float(np.nanmedian(resid[yy < 0.05 * H])), "bottom": float(np.nanmedian(resid[yy > 0.95 * H])),
            "left": float(np.nanmedian(resid[xx < 0.05 * W])), "right": float(np.nanmedian(resid[xx > 0.95 * W])),
        }
        return rms_axis, max_axis, {k: v * scale / sigma_ref for k, v in bands.items()}

    ax_o, mx_o, edges_o = axis_and_edges(bo, ours, a)
    ax_p, mx_p, edges_p = axis_and_edges(bp, pi, 1.0)
    metric("background", "RMS_axis axis-aligned residual (σ_ref)", ax_p, ax_o, ax_o - ax_p, 0.05)
    metric("background", "max|p| axis-aligned profile (σ_ref)", mx_p, mx_o, mx_o - max(mx_p + 0.10, 0.30), 0.0,
           note="PASS if ours <= max(pi+0.10, 0.30)")
    worst_edge_o = max(abs(v) for v in edges_o.values())
    worst_edge_p = max(abs(v) for v in edges_p.values())
    metric("background", "E_band worst edge roll-off |median| (σ_ref)", worst_edge_p, worst_edge_o,
           worst_edge_o - max(worst_edge_p + 0.10, 0.30), 0.0,
           note="pi " + json.dumps({k: round(v, 2) for k, v in edges_p.items()}) + " ours " + json.dumps({k: round(v, 2) for k, v in edges_o.items()}))
    report["info"]["background"] = {"sigma_ref": sigma_ref, "edges_pi": edges_p, "edges_ours": edges_o,
                                    "ptp_B_pi": ptp(bp_p - np.nanmedian(bp_p)) / sigma_ref,
                                    "ptp_B_ours": ptp(bo_p - np.nanmedian(bo_p)) / sigma_ref}

    # STF (common STF from P) zone spread and visible-structure fraction
    def mtf(m, x):
        return ((m - 1) * x) / ((2 * m - 1) * x - m)
    sky_free_p = pi.work[~pi.star_mask & pi.valid]
    med_p = float(np.median(sky_free_p))
    madn_p = madn(sky_free_p)
    c0 = max(0.0, med_p - 2.8 * madn_p)
    # The STF is P's own (c1 = 1 in P units); ours is clipped into it.
    c1 = max(1.0, float(np.nanmax(pi.work)), c0 + 1e-6)
    m = mtf(0.25, (med_p - c0) / (c1 - c0))
    def stf(img):
        return mtf(m, np.clip((img - c0) / (c1 - c0), 0, 1))
    S_p = stf(bp_p)
    S_o = stf(bo_p)
    def zone_spread(S, x, y):
        zx = np.clip((x / pi.width * 3).astype(int), 0, 2)
        zy = np.clip((y / pi.height * 3).astype(int), 0, 2)
        meds = [np.nanmedian(S[(zx == j) & (zy == i)]) for i in range(3) for j in range(3)]
        return 255.0 * (max(meds) - min(meds))
    Z_p, Z_o = zone_spread(S_p, lxv, lyv), zone_spread(S_o, lxv, lyv)
    metric("background", "Z STF zone spread (8-bit levels)", Z_p, Z_o, Z_o - Z_p, 2.0)
    f_vis_p = float(np.mean(np.abs(S_p - p2_fit(S_p, lxv, lyv)) > 0.02))
    f_vis_o = float(np.mean(np.abs(S_o - p2_fit(S_o, lxv, lyv)) > 0.02))
    metric("background", "f_vis visible high-order structure fraction (STF)", f_vis_p, f_vis_o, f_vis_o - f_vis_p, 0.02)

    # ---- family 4: artefacts ----
    def core_fraction_selfcheck(an: Analysis, idx: np.ndarray, snr: np.ndarray):
        c = an.flux[1.5][idx] / np.maximum(an.flux[10.0][idx], 1e-9)
        b1 = c[snr >= 1000]
        b3 = c[(snr >= 100) & (snr < 300)]
        return float(np.median(b1) - np.median(b3)) if b1.size >= 10 and b3.size >= 10 else float("nan")
    dc_o = core_fraction_selfcheck(ours, oi[clean], snr_p[clean])
    dc_p = core_fraction_selfcheck(pi, pj[clean], snr_p[clean])
    metric("artefacts", "Δc core-fraction self-consistency (B1-B3; < -0.03 = core clipping)", dc_p, dc_o,
           (-0.03 - dc_o) if math.isfinite(dc_o) else float("nan"), 0.0, note="PASS if ours Δc >= -0.03")

    def spikes(an: Analysis, sigma1: float):
        fp = np.ones((3, 3), dtype=bool); fp[1, 1] = False
        med8 = ndimage.median_filter(an.sub, footprint=fp)
        z = (an.sub - med8) / max(sigma1, 1e-9)
        cand = (np.abs(z) > 6) & ~an.star_mask & an.valid
        lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
        if n == 0:
            return 0.0, 0.0
        sizes = np.atleast_1d(ndimage.sum(cand, lab, index=np.arange(1, n + 1)))
        small = [int(index) + 1 for index in np.flatnonzero(sizes <= 2)]
        hot = cold = 0
        slices = ndimage.find_objects(lab)
        for k in small:
            sl = slices[k - 1]
            if sl is None:
                continue
            local = np.argwhere(lab[sl] == k)[0]
            y_, x_ = sl[0].start + local[0], sl[1].start + local[1]
            value = an.sub[y_, x_]
            nb = np.abs(an.sub[max(0, y_ - 1): y_ + 2, max(0, x_ - 1): x_ + 2]).copy()
            nb[min(1, y_), min(1, x_)] = 0
            if nb.max() < 0.3 * abs(value):
                if value > 0:
                    hot += 1
                else:
                    cold += 1
        mpx = np.count_nonzero(an.valid) / 1e6
        return hot / mpx, cold / mpx
    hot_o, cold_o = spikes(ours, sigma1_o)
    hot_p, cold_p = spikes(pi, sigma1_p)
    metric("artefacts", "hot pixels per Mpx", hot_p, hot_o, hot_o - (1.2 * hot_p + 2), 0.0, note="PASS if ours <= 1.2*pi+2")
    metric("artefacts", "cold pixels per Mpx", cold_p, cold_o, cold_o - (1.2 * cold_p + 2), 0.0, note="PASS if ours <= 1.2*pi+2")

    def trails(an: Analysis, sigma1: float):
        h = (an.height // 4) * 4
        w = (an.width // 4) * 4
        masked = an.star_mask | ~an.valid
        r = np.where(masked, 0.0, an.sub)[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3)) / max(sigma1 / 4, 1e-9)
        # Winsorize so extended sources (galaxy, halos) cannot masquerade as a line.
        r = np.clip(r, -5.0, 5.0)
        wgt = (~masked)[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
        best = []
        for theta in range(0, 180, 2):
            rr = ndimage.rotate(r, theta, order=1, reshape=True, prefilter=False)
            ww = ndimage.rotate(wgt, theta, order=1, reshape=True, prefilter=False)
            p = rr.sum(axis=0)
            nn = ww.sum(axis=0)
            okc = nn >= 100
            if okc.sum() < 20:
                continue
            z = np.full_like(p, np.nan)
            z[okc] = p[okc] / np.sqrt(nn[okc])
            z = z / max(madn(z[okc]), 1e-9)
            zz = np.nan_to_num(z, nan=0.0)
            k = int(np.argmax(np.abs(zz)))
            peak = abs(zz[k])
            # A trail is a narrow, isolated line: neighbours 4-8 bins away must be below half the peak.
            side = np.r_[zz[max(0, k - 8): max(0, k - 3)], zz[k + 4: k + 9]]
            isolated = side.size > 0 and np.max(np.abs(side)) < 0.5 * peak
            if peak >= 10 and isolated:
                best.append({"theta": theta, "u": int(k), "z": float(zz[k]), "length": float(nn[k])})
        return best
    tr_o = trails(ours, sigma1_o)
    tr_p = trails(pi, sigma1_p)
    unmatched = [t for t in tr_o if not any(abs(t["theta"] - s["theta"]) <= 2 for s in tr_p)]
    metric("artefacts", "trail residuals (count, ours without pi counterpart)", float(len(tr_p)), float(len(tr_o)),
           float(len(unmatched)), 0.0, note=json.dumps({"ours": tr_o[:5], "pi": tr_p[:5]}))

    def halo(an: Analysis, idx: np.ndarray, sigma1: float):
        sel = idx[(an.peak[idx] > 0.1 * an.saturation) & (an.peak[idx] < 0.5 * an.saturation) & an.clean_local[idx]]
        sel = sel[np.argsort(-an.flux[10.0][sel])][:60]
        if sel.size < 10:
            return float("nan"), float("nan"), float("nan")
        radii = np.arange(0.5, 24.5, 0.5)
        profiles = []
        wings = []
        for k in sel:
            cx, cy = an.x[k], an.y[k]
            x0, y0 = int(round(cx)) - 40, int(round(cy)) - 40
            if x0 < 0 or y0 < 0 or x0 + 81 > an.width or y0 + 81 > an.height:
                continue
            stamp = an.sub[y0: y0 + 81, x0: x0 + 81].astype(np.float64)
            yy, xx = np.mgrid[y0: y0 + 81, x0: x0 + 81]
            rr = np.hypot(xx - cx, yy - cy)
            sky = np.median(stamp[(rr >= 30) & (rr <= 40)])
            stamp -= sky
            prof = np.array([np.mean(stamp[(rr >= r_ - 0.25) & (rr < r_ + 0.25)]) if np.any((rr >= r_ - 0.25) & (rr < r_ + 0.25)) else np.nan for r_ in radii])
            peak_eq = an.flux[10.0][k] / (2 * math.pi * (an.r50[k] / 1.1774) ** 2)
            profiles.append(prof / peak_eq)
            fw = 2 * an.r50[k]
            inner = stamp[rr <= 2 * fw].sum()
            wings.append(1 - inner / max(an.flux[10.0][k], 1e-9))
        stack = np.nanmedian(np.array(profiles), axis=0)
        fw = 2 * float(np.median(an.r50[sel]))
        window = (radii >= 1.25 * fw) & (radii <= 4 * fw)
        return float(np.nanmin(stack[window])), float(np.nanmedian(wings)), fw
    dp_o, w_o, _ = halo(ours, np.arange(len(ours.objects)), sigma1_o)
    dp_p, w_p, _ = halo(pi, np.arange(len(pi.objects)), sigma1_p)
    metric("artefacts", "ringing depth D_peak (min normalised profile 1.25-4 FWHM)", dp_p, dp_o, (dp_p - 0.002) - dp_o, 0.0,
           note="PASS if ours >= pi - 0.002")
    metric("artefacts", "wing fraction W beyond 2 FWHM", w_p, w_o, w_o - w_p, 0.01)
    # border checks
    def border_rows(an: Analysis):
        img = an.image
        z = 0
        for row in (img[0], img[-1]):
            if np.all(~np.isfinite(row)) or np.all(row == 0) or np.nanstd(row) == 0:
                z += 1
        for col in (img[:, 0], img[:, -1]):
            if np.all(~np.isfinite(col)) or np.all(col == 0) or np.nanstd(col) == 0:
                z += 1
        return z
    metric("artefacts", "NaN/Inf pixels", float(pi.nan_count), float(ours.nan_count), float(ours.nan_count), 0.0)
    metric("artefacts", "constant/zero border rows+cols", float(border_rows(pi)), float(border_rows(ours)),
           float(border_rows(ours)), float(border_rows(pi)))
    def edge_sigma_ratio(sig: np.ndarray):
        ny, nx = sig.shape
        interior = np.nanmedian(sig[ny // 10: -max(1, ny // 10), nx // 10: -max(1, nx // 10)])
        band = np.nanmedian(np.r_[sig[:max(1, ny // 20)].ravel(), sig[-max(1, ny // 20):].ravel(), sig[:, :max(1, nx // 20)].ravel(), sig[:, -max(1, nx // 20):].ravel()])
        return float(band / interior)
    metric("artefacts", "edge-band noise / interior noise", edge_sigma_ratio(sig_p), edge_sigma_ratio(sig_o),
           edge_sigma_ratio(sig_o) - max(1.10, edge_sigma_ratio(sig_p) + 0.02), 0.0,
           note="PASS if ours <= max(1.10, pi + 0.02)")

    # ---- family 5: photometry ----
    lin_sel = clean & (bins["B1"] | bins["B2"] | bins["B3"] | bins["B4"]) & (f10_o > 0) & (f10_p > 0)
    slope, intercept, s_lo, s_hi = theilslopes(np.log10(f10_p[lin_sel]), np.log10(f10_o[lin_sel]))
    metric("photometry", "linearity slope s (log flux pi vs ours)", 1.0, float(slope), abs(slope - 1), 0.01,
           abs(s_lo - 1) if (s_lo - 1) * (s_hi - 1) > 0 else 0.0, max(abs(s_lo - 1), abs(s_hi - 1)), note=f"n={int(lin_sel.sum())}")
    q_bins = {}
    worst_q = 0.0
    for name, sel_b in bins.items():
        s = clean & sel_b & (f10_o > 0) & (f10_p > 0)
        if s.sum() >= 15:
            q = float(np.median(a * f10_o[s] / f10_p[s]))
            q_bins[name] = q
            if name != "B5":
                worst_q = max(worst_q, abs(q - 1))
    metric("photometry", "per-bin flux ratio |q-1| worst (B1-B4)", None, None, worst_q, 0.01, note=json.dumps({k: round(v, 4) for k, v in q_bins.items()}))
    sat_o_p = a * ours.saturation + b
    metric("photometry", "saturation level (P units)", pi.saturation, sat_o_p, 0.98 * pi.saturation - sat_o_p, 0.0, note="PASS if ours >= 0.98*pi")
    metric("photometry", "flat-topped star count", float(pi.n_flat_top), float(ours.n_flat_top),
           ours.n_flat_top - (1.05 * pi.n_flat_top + 2), 0.0, note="PASS if ours <= 1.05*pi+2")
    # flux-ratio field
    fr_sel = clean & (snr_p >= 100) & (f10_o > 0) & (f10_p > 0)
    q = a * f10_o[fr_sel] / f10_p[fr_sel] - 1
    surface = p2_fit(q, pi.x[pj][fr_sel], pi.y[pj][fr_sel])
    ptp_q = float(np.nanpercentile(surface, 99) - np.nanpercentile(surface, 1)) * 100
    metric("photometry", "flux-ratio field PtP_q (%)", None, ptp_q, ptp_q, 2.0, note="2-5% WARN, >5% FAIL",
           forced=("PASS" if ptp_q <= 2 else "WARN" if ptp_q <= 5 else "FAIL"))
    report["info"]["timing_seconds"] = time.perf_counter() - started
    report["info"]["valid_alignment"] = valid_alignment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{filter_name}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report


def cross_channel(label: str, paths: dict[str, str], out_dir: Path) -> dict[str, Any]:
    """Cross-channel star offsets within one pipeline (all channels on one grid)."""

    analyses = {f: Analysis(f, load_image(p)) for f, p in paths.items()}
    rows = []
    names = list(analyses)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = analyses[names[i]], analyses[names[j]]
            sa = a_.clean_local & (a_.flux[4.0] / np.maximum(a_.flux_err[4.0], 1e-9) >= 50)
            sb = b_.clean_local & (b_.flux[4.0] / np.maximum(b_.flux_err[4.0], 1e-9) >= 50)
            ia, ib = np.flatnonzero(sa), np.flatnonzero(sb)
            tree = cKDTree(np.c_[b_.x[ib], b_.y[ib]])
            d, k = tree.query(np.c_[a_.x[ia], a_.y[ia]], distance_upper_bound=1.0)
            m = np.isfinite(d)
            dx = a_.x[ia][m] - b_.x[ib][k[m]]
            dy = a_.y[ia][m] - b_.y[ib][k[m]]
            zx = np.clip((a_.x[ia][m] / a_.width * 3).astype(int), 0, 2)
            zy = np.clip((a_.y[ia][m] / a_.height * 3).astype(int), 0, 2)
            zones = {}
            for zi in range(3):
                for zj in range(3):
                    s = (zy == zi) & (zx == zj)
                    if s.sum() >= 10:
                        zones[f"{zi}{zj}"] = [float(np.median(dx[s])), float(np.median(dy[s]))]
            worst_zone = max((math.hypot(*v) for v in zones.values()), default=float("nan"))
            rows.append({"pair": f"{names[i]}-{names[j]}", "matched": int(m.sum()),
                         "median": [float(np.median(dx)), float(np.median(dy))],
                         "rms": math.hypot(madn(dx), madn(dy)), "worstZone": worst_zone, "zones": zones})
    (out_dir / f"cross-channel-{label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"label": label, "pairs": rows}


def summarize(reports: list[dict[str, Any]], cross: list[dict[str, Any]], out_dir: Path) -> str:
    lines = ["# Master evaluation summary", ""]
    overall = []
    for rep in reports:
        lines.append(f"## Filter {rep['filter']}")
        info = rep["info"]
        pm = info["photometric_model"]
        n = info["noise"]
        lines.append(f"alignment: {info['alignment']['matched']} stars, RMS {info['alignment']['rms']:.3f} px; "
                     f"photometric model P = {pm['a']:.4g}*O + {pm['b']:.4g} (ln-ratio MADN {pm['lnRatioMadn']:.3f}, {pm['stars']} stars); "
                     f"σ_ref = {info['background']['sigma_ref']:.4g}; per-pixel σ1 ours/PI = {n['sigma1_ours_Punits']/n['sigma1_pi']:.4f}; "
                     f"ρ1 PI {n['rho_pi']['x1']:.3f}/{n['rho_pi']['y1']:.3f}, ours {n['rho_ours']['x1']:.3f}/{n['rho_ours']['y1']:.3f}; "
                     f"runtime {info['timing_seconds']:.0f} s")
        lines.append("")
        lines.append("| family | metric | PI | ours | d | τ | CI | status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        fam_status: dict[str, list[str]] = {}
        for m in rep["metrics"]:
            def fmt(v):
                return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4g}"
            ci = "" if not all(math.isfinite(v) for v in m["ci"]) else f"[{m['ci'][0]:.3g}, {m['ci'][1]:.3g}]"
            lines.append(f"| {m['family']} | {m['metric']} | {fmt(m['pi'])} | {fmt(m['ours'])} | {fmt(m['d'])} | {fmt(m['tau'])} | {ci} | **{m['status']}** |")
            if m["verdict"]:
                fam_status.setdefault(m["family"], []).append(m["status"])
        g8 = info["noise"]["G"]["8"]
        g4 = info["noise"]["G"]["4"]
        dm = info["depth"]
        visual = [s for f in ("psf", "background", "artefacts") for s in fam_status.get(f, [])]
        not_worse = "FAIL" not in visual and visual.count("WARN") <= 3
        sci = "FAIL" not in fam_status.get("photometry", [])
        psf_pass = all(s == "PASS" for s in fam_status.get("psf", []))
        better_snr = g8[1] > 1.02 and g4[1] > 1.00 and dm["ci"][0] >= 0.02 and psf_pass
        if better_snr and not_worse and sci:
            verdict = "BETTER"
        elif not_worse and sci and (g8[1] <= 1.0 <= g8[2] or abs(g8[0] - 1) <= 0.02):
            verdict = "EQUIVALENT"
        elif not not_worse or not sci:
            verdict = "WORSE" if ("FAIL" in visual or not sci) else "INCONCLUSIVE"
        else:
            verdict = "INCONCLUSIVE"
        overall.append(verdict)
        lines.append("")
        lines.append(f"**{rep['filter']} verdict: {verdict}** — not visually worse: {not_worse}; scientifically consistent: {sci}; "
                     f"better in SNR: {better_snr} (G_8 = {g8[0]:.3f} CI[{g8[1]:.3f}, {g8[2]:.3f}], G_4 = {g4[0]:.3f}, Δm = {dm['dm']:+.3f} mag CI[{dm['ci'][0]:+.3f}, {dm['ci'][1]:+.3f}])")
        lines.append("")
    for c in cross:
        lines.append(f"## Cross-channel alignment ({c['label']})")
        lines.append("| pair | matched | median dx,dy | RMS | worst zone |")
        lines.append("|---|---|---|---|---|")
        for r in c["pairs"]:
            st = "PASS" if (math.hypot(*r["median"]) <= 0.05 and r["rms"] <= 0.10 and r["worstZone"] <= 0.10) else ("WARN" if r["worstZone"] <= 0.20 else "FAIL")
            lines.append(f"| {r['pair']} | {r['matched']} | {r['median'][0]:+.3f}, {r['median'][1]:+.3f} | {r['rms']:.3f} | {r['worstZone']:.3f} | **{st}** |")
        lines.append("")
    order = {"WORSE": 0, "INCONCLUSIVE": 1, "EQUIVALENT": 2, "BETTER": 3}
    run_verdict = min(overall, key=lambda v: order[v]) if overall else "N/A"
    lines.insert(2, f"**Run verdict: {run_verdict}** (worst per-filter verdict)")
    lines.insert(3, "")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("FILTER", "OURS", "PI"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cross-channel", action="store_true", help="also evaluate cross-channel alignment of each side")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = [evaluate_pair(f, o, p, out) for f, o, p in args.pair]
    cross = []
    if args.cross_channel and len(args.pair) > 1:
        cross.append(cross_channel("ours", {f: o for f, o, _ in args.pair}, out))
        cross.append(cross_channel("pi", {f: p for f, _, p in args.pair}, out))
    text = summarize(reports, cross, out)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
