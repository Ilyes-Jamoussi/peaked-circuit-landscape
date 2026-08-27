"""Digitize the tau_p = tau_r/2 series of Ref. [aaronson2024peaked] Fig. 3c.

Their figure is published as a raster embedded in the PDF, with no tabulated
values, so the five instance-averaged points of the operating regime are read
off the panel. This script is the reading method: everything quoted in the
manuscript's Fig. [fig:scaling] overlay comes out of it, and rerunning it
against the arXiv PDF reproduces the committed numbers.

Method
------
1. The page-10 image is extracted; panel c is its bottom-left quadrant.
2. The axes are calibrated on the tick marks, not on the frame: five x ticks
   at n = 8, 10, 12, 14, 16, and log-y major ticks at one decade per 210 px
   with 10^0 on the top spine.
3. Each marker is a matplotlib '^' triangle, rendered 21 px tall with a
   21 px base in this raster (the legend glyph shows the same shape), and
   centred on its data point. The per-row color width of the series
   (RGB (97, 15, 161)) near each tick column is fitted to the triangle
   template width(row) = row - apex + 1 by least squares; rows fattened by
   the fitted line drop out of the fit. The data point is the centre of the
   fitted 21-row extent. An intensity centroid is not used: it mixes the
   fitted line into the blob and sits below the marker centre by up to 9%
   at n = 12.
4. The reading uncertainty is the half-height of the marker in data units
   (about 12% of the value), which dominates the one-pixel calibration
   error. It is a digitization uncertainty; their statistical error is
   stated in their caption as too small to be visible, and the raster
   resolves neither.

Self-check: the digitized points give a mean base of 1.20 per qubit over
n = 8..16 against the 1.189 of their own fit.

Usage:
    python analysis/az_fig3c_digitize.py path/to/aaronson-zhang.pdf
"""

from __future__ import annotations

import sys

import numpy as np

X_TICK_VALUES = (8, 10, 12, 14, 16)
SERIES_RGB = np.array([97, 15, 161])   # the tau_p = tau_r/2 triangles
COLOR_TOLERANCE = 90                   # L1 distance in RGB
MARKER_HEIGHT = 21                     # px; height and base width of the
                                       # legend glyph in the same raster


def digitize(pdf_path: str) -> dict[int, tuple[float, float]]:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    page = doc[9]                      # page 10: Figure 3
    info = page.get_images(full=True)[0]
    pix = pymupdf.Pixmap(doc, info[0])
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].astype(int)

    # Panel c: bottom-left quadrant.
    panel = img[img.shape[0] // 2:, : img.shape[1] // 2]
    dark = panel.sum(axis=2) < 200

    # Frame and ticks. The top divider line between the panel rows is
    # skipped when locating the frame's top spine.
    ph = panel.shape[0]
    top = int(np.argmax(dark[30:, :].sum(axis=1)[: ph // 2]) + 30)
    bottom = int(np.argmax(dark[ph // 2:, :].sum(axis=1)) + ph // 2)
    left = int(np.argmax(dark.sum(axis=0)[: panel.shape[1] // 2]))

    band = dark[bottom + 4: bottom + 12, :]
    cols = np.flatnonzero(band.sum(axis=0) >= 6)
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 3) + 1)
    x_ticks = [int(g.mean()) for g in groups if len(g)]
    assert len(x_ticks) == len(X_TICK_VALUES), x_ticks

    band = dark[:, max(0, left - 15): left - 3]
    rows = np.flatnonzero(band.sum(axis=1) >= 4)
    rows = rows[(rows > top + 5) & (rows < bottom - 2)]
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
    y_ticks = [int(g.mean()) for g in groups if len(g)]
    # Major ticks are one decade apart; 10^0 sits on the top spine itself.
    per_decade = float(np.mean(np.diff(y_ticks)))
    y_zero = y_ticks[0] - per_decade   # row of 10^0
    assert abs(y_zero - top) < 3, (y_zero, top)

    match = np.abs(panel - SERIES_RGB).sum(axis=2) < COLOR_TOLERANCE
    points: dict[int, tuple[float, float]] = {}
    H = MARKER_HEIGHT
    for value, x in zip(X_TICK_VALUES, x_ticks):
        width = match[:, x - 12: x + 12].sum(axis=1).astype(float)
        width[: top + 5] = 0
        if value == 8:                 # keep clear of the legend box
            width[400:] = 0
        rows = np.flatnonzero(width)
        best, apex = None, None
        for a in range(rows.min() - H, rows.max() + 1):
            r = np.arange(a, a + H)
            r = r[(r >= 0) & (r < len(width))]
            template = np.clip(r - a + 1, 0, H).astype(float)
            measured = width[r]
            keep = measured <= template + 4   # line-fattened rows drop out
            if keep.sum() < H // 2:
                continue
            sse = float(((measured - template)[keep] ** 2).mean())
            if best is None or sse < best:
                best, apex = sse, a
        row = apex + (H - 1) / 2
        delta = 10 ** (-(row - y_zero) / per_decade)
        half = (H / 2) / per_decade
        points[value] = (delta, delta * (10 ** half - 1))
    return points


if __name__ == "__main__":
    for n, (delta, err) in digitize(sys.argv[1]).items():
        print(f"n = {n:2d}   delta = {delta:.4f} +/- {err:.4f} (reading)")
