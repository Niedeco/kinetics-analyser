# Kinetics analyser

A Tkinter + matplotlib GUI for extracting enzyme kinetics (slopes, k<sub>cat</sub>/K<sub>M</sub>) from Cary 60 UV-Vis CSV files. Built for the HG3 Kemp eliminase directed-evolution lineage; works on any continuous-monitoring assay where each cuvette swap produces a spike followed by a blank or enzyme trace at 380 nm.

The pipeline segments the trace into blank and enzyme regions, lets the user verify or hand-edit those regions, then fits initial-velocity slopes and a k<sub>cat</sub>/K<sub>M</sub> from k<sub>obs</sub> vs [S].

## Files

- `kinetics_pipeline.py` — pure-numeric pipeline (segmentation, classification, fitting). No GUI dependencies; importable from notebooks.
- `kinetics_gui.py` — Tkinter GUI on top of the pipeline. The main entry point.
- `environment.yml` — conda environment specification.

## Installation

The GUI needs Python 3.12, NumPy, pandas, SciPy, matplotlib, and tkinter. The `environment.yml` file pins everything from `conda-forge`.

```
conda env create -f environment.yml
conda activate kinetics
```

On Linux, the `tk` package in the environment provides the tkinter runtime; on Windows and macOS, tkinter ships with the conda Python build itself. If tkinter is somehow missing on Linux (`import tkinter` fails), install it system-wide: `sudo apt install python3-tk` on Debian/Ubuntu.

If you already have a Python environment you'd rather not duplicate, the same dependencies via pip:

```
pip install numpy pandas scipy matplotlib openpyxl
```

(`openpyxl` is only needed for the Excel export; everything else runs without it.)

## Running

From inside the activated environment:

```
python kinetics_gui.py
```

This opens an empty GUI. Click **Choose folder…** in the top bar and point it at a folder of Cary CSVs (one CSV per enzyme variant, with the filename serving as the variant name).

To skip the dialog and load a folder directly:

```
python kinetics_gui.py /path/to/folder
```

If you're on Windows and the folder lives on OneDrive, give the full local path (e.g. `C:\Users\you\OneDrive\Master\Masterthesis\Kinetics\rundata`).

## Input file format

The pipeline expects Cary 60 CSV exports with the usual structure:

```
Sample 1,,
Time (min),Abs,
0.0008333333,-4.084613829E-005,
...
```

The parser tolerates both `latin-1` and `utf-8`, handles the `"Sample 1"` header row, and reads `time (min)` and `Abs` columns. Each row should be a single sample point — 0.1 s sampling (10 Hz) is what the algorithms are tuned for. Trace length up to ~30 minutes is fine.

Each CSV holds one variant's full kinetics run: alternating blank/enzyme measurements at increasing substrate concentrations. The pipeline does NOT require strict alternation — extra blanks, missing blanks, and enzyme replicates without an interleaved blank are all handled correctly.

## Workflow

The GUI has four kinds of tabs:

- **Setup** — per-variant table of [E] (enzyme concentration in µM) and ε (extinction coefficient in M⁻¹cm⁻¹). Edit these before reading off any k<sub>cat</sub>/K<sub>M</sub> values, since they're used to convert raw slopes into k<sub>obs</sub>.
- **Parameters** — sliders/dropdowns for the pipeline's tunable parameters. Most users never need to touch this. Click **Re-extract slopes** after changing anything here.
- **One tab per variant** — the actual workflow. Each has a two-step layout (see below).
- **Plot** — overlaid k<sub>obs</sub> vs [S] plots for every confirmed variant. Aggregate view for comparing enzymes.

Each variant tab is split into two steps:

**Step 1 — Region selection.** A wide trace plot showing the full file with auto-detected segments shaded (blue = blank, red = enzyme, paired numbering B1/E1, B2/E2, ...). You can:

- **Lasso a new region**: click-drag on the trace, then click "Add as blank" or "Add as enzyme".
- **Drag a segment edge** to retime it.
- **Right-click a segment** to delete it.
- **Re-run auto-detect** or **Clear all** with the top buttons.

The matplotlib navigation toolbar gives you pan/zoom for crowded regions. Confirm the regions with **Confirm pairs →** to advance to step 2.

**Step 2 — Pair analysis.** Per-pair zoom panels with the linear fit drawn over each enzyme trace and its baseline-shifted blank. You can:

- **Click a panel** to toggle that pair's inclusion in the k<sub>cat</sub>/K<sub>M</sub> fit.
- **Drag a fit-window edge** to override the auto-chosen linear region.
- **Right-click a panel** to reset to the auto fit.

The result strip at the top shows the current k<sub>cat</sub>/K<sub>M</sub>, the intercept, R², and the number of included pairs. The table below lists per-pair slopes, R², windows, [S], v, k<sub>obs</sub>, and the include checkbox.

## Persistence

Region selections, [E], ε, per-pair concentrations, manual fit-window edits, and include flags are all saved to a sidecar file in the folder:

```
<folder>/.kinetics_session.json
```

This file is written after every confirmation and every per-pair edit. Closing and reopening the GUI on the same folder restores all confirmed variants exactly as you left them. Unconfirmed variants get fresh auto-detection.

The sidecar is plain JSON — if you ever need to inspect or back up the analysis state, that's the file.

Separately, `kinetics_defaults.json` (shipped alongside the code, not per-folder) holds the pipeline's default analysis parameters — segment/trim lengths, the fit method, `max_slope_dev`, bootstrap iteration count, and the k<sub>cat</sub>/K<sub>M</sub> regression mode. These are the values the GUI loads on startup, and they override the in-code defaults where the two differ (for example the shipped file sets `fit_method: curvature`, `kcat_method: per_replicate`, and `max_slope_dev: 0.01`). Edit this file to change the defaults applied to every folder.

## Default values

These are baked in as starting points but you should override them per experiment on the Setup tab:

- ε = 15784.925513164 M⁻¹cm⁻¹ (the value for the Kemp product chromophore at 380 nm, 1 cm pathlength)
- [E] = 1.0 µM (placeholder — set to your actual enzyme concentration per variant)
- Default substrate concentrations (only auto-applied when the variant has exactly 15 pairs): 20 / 37.5 / 75 / 150 / 300 µM, each in triplicate

If your assay uses different substrate concentrations or a different number of pairs, the auto-defaults won't fire and you'll need to enter them by hand on each variant tab.

## Pipeline parameters worth knowing

Most defaults are correct; the ones you might want to change on the **Parameters** tab:

- **Enzyme fit method** — `curvature` (default, noise-robust) or `r2_window` (older R²-based). The curvature method fits a quadratic to the full enzyme segment, checks whether the curvature is both statistically resolvable and practically large (>5% slope drop), and truncates the fit window only if both apply. Use `r2_window` if you specifically want a window selected by R² threshold.
- **Max slope deviation** — fraction by which the linear-fit slope is allowed to drop across the fit window before the curvature method truncates. Lower = stricter linearity, shorter windows. The shipped default (`kinetics_defaults.json`) is 0.01 = 1%; the in-code fallback is 0.05 = 5%.
- **Curvature significance σ** — how confident the quadratic test has to be (default 4σ) before it's allowed to truncate. Higher = more tolerant of slight curvature on clean traces.
- **Min fit window (s)** — floor on window length. Too short → noise-dominated slopes. The shipped default (`kinetics_defaults.json`) is 1 s; the in-code fallback is 5 s.

## Output / export

Click **Export results…** in the top bar. This writes a single Excel workbook, `kinetics_results.xlsx`, to a folder you choose. Excel export needs the `openpyxl` package (`pip install openpyxl`); the GUI prompts you if it's missing. The workbook contains:

- A **per-pair detail** sheet: one row per pair across all variants, with raw blank/enzyme/net slopes, R², [S], v, k<sub>obs</sub>, the include flag, and whether the fit window was manually adjusted.
- A **Setup** sheet: the [E] and ε used for each variant.
- **One sheet per variant**: the per-pair table plus a footer carrying the k<sub>cat</sub>/K<sub>M</sub>, the method used, its uncertainty (SD across replicates, weighted SE, or bootstrap 95% CI depending on the method), and n_used.

## Troubleshooting

- **GUI is sluggish / unresponsive with many files.** This shouldn't happen anymore — variant tabs are built lazily on first click. If a particular tab is slow to open the first time, that's the heavy matplotlib figure construction for that variant only; subsequent visits are instant.
- **Manual fit-window drag doesn't update visibly until I click elsewhere.** This was a known bug; the fix is in place. If it ever comes back, it's usually a focus/blit issue and forcing a fresh tab build (re-extract) clears it.
- **"No segments detected"** for a particular file. The auto-segmenter expects clear cuvette-swap spikes. If your trace doesn't have them (e.g. continuous-flow assay), use step 1's lasso to define regions manually.
- **k<sub>cat</sub>/K<sub>M</sub> looks wrong / negative.** Most often: [E] or ε is wrong on the Setup tab, or the included pairs include a saturated tail. Check the per-pair k<sub>obs</sub> column — they should ascend roughly linearly with [S] for a well-behaved enzyme.

## Algorithm notes

The pipeline does the following, in order:

1. **Spike detection.** Points where the absorbance moves more than 0.03 AU between consecutive samples, or drops below -0.05 AU, are flagged as cuvette-swap disturbances and dilated by a small buffer. Continuous regions between dilated spikes become segments.
2. **Segment classification.** Each segment's absorbance change |ΔA| is computed across the post-spike, post-tail interior. The largest ratio gap in the sorted ΔA values (above a noise floor of 0.005 AU) separates blanks from enzymes; if there's no clean gap, a fixed threshold of ΔA > 0.015 AU and slope > 0.025 AU/min is used.
3. **Enzyme onset refinement.** Each enzyme segment's start is walked back to the absorbance minimum after the cuvette swap (the actual reaction onset).
4. **Blank slope.** A linear fit on the segment interior with the head transient detected and skipped (the few hundred ms of ringing after a spike).
5. **Enzyme slope.** Either the curvature method (default) or the R²-window method, applied to the trimmed segment.
6. **Pair construction.** Each enzyme segment is paired with the most-recently-preceding blank segment.
7. **k<sub>obs</sub> per pair.** v[µM/s] = slope_AU_per_min / (ε / 1e6) / 60; k<sub>obs</sub> = v / [E]_µM.
8. **k<sub>cat</sub>/K<sub>M</sub>.** Linear fit of k<sub>obs</sub> vs [S], free intercept; slope × 1e6 gives the answer in M⁻¹s⁻¹. Three regression modes are available, selected by `kcat_method`:
   - `weighted` — replicate-aware weighted least squares. Each concentration with three or more replicates is weighted by 1/σ² from its triplicate SD; singletons and sparse concentrations fall back to the residual SD of a preliminary OLS fit. Reports a weighted slope SE.
   - `per_replicate` (shipped default) — fits each technical replicate separately, then averages the per-replicate slopes and reports the SD across them as the uncertainty.
   - `bootstrap` — replicate-clustered bootstrap (default 10 000 iterations) giving a 95% percentile CI. With few replicates the resampling space is small, so the bootstrap distribution is discrete by construction.

A large intercept on the k<sub>obs</sub> vs [S] fit usually indicates baseline contamination — either bad blanks or pairs that shouldn't be included.
