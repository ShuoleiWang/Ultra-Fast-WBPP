# Getting started

A first run of Ultra-Fast WBPP, step by step, for someone who shoots with N.I.N.A. and may own PixInsight. It ends with linear, plate-solved masters in a new folder. The FAQ at the end answers what people ask before the first run. 简体中文：[getting-started.zh-CN.md](getting-started.zh-CN.md).

## What you need

- A Mac with Apple Silicon on macOS 15 or later, or a Windows 10/11 x64 PC. Real-data runs are validated on one M3 Pro and one Ryzen 7 5800H laptop; other machines run generic profiles ([validation matrix](validation-matrix.md)).
- Your Lights as FITS or XISF, plus raw Flat/Dark/Bias frames or existing masters (PixInsight WBPP's XISF masters work). DSLR RAW files are not supported.
- Free disk space on the output volume: about 12 bytes per pixel per Light for the full-resolution intermediates (a 26 MP camera: about 310 MB per Light).
- A plate solver on this computer (step 2). Nothing is bundled.

## 1. Get the app

Releases will appear on the [Releases page](https://github.com/ShuoleiWang/Ultra-Fast-WBPP/releases). Today the way in is building from source ([guide](building-from-source.md)). On a Mac:

```bash
git clone https://github.com/ShuoleiWang/Ultra-Fast-WBPP.git
cd Ultra-Fast-WBPP
make bootstrap BOOTSTRAP_PYTHON=python3.12
make desktop-dev                                     # runs the app from the clone
# or, for a local app bundle:
CARGO_PROFILE_RELEASE_STRIP=none make desktop-build-macos-prerelease
```

Windows builds are described in [windows.md](windows.md#developing-on-windows).

**First launch.**

- macOS: the app is ad-hoc signed and not notarized. On macOS 15 and later the right-click → Open bypass no longer works. Open System Settings → Privacy & Security and press **Open Anyway**, or remove the quarantine flag: `xattr -dr com.apple.quarantine "Ultra-Fast WBPP.app"`.
- Windows: the installer is unsigned. SmartScreen shows "Windows protected your PC"; choose More info → **Run anyway**.

On first launch the interface follows the system language: Simplified Chinese when the system is set to Chinese, English otherwise. The language selector is in the toolbar; the choice is remembered.

## 2. Set up the plate solver

Every master is plate-solved and verified before it is published, so the app needs a solver on each computer it runs on ([why nothing is bundled](licensing.md)).

1. **Install the solver.**
   - macOS: `brew install astrometry-net`, which provides `solve-field`.
   - Windows: install [ASTAP](https://www.hnsky.org/astap.htm) (`astap_cli.exe`) and unpack a star database such as D20 or D50 into the ASTAP folder ([Windows](windows.md)).
2. **Download the index set in the app.** Open *Advanced options and solver setup*, read and accept the provider terms, and download the managed Astrometry.net index set (about 350 MB). It is verified by hash and recorded. Then press **Recheck setup**; the solver card should read *READY*.
3. **Optional: the catalog refit.** With the Gaia DR2 LITE tiles installed (587 MiB), the published WCS is refitted on Gaia stars with a TAN-SIP model of up to order 3, about 0.15″ on held-out stars on real masters, where the solver's own solution is off by about 2.5″ on a fast Newtonian ([astrometry](recipes/astrometry.md)). Mosaic panels with too few Tycho-2 stars need them to be placed. Once the index set is installed, the same section shows the card *Gaia DR2 tiles (optional, recommended for mosaics)*: accept the terms and download. The engine's command line installs them too:

   ```bash
   ultra-fast-wbpp catalog list
   ultra-fast-wbpp catalog install astrometry-net-5205-lite --accept-provider-terms <id printed by catalog list> --progress-json
   ```

   In a source checkout the command is `.venv/bin/ultra-fast-wbpp`. In a built macOS app the engine is `Ultra-Fast WBPP.app/Contents/Resources/resources/ufwbpp-engine/ufwbpp-engine-aarch64-apple-darwin/ufwbpp-engine-aarch64-apple-darwin`. Details: [offline solver catalogs](recipes/offline-solver-catalogs.md).

Moving to another computer: copy the catalog folder (`<data root>/catalogs/astrometry-net`) to the same data root there, then open the app or run `ultra-fast-wbpp catalog doctor`; the engine re-points its config and receipts to the new place, while the solver program itself must be installed on each computer ([moving the catalog](recipes/offline-solver-catalogs.md)).

## 3. Import your nights

Drop everything at once onto the window: the Lights of every night, raw Flat/Dark/Bias or existing masters, whole N.I.N.A. folder trees, or a folder you prepared for PixInsight WBPP.

- Frame types, targets, filters, exposure, gain and camera geometry are read from the headers. Frames whose type is not in the header are named by their folder or file name, as WBPP does; conflicts are shown, never guessed. Confirm any type the app asks you to confirm.
- Calibration is paired by WBPP's rules, night by night: `NIGHT_1`, `DATE_2026-03-22`, `SESSION-2` folders keep each night's Flats with that night's Lights, and a Bias or Dark library outside the night folders serves every night. Compromises (a Dark of another exposure, a Flat without its own Dark) are shown as warnings. Lights that WBPP already calibrated or registered are refused; import the raw Lights ([calibration](recipes/calibration.md)).
- A supplied Master Dark is taken to include the Bias; say otherwise in the advanced calibration settings if yours was bias-subtracted.
- The *Calibration check* in the inspector tells you what is matched, what is missing and what is ambiguous.
- Choose an output folder; the results go into a new directory inside it, and the choice is remembered. It may be on an external exFAT disk. Your source files are never written to.

## 4. Screening: automatic, or Blink

*Automatic screening* in the launch bar is checked by default. Then nothing has to be reviewed: the run measures every Light, leaves out what the pipeline cannot use (cloud, a moonlit night, trails, bad focus, a frame off the field) and weights the rest, and the result page lists every Light it left out with the reason and a preview. Channels it was unsure about are named under *Worth a look in Blink*. A mosaic works the same way: drop every panel's folders at once and press Start ([automatic screening](recipes/automatic-screening.md)).

To decide every frame yourself, uncheck *Automatic screening* (the choice is remembered), or press **Review in Blink instead**, which unchecks it. Then press **Blink & select (N Lights)**. The app measures every Light and opens a paused, chronological review, one channel (target × filter) at a time.

- Every Light of a channel is shown aligned to the channel's reference frame, with one shared stretch, so what changes between frames is the sky, not the display.
- The flags beside each frame are measured: sky brightness against the channel's clean nights, star count, extinction, native FWHM, ellipticity, registration quality, gradient. Each flag states its value and threshold. They are hints; nothing is dropped for you.
- Step with `←` `→`, play with `P`, jump to flagged frames with `F`, hold `C` to A/B against the reference. `K` keeps and moves on, `D` drops and moves on, `N` drops the whole night, `Z` undoes. `1`–`9` or `Tab` switch channels, `Space` toggles keep/drop, `R` jumps to the reference. The night header has **Keep night** / **Drop night**.
- When every frame of the channel has been displayed, press **Confirm &lt;filter&gt; and continue**. Changing a decision afterwards re-opens that channel's confirmation. A preview that fails to display is not counted as viewed; retry it or drop the frame explicitly.
- Each channel needs at least two kept Lights.

The flags, thresholds and the reference-frame rule are documented in [blink screening](recipes/blink-screening.md).

## 5. Process

**Start processing** becomes available when the solver is ready and an output folder is chosen, and, with automatic screening unchecked, when every Light has been reviewed and every channel confirmed; until then the launch bar lists what is missing. The defaults are the recommended settings. Under *Advanced options and solver setup* you can turn on drizzle (1×–4×), turn off the super luminance, or enable the opt-in algorithms.

The run re-measures every Light, applies automatic screening or exactly your Blink decisions, builds or reuses the calibration masters, registers, normalizes, integrates with robust rejection, plate-solves each channel and verifies the solutions. Progress is live per stage. On an M3 Pro, 61 Lights of 26 MP over four filters take about 76 s; the same project on a Ryzen 7 5800H laptop takes 4.5 minutes.

A run that fails publishes nothing. A run that fails or that you cancel leaves no hidden staging folder beside the output; the app removes it. If a channel does not solve, the attempt's receipt and evidence are kept in `<output>.unsolved` and no science product is written.

## 6. The results

For an ordinary (single-panel) project the new directory contains:

```
<output>/
  L.fits  R.fits  G.fits  B.fits   # one linear master per filter, verified WCS in the header
  RGB.fits or LRGB.fits            # when R, G and B exist (LRGB when L exists too)
  SuperL.fits                      # super luminance of the broadband channels (on by default)
  previews/                        # stretched PNG previews for inspection
  receipt.json                     # inputs, decisions, algorithm ids, hashes, the accepted solve
  details/                         # shared-calibration/, runs/<target>/ with per-stage receipts
```

A mosaic project has one mosaic per filter on the shared canvas and `details/mosaics/<filter>/` ([mosaics](recipes/project-mosaic-rgb.md)). Drizzle adds science, weight and coverage products ([drizzle](recipes/drizzle.md)).

The masters are linear and unstretched. Open them in PixInsight, Siril or any FITS viewer and post-process as you would WBPP's masters; the WCS is already in the header and was verified against catalog stars. `SuperL.fits` is the broadband channels combined into one deeper luminance, each weighted by its noise, with a channel whose stars are clearly worse left out; use it as you would a luminance master, or ignore it.

## FAQ

**Do I need PixInsight?** No. Ultra-Fast WBPP replaces the preprocessing that WBPP does. For stretching, gradient removal, deconvolution and colour calibration you still need a post-processing tool; the masters open in PixInsight as ordinary linear FITS.

**Is my data uploaded anywhere?** No. The solver, the indexes and the engine run on your computer. The only downloads are the index files you ask for.

**Can I import the folder I prepared for WBPP?** Yes, as it is. Frame types come from the headers or from the folder names (`Lights`, `Flats`, `Darks`, `Bias`, `DarkFlats`), the grouping keywords `NIGHT`, `DATE`, `SESSION` and `PANEL` are read as WBPP reads them, and WBPP's XISF masters are used as supplied masters. WBPP's calibrated or registered Lights are refused ([calibration](recipes/calibration.md#files-from-pixinsight-wbpp)).

**Why is "Start processing" disabled?** The launch bar says why: with automatic screening unchecked, a channel not fully reviewed or not confirmed, or a channel with fewer than two kept Lights; a channel with fewer than two Lights, the solver or index set not ready, no output folder, an unconfirmed frame type, or an unsaved calibration override.

**My camera is a one-shot-colour camera.** Bayer Lights are recognised and processed into R, G and B channel masters and an RGB product, with Bayer drizzle available ([OSC / CFA](recipes/osc-cfa.md)). This path is validated on synthetic data only; no real OSC data set has been run yet.

**Narrowband, or only two filters?** Yes. Each filter becomes its own master; a project without a full R, G, B set publishes its masters without a colour product.

**Several panels of a mosaic?** Import all panels together and press Start. N.I.N.A. panel names (`M 31 Panel 3`) or `PANEL_3` folders split them; the panels are placed by one astrometric fit of all of them on one shared canvas and matched by their overlap stars. Validated on real M31 mosaics of 2×2, five and six panels ([mosaics](recipes/project-mosaic-rgb.md)). Mosaic panels need the Gaia tiles when the Tycho-2 set has too few stars in a panel; the app says so and points to their card under *Advanced options and solver setup*.

**The run says it could not solve a channel.** The attempt is kept in `<output>.unsolved`. Check that the solver and the index set are ready (**Recheck setup**), that the Lights' pixel scale lies within the installed indexes' range, and look at the receipt in the `.unsolved` directory. A filter whose own solve misses a gate takes its grid's solve when another filter solved, verified by the catalog refit; that needs the Gaia tiles.

**Why does my run take longer than 76 s?** That number is 61 Lights of 26 MP on an M3 Pro. Time scales with the number and size of Lights and with per-core speed and memory bandwidth (a Ryzen 7 5800H laptop takes 4.5 minutes for the same project). The Gaia catalog refit adds about 15 s on that project (79 s against 64 s). Drizzle is a second integration: 2× drizzle of all four filters took 178 s, including 1 GB of drizzle products, when the ordinary run took 97 s ([features §1](features.md#1-speed-76-seconds-for-61--26-mp-lights)).

**Does the result change if I run it twice?** Mono masters are bit-identical on the same machine. RGB pixels are identical while the file's checksum cards differ. macOS and Windows masters differ by math-library rounding and are compared with a tolerance gate, not by hash.

**How good are the masters compared with WBPP's?** They are measured against WBPP's masters of the same frames with a published standard: historical binned noise gain G₈ of 1.01–1.03 on all four filters, PSF, background and photometry within tolerance; full equivalence is not certified because some standard gates are unmeasured ([features §5](features.md#5-master-quality-against-pixinsight-wbpp), [standard](master-evaluation-standard.md)).

**Do I have to blink every frame?** No. Automatic screening (`unattended-v2`) is the desktop's default and needs no review; Blink is for deciding yourself. On the command line the default stays `legacy-gate`, and `{"selection": {"policy": "unattended-v2"}}` in the recipe selects automatic screening ([automatic screening](recipes/automatic-screening.md)).

**Where are the index files, and can I copy them to another computer?** `~/.ultra-fast-wbpp/catalogs/astrometry-net` on macOS, `%LOCALAPPDATA%\Ultra-Fast-WBPP\catalogs\astrometry-net` on Windows (an installation older than the rename keeps `~/.openastroflow` or `%LOCALAPPDATA%\OpenAstroFlow`); `ultra-fast-wbpp catalog doctor` prints it. Copy that folder to the same place on the other computer and open the app; the engine re-points its config to the new place, and the solver program itself must be installed there ([moving the catalog](recipes/offline-solver-catalogs.md)).

**Where do I report a problem?** Open an issue with the `receipt.json` (it holds no pixel data; look through it for file paths you do not want to share) and, if the run failed, the receipt from the `.unsolved` directory. [CONTRIBUTING.md](../CONTRIBUTING.md) describes the setup for fixing it yourself.
