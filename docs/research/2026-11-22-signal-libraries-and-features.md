# Libraries, features and band edges for band features on DEAP

Snapshot of 2026-11-22. Question (ticket `.scratch/ml-rebuild/issues/10-signal-libraries-and-features-for-band-features.md`): for a within-subject classifier on DEAP's preprocessed arrays (32 channels, 128 Hz, 4 s windows, about 480 training windows per participant), which library and which feature are the current, well-regarded way to compute band features, and are our band edges standard?

This builds on `2026-11-22-braindecode-mne-deap-pipeline.md` (branch `research/ml-rebuild-braindecode-mne-deap`) and does not repeat it.

Every claim is marked with where it comes from:

- **[installed]**: read from the backend virtualenv, read-only (Python 3.14.8, `torch 2.15.0`, `braindecode 2.0.0`, `scipy 1.18.1`, `numpy 2.5.5`, `mne 1.13.2`, `scikit-learn 1.9.1`, `numba 0.68.1`, `pandas 3.1.1`).
- **[tested]**: run in a throwaway virtualenv on the same Python with `torch==2.15.0 braindecode==2.0.0 scipy==1.18.1 numpy==2.5.5` plus `librosa` and `transformers`, then the library under test.
- **[measured]**: computed on `backend/data/deap/s01.dat` (600 windows of 32 × 512) in that virtualenv. One participant, one run, the owner's Mac: indicative only.
- **[pypi]**: the package's PyPI JSON (`https://pypi.org/pypi/<name>/json`), read 2026-11-22.
- **[doc]** / **[paper]**: a linked first-party page or the paper's own text.

Section 6 lists what was not verified. No model was trained for this note.

## Short answer

- **There is no industry-standard library for "band features from epoched EEG arrays".** MNE-Python is the standard EEG toolbox and is already installed (braindecode depends on it), but for this job it offers only spectrum estimators; it has no band-feature function and no differential entropy. The one library that ships differential entropy with DEAP's conventions, torcheeg, cannot be installed next to scipy 1.18. Everything else is either a thin wrapper over `scipy.signal.welch` or built for another purpose (sleep, complexity measures, BCI covariance pipelines).
- **Differential entropy is the most used feature in this literature, not a demonstrably better one on DEAP.** It is 0.5 × log of band power, so it carries the same information as log band power from Welch or multitaper. The papers that show it beating raw band power are on SEED, not DEAP. In trial-safe DEAP results, band power, differential entropy and Riemannian covariance features all land within a few points of each other and of the majority baseline.
- **Our band edges are within the range the field uses.** There is no single canonical set; theta 4 to 8 and beta ending at 30 are near-universal, the alpha/beta edge is 12, 13 or 14 depending on the source, and dropping delta on DEAP is what torcheeg and Khan et al. do.
- **Recommendation:** keep plain scipy and keep differential entropy. It needs about 12 lines, adds no dependency, and matches what the literature calls the feature. The main alternative, if a second feature family is wanted later, is pyRiemann's covariance and tangent-space pipeline (three small packages, installs cleanly).

## 1. Libraries

### What each offers, and whether it installs

"Adds" is what `uv pip install` would add to the throwaway virtualenv, which already holds the repo's stack (so `mne`, `numba`, `scikit-learn`, `pandas`, `matplotlib` and `joblib` count as present; they arrive today through braindecode and librosa, and none of them is a direct dependency in `backend/pyproject.toml`).

| Library | Latest release [pypi] | Installs and imports on Python 3.14 with torch 2.15, braindecode 2.0, scipy 1.18 [tested] | Adds | What it gives this job |
|---|---|---|---|---|
| scipy | 1.18.1, 2026-08-21 | Already a direct dependency | nothing | `signal.butter`, `sosfiltfilt`, `welch`, `periodogram`. Everything the current code uses. |
| MNE-Python | 1.13.2, 2026-09-11 | Already installed (through braindecode) | nothing | `mne.time_frequency.psd_array_welch` and `psd_array_multitaper` take plain arrays `(..., n_times)` and return a spectrum; summing it into bands is left to the caller. No band-power function, no differential entropy. `mne.decoding` has `CSP`, `SPoC`, `PSDEstimator`, `FilterEstimator`. [installed] |
| braindecode | 2.0.0, 2026-11-02 | Already a direct dependency | nothing | No feature extraction: its modules are `augmentation`, `datasets`, `models`, `preprocessing`, `training` and so on. [installed] |
| mne-features | 0.3.2, 2026-02-11 (before that 0.3.1 on 2025-11-20 and 0.3 on 2023-09-05) | Yes | `mne-features`, `pywavelets` | `extract_features(X, sfreq, ["pow_freq_bands"], params)` on `(n_epochs, n_channels, n_times)`, and a scikit-learn `FeatureExtractor`. `pow_freq_bands` is Welch (or multitaper) power per band, optionally logged. Also Hjorth, spectral entropy, fractal dimensions, and a few bivariate features. No differential entropy. [API list](https://mne.tools/mne-features/api.html). 164 GitHub stars, last push 2026-03-16. |
| pyRiemann | 0.12, 2026-07-01 (four releases in the last 12 months) | Yes | `pyriemann`, `array-api-compat`, `array-api-extra` | `Covariances` (with shrinkage estimators) and `TangentSpace` as scikit-learn transformers, `MDM`, `FgMDM`, `TSClassifier`. Not a band-feature library: it turns each window into a channel covariance matrix. [Introduction](https://pyriemann.readthedocs.io/en/latest/introduction.html). 785 stars, last push 2026-11-04. |
| torcheeg | 1.1.3, 2024-12-23 | **No.** 1.1.3 declares `scipy>=1.7.3,<=1.10.1`, so the resolver refuses it beside scipy 1.18.1. It falls back to 1.1.2, which installs and then fails on import: `cannot import name 'hann' from 'scipy.signal'`. Without the scipy pin, uv would downgrade numpy to 1.26.4, scipy to 1.10.1, mne to 1.9.0 and pandas to 2.3.3. | 20+ packages, including `pytorch-lightning`, `lmdb`, `spectrum`, `mne-connectivity`, `netcdf4`, `xarray` | `transforms.BandDifferentialEntropy`, `BandPowerSpectralDensity`, and the DEAP loader. The only library here that ships differential entropy. The [repository](https://github.com/torcheeg/torcheeg) is active (last push 2026-11-04, and `main` already imports `hann` from `scipy.signal.windows`), but none of that is released. |
| YASA | 0.8.1, 2026-10-25 | Yes | `yasa`, `antropy`, `lightgbm`, `lspopt`, `seaborn` | `bandpower_from_psd_ndarray(psd, freqs, bands)`: integrates a Welch spectrum over bands. A sleep-staging library; its default bands include sigma (12 to 16 Hz). |
| antropy | 0.2.3, 2026-10-17 | Yes | `antropy` | Entropy and fractal measures of a time series (permutation, spectral, sample entropy, Hjorth). No band power and no Gaussian differential entropy. |
| NeuroKit2 | 0.3.0, 2026-11-01 | Yes, but it requires `setuptools<82` and downgraded setuptools from 84.2.1 to 81.0.0 | `neurokit2`, `pywavelets` | `eeg_power(eeg, sampling_rate, frequency_band)`: band power for an MNE object or data frame. A general biosignal library (ECG, EDA, respiration); its EEG module has nine functions. |
| eegdash | 0.9.1, 2026-09-10 | Resolves (dry run only) | 20 packages, including `boto3` and `zarr` | `eegdash.features` extracts features from braindecode dataset objects ([overview](https://eegdash.org/api/features_overview.html)). Its job is fetching datasets from a cloud archive. Not installed or run. |
| LibEER | not checked on PyPI | Not tested | not checked | A research benchmark that extracts differential entropy for DEAP ([README](https://github.com/XJTU-EEG/LibEER)). A benchmark codebase, not a feature library. |

The repo runs mypy strict. None of the candidates was checked for type stubs; scipy and braindecode already need an `ignore_missing_imports` override.

### Is there an industry standard?

No, not for this step.

- **MNE-Python is the standard general EEG toolbox.** Evidence that can be checked: braindecode, mne-features and YASA all declare it as a dependency [pypi], and it is the most starred of these repositories (3,614; last push 2026-11-20). It is already in the virtualenv. But for epoched arrays it stops at the spectrum.
- **mne-features is the MNE organization's feature library, and it is small.** It works, it installed, and its API is the closest fit to "features from epochs". It also went from September 2023 to November 2025 without a release, has 164 stars, and does not have the feature this literature uses.
- **pyRiemann is the well-regarded library for a different approach.** The MOABB benchmark of 30 pipelines on 36 datasets found that "Riemannian approaches utilizing spatial covariance matrices exhibit superior performance" ([arXiv 2404.15319](https://arxiv.org/abs/2404.15319)). Those datasets are motor imagery, P300 and SSVEP; none is an emotion dataset.
- **torcheeg is the emotion-recognition library, and it is not usable here** (table above).

### Does adopting one shorten the code?

The differential-entropy part of [`ml/preprocessing.py`](../../backend/src/cortexdj/ml/preprocessing.py) is `bandpass_filter`, `compute_differential_entropy` and `extract_features`: about 40 lines with docstrings and annotations. The computation itself is this [measured, runs as written]:

```python
def differential_entropy(windows, fs=128):  # (n, 32, 512) -> (n, 32, 4)
    out = []
    for low, high in BANDS.values():
        sos = signal.butter(5, [low, high], btype="band", fs=fs, output="sos")
        out.append(0.5 * np.log(np.var(signal.sosfiltfilt(sos, windows, axis=-1), axis=-1)))
    return np.stack(out, axis=-1)
```

- **mne-features** replaces it with one `extract_features` call and a parameter dict (about 5 lines). That saves under 10 lines, changes the feature to Welch log power, adds two packages, and makes the repo rely directly on `mne`, `numba` and `pandas`.
- **MNE's `psd_array_welch`** does not shorten it: the band sum is still ours, and scipy's `welch` does the same thing.
- **pyRiemann** is shorter than anything else (`make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression())`), but it is a different feature, covered in section 2.

Two details of the scipy code, for whoever rewrites it:

- The current code uses the `(b, a)` filter form. The second-order-sections form above is what scipy's own documentation recommends for general filtering ([`scipy.signal.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)). On `s01` the two agree to 2e-8 [measured], so this is idiom, not a fix.
- Filtering each 4 s window on its own (which is what inference on a single window must do) against filtering the whole 60 s trial and then cutting windows: the features correlate 0.995 (theta) to 0.9998 (gamma) [measured]. Edge effects are not a reason to change the design.

### Cost of each route on one participant [measured]

600 windows, 32 channels, four bands, wall time in seconds:

| Route | Output | Time |
|---|---|---|
| scipy Butterworth `sosfiltfilt` + 0.5 log variance | 600 × 32 × 4 | 0.62 |
| scipy `welch` (2 s segments, 50% overlap) + 0.5 log band power | 600 × 32 × 4 | 0.14 |
| MNE `psd_array_welch` + 0.5 log band power | 600 × 32 × 4 | 0.72 |
| MNE `psd_array_multitaper` (2 Hz bandwidth) + 0.5 log band power | 600 × 32 × 4 | 0.27 |
| mne-features `pow_freq_bands` (log) | 600 × 128 | 0.23 |
| pyRiemann `Covariances("oas")` + `TangentSpace` | 600 × 528 | 0.44 |
| pyRiemann, same per band (four bands) | 600 × 2112 | 2.08 |

Every route is under a second per participant except the last. Speed does not separate them.

## 2. Features

### What differential entropy is, and how libraries compute it

- **Definition.** For a Gaussian signal, `h(X) = 0.5 * log(2 * pi * e * sigma^2)` [paper: Duan, Zhu and Lu 2013, eq. 1, [PDF](https://bcmi.sjtu.edu.cn/home/lubaoliang/papers/2013/2013-7.pdf)]. The same paper states: "for a fixed length EEG sequence, DE is equivalent to the logarithm ES in a certain frequency band", where ES is the energy spectrum, citing Shi, Jiao and Lu 2013. So differential entropy is log band power with a constant added and a factor of one half.
- **Two computations are in use.**
  - *From a spectrum.* The SEED papers use "a 512-point short-time Fourier transform with a non-overlapped Hanning window of 1s" and take the log of band energy [paper: Duan 2013; Zheng and Lu 2015, [PDF](https://bcmi.sjtu.edu.cn/~blu/papers/2015/2015-16.pdf)]. torcheeg's `BandDifferentialEntropyV1` does this: Hann window, FFT, mean magnitude in the band, `log2(100 * x)` ([source](https://github.com/torcheeg/torcheeg/blob/main/torcheeg/transforms/numpy/band.py)).
  - *Band-pass, then log-variance.* torcheeg's default `BandDifferentialEntropy`: fifth-order Butterworth band-pass with `scipy.signal.lfilter` per channel, then `0.5 * log2(2 * pi * e * var)`. Its docstring calls this "a widely accepted differential entropy calculation method by the community, which is often applied to the DEAP and DREAMER datasets" (same source file).
- **Ours is the second kind.** It differs from torcheeg's default in three ways: natural log without the constant (an offset and a scale, which standardization removes), zero-phase `filtfilt` where torcheeg uses one-pass `lfilter`, and the band edges (section 3).
- **`scipy.stats.differential_entropy` is not this.** It is a non-parametric estimator from order statistics [installed: signature has `window_length` and `method`], not the Gaussian closed form.
- **LibEER** extracts "1-second DE (Differential Entropy) features from the DEAP dataset" with `feature_type="de_lds"` ([README](https://github.com/XJTU-EEG/LibEER)); its formula was not read.

### How the candidate features relate [measured on `s01`]

Correlation with Butterworth differential entropy, over 600 windows × 32 channels, per band (theta, alpha, beta, gamma), each alternative expressed as 0.5 × log band power:

| Alternative | theta | alpha | beta | gamma |
|---|---|---|---|---|
| scipy `welch`, 2 s segments | 0.963 | 0.953 | 0.959 | 0.988 |
| MNE `psd_array_welch` | 0.967 | 0.957 | 0.962 | 0.989 |
| MNE multitaper, 2 Hz bandwidth | 0.994 | 0.993 | 0.991 | 0.998 |
| scipy Hann periodogram (one 4 s segment) | 0.901 | 0.880 | 0.909 | 0.971 |
| mne-features `pow_freq_bands`, log | 0.932 | 0.911 | 0.927 | 0.976 |

- **Welch and multitaper log band power are close to the same number as differential entropy.** Multitaper is nearly identical; Welch differs mostly through spectral leakage at the band edges in a 4 s window.
- **Left-right asymmetry is contained in differential entropy for a linear model.** The usual "DASM" feature is `DE(left) - DE(right)` [paper: Duan 2013, eq. 2], a linear combination of features a logistic regression already has. The ratio form (RASM) is not linear.
- **Differential entropy is the diagonal of what Riemannian features use.** 0.5 × log of the diagonal of the alpha-band covariance matrix equals alpha differential entropy to 1e-15 [measured]. Tangent-space features add the off-diagonal terms, the coupling between channels: 528 values per window for 32 channels, or 2,112 with four bands, against about 480 training windows.

### What comparisons say

**On SEED (three classes, 62 channels; train and test on different film clips of one session, so clip-safe).**

- Duan 2013: "DE and its combination on symmetrical electrodes perform better than ES feature." In the paper's Table II, differential entropy on all bands beats energy spectrum in ten of the eleven experiments legible in the extracted table and ties in one.
- Zheng and Lu 2015: "the DE features from the total frequency bands perform the best", and the asymmetry features "can achieve comparable accuracies" with fewer dimensions. Beta and gamma carried most of the signal. Bands and windows as in Duan 2013.
- These are the source of differential entropy's standing. They are from one lab, on one dataset family, with 1 s windows, and are not DEAP.

**On DEAP, trial-safe.** All accuracy, arousal / valence, labels thresholded at 5:

| Source and protocol | Method | Arousal | Valence |
|---|---|---|---|
| TSception paper, Table 5, leave-one-trial-out, features from the whole trial ([arXiv 2104.02935](https://arxiv.org/abs/2104.02935)) | SVM, as reported by the DEAP paper (log band power and asymmetry, below) | 62.00% | 57.60% |
| same | CSP (from Appriou, Cichocki and Lotte 2020) | 58.26% | 57.59% |
| same | Tangent-space classifier (Appriou et al.) | 60.04% | 59.47% |
| same | Filter-bank tangent-space classifier (Appriou et al.) | 60.60% | 61.09% |
| same | FgMDM / filter-bank FgMDM (Appriou et al.) | 60.04% / 60.30% | 58.87% / 61.01% |
| TSception paper, Table 3, trial-wise 10-fold, 4 s windows | SVM | 60.37% | 55.19% |
| same | EEGNet | 58.29% | 54.56% |
| LibEER, subject-dependent, trial-level 0.6 : 0.2 : 0.2 split ([arXiv 2410.09767](https://arxiv.org/html/2410.09767)) | SVM | 63.49% | 54.17% |

- The DEAP paper's own EEG features are "the logarithms of the spectral power" in five bands for 32 electrodes plus "the spectral power asymmetry between 14 pairs of electrodes", 216 features, with Welch's method and windows of 256 samples [paper: Koelstra et al. 2012, [PDF](https://access.archive-ouverte.unige.ch/access/metadata/1d5ce2a8-00d8-4793-8943-cf9e8aa56693/download)].
- Khan et al. use Welch band power with k-nearest neighbours on DEAP and show the leak between window-level and trial-level splits; they do not compare features ([arXiv 2508.02417](https://arxiv.org/abs/2508.02417)).
- **Reading:** on DEAP with trial-safe splits, log band power, differential entropy and Riemannian features are all within about three points of each other, and all sit near the per-participant majority baselines recorded in ticket 02 (0.58 valence, 0.64 arousal). No trial-safe source found shows one of them clearly ahead. Our logistic-regression baseline (0.60 valence, 0.55 arousal balanced accuracy) is in the same range.
- **No trial-safe DEAP comparison of differential entropy against Welch band power, multitaper band power or frontal asymmetry as separate feature sets was found.** Given the correlations above, a large difference would be surprising.

**On frontal asymmetry specifically.** No source read here tests frontal alpha asymmetry alone as a DEAP classifier feature. The two places asymmetry appears (the DEAP paper's 14 pairs, SEED's 27 pairs) use it across the whole head and alongside per-channel power.

## 3. Bands

Edges in Hz. "None" means the source does not use that band.

| Source | delta | theta | alpha | beta | gamma |
|---|---|---|---|---|---|
| **Ticket's proposal** | none | 4–8 | 8–13 | 13–30 | 30–45 |
| Current code, `FREQ_BANDS` [installed] | 1–4 | 4–8 | 8–14 | 14–30 | 30–40 |
| DEAP paper, classification features [paper] | none | 4–8 | 8–12 (plus slow alpha 8–10) | 12–30 | 30+ |
| DEAP paper, correlation analysis [paper] | none | 3–7 | 8–13 | 14–29 | 30–47 |
| Duan 2013; Zheng and Lu 2015 (SEED) [paper] | 1–3 | 4–7 | 8–13 | 14–30 | 31–50 |
| Khan et al. on DEAP [paper] | none | 4–8 | 8–14 | 14–30 | 30–42 |
| torcheeg `BandDifferentialEntropy` default [doc, source] | none | 4–8 | 8–14 | 14–31 | 31–49 |
| LibEER README example [doc] | 0.5–4 | 4–8 | 8–14 | 14–30 | 30–50 |
| mne-features `pow_freq_bands` default [tested] | 0.5–4 | 4–8 | 8–13 | 13–30 | 30–100 |
| MNE `plot_topomap` default bands [installed] | 0–4 | 4–8 | 8–12 | 12–30 | 30–45 |
| NeuroKit2 `eeg_power` [tested] | 1–4 | 4–8 | 8–13 | 13–30 | 30–80 |
| YASA `bandpower` default [tested] | 0.5–4 | 4–8 | 8–12 (then sigma 12–16) | 16–30 | 30–40 |

- **Theta 4–8 and beta ending at 30 are shared by nearly every source.**
- **The alpha/beta edge is 12, 13 or 14.** 13 is what mne-features and NeuroKit2 use; 14 is what torcheeg, LibEER and Khan et al. use on DEAP; 12 is the DEAP paper and MNE's plotting default. None is more correct.
- **Gamma's top is set by the data.** DEAP's files are band-passed 4–45 Hz, so 45 is the natural limit. torcheeg's 49 and LibEER's 50 reach into the filter's stop band.
- **Dropping delta on DEAP is standard** for the same reason: torcheeg's default, Khan et al. and the DEAP paper's features all start at theta. The earlier research note measured essentially no power below 4 Hz in `s01`.
- **So the proposal (4–8, 8–13, 13–30, 30–45) can be called standard.** It is not the single canonical set, because there is none. The current code's bands differ from it (a delta band, alpha to 14, gamma to 40).

## 4. Recommendation

**Keep scipy. Keep differential entropy as Butterworth band-pass then 0.5 × log-variance, with the four proposed bands.**

- It is the feature the DEAP literature names and the computation torcheeg's default uses, so the code can be described in the field's own words.
- It needs no new dependency and about 12 lines.
- No library that installs here ships it, and the alternatives that do install compute something that correlates 0.93 to 0.99 with it.
- Nothing trial-safe on DEAP shows another feature doing better.

**Main alternative: pyRiemann covariance and tangent space**, if the rebuild later wants to test a second feature family.

- It is the one candidate with real standing (MOABB) and a healthy release record, it installs cleanly, and it adds three small packages.
- It uses information differential entropy does not (coupling between channels).
- Against it: the evidence is from motor imagery and evoked potentials, the one trial-safe DEAP result is within about three points of band power, and 528 or more features per window against about 480 training windows needs strong regularization.

**Not recommended:**

- **mne-features**: saves under 10 lines, changes the feature, sparse release history.
- **Welch through scipy** (`signal.welch` then a band sum): a fair choice with the same dependencies and four times faster, but it is called band power, not differential entropy, and it gains nothing measurable.
- **torcheeg**: does not install.
- **YASA, antropy, NeuroKit2**: built for other jobs.

## 5. Probe scripts

Throwaway scripts in a scratch directory, not committed. In outline: read each package's PyPI JSON for version, date and requirements; `uv venv --python 3.14`, install the repo's pinned stack, then `uv pip install --dry-run` and a real install of each library and an import; a second virtualenv for torcheeg with the pins; unpickle `s01.dat`, cut 600 windows by reshape, compute each feature route, time it, and correlate it with Butterworth differential entropy; read default band edges from each library's source or signature.

## 6. Not verified

- **No accuracy was measured here.** No model was trained, so the claim that the features are interchangeable on DEAP rests on their correlation on one participant and on published tables, not on a head-to-head run with our split.
- **Appriou, Cichocki and Lotte 2020** (the Riemannian DEAP numbers) was not read; its PDF would not download. The numbers are quoted from TSception's Table 5, and its band edges, window length and software are unknown.
- **Which features TSception's own SVM used** in its Table 3 was not found in the text. LibEER's paper does not state its SVM's features or its DEAP label threshold; its README example uses 5.
- **Zheng and Lu 2015's Table III** is an image in the PDF; only the surrounding text was read. Shi, Jiao and Lu 2013 (the proof that differential entropy equals log band energy) was not read; the statement is quoted from Duan 2013 and Zheng and Lu 2015.
- **"MNE is the standard EEG toolbox"** rests on dependency declarations and repository activity, not on a survey.
- **torcheeg's unreleased `main`** was not installed; whether it works with scipy 1.18 is unknown. Its PyPI metadata lists no requirements, so the scipy cap was learned from the resolver's error.
- **eegdash and LibEER** were not installed or run. The other libraries were imported and called once; none was run under the repo's `filterwarnings = ["error"]` test setting or checked with mypy.
- **Timings and correlations** are from `s01` only. The earlier note found a tenfold amplitude spread between participants; correlations for others may differ.
- **GitHub star counts and push dates** are from the GitHub API on 2026-11-22.
- **Frontal alpha asymmetry** as a classifier feature on its own was not researched beyond the papers above.
