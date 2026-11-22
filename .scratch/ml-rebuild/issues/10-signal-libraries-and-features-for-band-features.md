# 10: Which libraries and which features for band features on DEAP

**Type:** research
**Status:** resolved

## Question

For a within-subject emotion classifier on DEAP's preprocessed arrays (32 channels, 128Hz, 4-second windows with no overlap, about 480 training windows per participant), what is the current, well-regarded way to compute the features, and with which library?

- **Libraries.** Which maintained Python libraries do EEG practitioners use today for spectral and band features on arrays that are already epoched (MNE's `time_frequency` functions, `mne-features`, braindecode, torcheeg, pyRiemann, NeuroKit2, antropy, YASA, plain scipy, others)? For each: what it offers for this job, its latest release and date, whether it installs on this repo's Python and alongside torch 2.15, braindecode 2.0 and scipy 1.18, and what it costs in dependencies. Is there an industry standard, and does adopting it shorten or lengthen the roughly 40 lines of scipy we have?
- **Features.** Is differential entropy the best-supported feature for this task, against band power from Welch or multitaper, frontal asymmetry, and Riemannian covariance features? What do trial-safe comparisons on DEAP or similar sets say, and how is differential entropy computed in the libraries that ship it (band-pass then log-variance, or from a spectrum)?
- **Bands.** Which band edges do DEAP papers and the libraries use by default, so ours (theta 4 to 8, alpha 8 to 13, beta 13 to 30, gamma 30 to 45Hz, no delta) can be called standard or not?

Ticket 01 already found that no library ships a maintained DEAP loader and that braindecode and MNE have no differential entropy; do not repeat it, build on it.

## Answer

Resolved 2026-11-22. Full findings: `docs/research/2026-11-22-signal-libraries-and-features.md` on branch `refs/heads/research/ml-rebuild-signal-libraries` (commit 0a98258, pushed, never merged). Installs were tested in a throwaway venv on Python 3.14.8 beside torch 2.15.0, braindecode 2.0.0, scipy 1.18.1 and numpy 2.5.5, each library called once on `s01.dat`.

- **No industry standard for this step.** MNE is the standard EEG toolbox and is already installed through braindecode, but it stops at the spectrum (`psd_array_welch`, `psd_array_multitaper`): no band-feature function, no differential entropy. braindecode has no feature extraction.
- **The libraries that come close.** `mne-features` 0.3.2 installs (2 more packages) and gives Welch log band power, not differential entropy; it had no release from September 2023 to November 2025. pyRiemann 0.12 installs (3 small packages) and is well regarded, but its evidence is from motor imagery, P300 and SSVEP, and it is a different feature. torcheeg is the only one that ships differential entropy and does not install: 1.1.3 needs scipy <= 1.10.1, and 1.1.2 fails on import. YASA, antropy, NeuroKit2 and eegdash install or resolve but are built for other jobs.
- **Adopting one barely shortens the code.** The differential-entropy computation is about 6 lines of scipy. `mne-features` would save under 10 lines, change the feature, and add two packages. Every route runs in under a second per participant.
- **Differential entropy is log band power.** It is 0.5 x log(band variance). On `s01` it correlates 0.99+ with multitaper power, 0.95 to 0.99 with Welch. Left-minus-right asymmetry is a linear combination of it, so a linear model already has it. It is exactly the diagonal of the band covariance matrix that Riemannian features use; tangent space adds cross-channel terms, 528 features a window against about 480 training windows.
- **Nothing measurably beats it on DEAP.** Trial-safe published results (log band power SVM, tangent space, filter-bank tangent space) sit within about three points of each other and of the majority baseline, the same range as our 0.60 / 0.55. The evidence that differential entropy beats raw band power is from SEED, not DEAP. No trial-safe DEAP comparison of the feature sets head to head was found.
- **Our computation matches the DEAP convention.** Band-pass then log-variance is torcheeg's default and what it calls the accepted method for DEAP; the SEED papers compute it from a spectrum instead.
- **The band edges are standard, though no single set is canonical.** Theta 4 to 8 and beta ending at 30 are near-universal; the alpha/beta edge is 12, 13 or 14 depending on the source; 45 is the natural gamma top because DEAP is filtered there; dropping delta matches torcheeg's default and the DEAP paper.
- **Two side findings.** Filtering each 4-second window alone or the whole trial before windowing gives features correlated 0.995 to 0.9998. scipy's docs recommend the second-order-sections filter form over the `(b, a)` form in `ml/preprocessing.py`; the two agree to 2e-8 here.
- **Recommendation:** keep plain scipy and differential entropy. Main alternative, if a second feature family is ever wanted: pyRiemann covariance and tangent space.
- **Not verified:** no model was trained, so "interchangeable on DEAP" rests on one participant's correlations and published tables; the Riemannian DEAP numbers are quoted second-hand from the TSception paper; correlations and timings are from `s01` only; no library was checked under mypy strict; torcheeg's unreleased `main` was not installed.

## Comments
