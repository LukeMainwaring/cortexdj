# Loading, preprocessing and windowing DEAP with braindecode 2.0 and MNE 1.13

Snapshot of 2026-11-22. Question: what is the current way to load DEAP's `data_preprocessed_python` files, preprocess them, cut windows, and train a small model, given the versions this repo installs?

Every claim is marked with where it comes from:

- **[installed]**: read or run against the backend virtualenv (`braindecode 2.0.0`, `mne 1.13.2`, `torch 2.15.0`, `scipy 1.18.1`, `skorch 1.4.0`, `scikit-learn 1.9.1`, Python 3.14.8).
- **[measured]**: computed from the 32 `.dat` files in `backend/data/deap` (the Kaggle mirror named in `backend/data/DEAP_SETUP.md`).
- **[doc]**: a linked first-party page.

Timings were taken on the owner's Mac and are indicative only. Section 6 lists what was not verified.

## 1. Is there a maintained DEAP loader?

| Library | DEAP loader | Evidence |
|---|---|---|
| braindecode 2.0.0 | No | [installed] `braindecode.datasets.__all__` holds `MOABBDataset`, `HGD`, `BNCI2014_001`, `TUH*`, `SIENA`, `NMT`, `CHBMIT`, `SleepPhysionet*`, `BCICompetitionIVDataset4`, `BIDSDataset`, plus the `create_from_*` builders. Nothing for DEAP or any emotion dataset. The 2.0.0 changelog (released 2026-11-02) adds none: [whats_new](https://braindecode.org/stable/whats_new.html). |
| MOABB 1.7.2 | No | [doc] The [dataset summary](https://moabb.neurotechx.com/docs/dataset_summary.html) lists imagery, P300/ERP, SSVEP, c-VEP, resting state, and a few cognitive and imagined-speech sets. No DEAP, SEED or DREAMER. MOABB is not installed here (it is the optional `braindecode[moabb]` extra). |
| MNE 1.13.2 | No | MNE has no DEAP fetcher; it supplies the containers (`RawArray`, `EpochsArray`). |
| torcheeg 1.1.3 | Yes, `torcheeg.datasets.DEAPDataset` | [doc] [DEAPDataset](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.datasets.DEAPDataset.html) reads the same `data_preprocessed_python` folder. Defaults: `chunk_size=128` (1 s windows), `overlap=0`, `num_channel=32`, `num_baseline=3`, `io_mode='lmdb'` (it writes a cache database). Each item is `(eeg, baseline, label)`. Latest PyPI release is 1.1.3 from 2024-12-23 ([PyPI](https://pypi.org/project/torcheeg/)); the [repository](https://github.com/torcheeg/torcheeg) had a commit on 2026-11-01. Not installed here, and not tested against Python 3.14 or torch 2.15. |

So with the installed stack, wrapping the arrays is the only route, and braindecode documents it as a supported one. The routes, all checked on a real participant file:

**a. Plain numpy, no dataset object.** [installed] `EEGClassifier.fit(X, y)` accepts a float32 array `(n_windows, 32, n_times)` and an int64 label array. Windowing a 60 s trial into non-overlapping 4 s windows is one `reshape`. A plain PyTorch loop takes the same array through `TensorDataset`.

**b. `braindecode.datasets.create_from_X_y`.** [doc] The one route the [Custom Dataset Example](https://braindecode.org/stable/auto_examples/datasets_io/plot_custom_dataset_example.html) shows for numpy data. [installed] Signature: `create_from_X_y(X, y, drop_last_window, sfreq, ch_names=None, window_size_samples=None, window_stride_samples=None)`. It wraps each trial in an `mne.io.RawArray`, then calls `create_fixed_length_windows`. Run on `s01.dat` with size and stride 512 it returned 600 windows; each item is `(x, y, [i_window_in_trial, i_start, i_stop])` with `x` float32 `(32, 512)`, values unchanged. Stride 256 gave 29 windows per trial. The metadata frame carries `i_dataset`, which identifies the source trial. Two things to know:

- It calls `mne.create_info(ch_names, sfreq)` without channel types, so every channel is typed `misc`. As a result `preprocess(ds, [Filter(...)])` and `preprocess(ds, [Preprocessor(exponential_moving_standardize, ...)])` both failed with `picks ... yielded no channels`. Preprocessing has to happen on the numpy array before the call, or with explicit `picks`.
- It takes one label per trial; a second label (valence and arousal together) does not fit its `y` without building the `RawDataset` descriptions by hand.

**c. `mne.EpochsArray`.** [installed] `mne.EpochsArray(data, info, tmin=-3.0)` with `info = mne.create_info(names, 128.0, "eeg")` holds a participant as 40 epochs running from −3 s to 60 s. [doc] The [Simple training on MNE epochs](https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html) tutorial passes an `EpochsArray` straight to `EEGClassifier.fit(epochs, y)`. [installed] That worked with `EEGClassifier("EEGNet", ...)`, and the wrapper inferred `n_chans=32`, `n_times=512`, `n_outputs=2`, `sfreq=128.0`. Things to know:

- MNE expects EEG in volts, and DEAP is in microvolts, so the array needs `* 1e-6` on the way in. What reaches the model is then about 3e-6 in magnitude unless it is scaled back; the [trialwise tutorial](https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html) multiplies by `1e6` for this reason.
- `apply_baseline((None, 0))` and `crop(tmin=0)` exist, but `crop(tmin=0, include_tmax=False)` returned 7679 samples, not 7680.
- `create_from_mne_epochs` re-windows epochs, but its targets come from the epochs' event codes (every window got `y=1` in the probe), so labels need an `events` array and a `mapping`.
- MNE's built-in `biosemi32` montage has the same 32 names as DEAP but in a different order. `set_montage` matches by name, so positions attach correctly; the montage's own order must not be used as DEAP's channel order.

**d. `create_from_mne_raw`.** Needs annotated continuous recordings. The preprocessed files are already cut into trials, so it does not apply.

The pickle read itself has no replacement in any of these routes: the file is a Python 2 pickle, and `pickle.load(f, encoding="latin1")` is what the [DEAP readme](https://web.archive.org/web/20250118184132/http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html) shows in its Python 2 form (`cPickle.load`). torcheeg's loader does the same read.

## 2. What braindecode and MNE helpers offer over the current hand-written code

What the current code does ([`ml/dataset.py`](../../backend/src/cortexdj/ml/dataset.py), [`ml/preprocessing.py`](../../backend/src/cortexdj/ml/preprocessing.py), [`ml/predict.py`](../../backend/src/cortexdj/ml/predict.py)):

- `load_deap_participant` unpickles, keeps the first 32 channels, and drops the first 384 samples. It applies no normalization.
- `preprocessing.py` computes differential entropy per band by a fifth-order Butterworth `filtfilt` over five bands and log-variance, and band powers by Welch. Its module docstring says "using MNE-Python and scipy", but it imports only scipy.
- `predict.py` has two input paths: a 160-value differential-entropy vector for `EEGNetClassifier`, and the raw window resampled to 200 Hz and scaled by 0.01 for CBraMod.
- `ml/model.py`'s `EEGNetClassifier` takes that 160-value vector as a 32×5 grid. It is not the raw-signal EEGNet of Lawhern et al. that `braindecode.models.EEGNet` implements.

What the libraries provide for each step:

| Step | Helper | Notes |
|---|---|---|
| Read the file | None | See section 1. |
| Pick channels, drop pre-trial | numpy slice, or `EpochsArray.crop` / `pick` | The slice is shorter. |
| Windowing | `create_from_X_y`, `create_fixed_length_windows` | [installed] Gives stride and overlap, a last-window policy (`drop_last_window`, or `on_last_window`), and per-window metadata (`i_dataset`, start, stop) that maps a window back to its trial. For non-overlapping windows of one array a `reshape` gives the same result. |
| Filtering | `braindecode.preprocessing.Filter`, `Preprocessor("filter", ...)`, `mne.filter.filter_data` | [installed] `Filter` subclasses `Preprocessor` and forwards to MNE's `filter`. MNE's default is a zero-phase FIR (`method='fir'`, `phase='zero'`, `fir_design='firwin'`), not the Butterworth IIR the repo uses. Needs channels typed `eeg` (see 1b). |
| Standardization | `exponential_moving_standardize(data, factor_new=0.001, init_block_size=None, eps=1e-4)` | [doc] Used in the trialwise tutorial with `factor_new=1e-3`, `init_block_size=1000`. It is a running standardization meant for continuous recordings. |
| Band power | `mne.time_frequency.psd_array_welch(x, sfreq, fmin, fmax, n_fft=256, ...)` | Same method as `scipy.signal.welch`; different defaults (`window='hamming'`, `n_overlap=0`). |
| Resampling | `Resample` preprocessor, `mne.filter.resample` | Only the CBraMod path resamples today. |
| Differential entropy | None | Neither braindecode nor MNE ships it. torcheeg has `BandDifferentialEntropy`. |
| Train/valid split | `ds.split(...)`, skorch `ValidSplit`, `predefined_split` | [doc] See the [Split Dataset Example](https://braindecode.org/stable/auto_examples/datasets_io/plot_split_dataset.html). |
| Training loop | `EEGClassifier` (skorch) | See section 4. |

Two facts bear on the hand-written feature code specifically:

- [doc + measured] The files are already band-passed 4–45 Hz. In `s01.dat` the mean power spectral density is 0.0000 in 0.5–2 Hz and 0.008 in 2–4 Hz, against 1.42 in 4–8 Hz. `FREQ_BANDS["delta"] = (1.0, 4.0)` therefore measures the filter's roll-off, not delta activity, in both the differential-entropy features and the `band_powers` returned to the app.
- [doc] The raw-signal models in section 4 take the window itself as input, so with one of them the differential-entropy path (five `filtfilt` passes per window) is not needed for classification. Band powers for display are a separate use.

## 3. What DEAP's preprocessing already did, and what is left

[doc] From the "Data_preprocessed_matlab.zip and Data_preprocessed_python.zip" section of the DEAP readme. The official site (`eecs.qmul.ac.uk`) returned 503 and a TLS error on 2026-11-22; this is the [Wayback capture of 2025-01-18](https://web.archive.org/web/20250118184132/http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html).

Each file holds `data` of shape 40 × 40 × 8064 (trial × channel × sample) and `labels` of shape 40 × 4 (valence, arousal, dominance, liking). Applied to the 32 EEG channels, in the readme's words:

1. "The data was downsampled to 128Hz."
2. "EOG artefacts were removed as in [1]." ([1] is the DEAP paper, Koelstra et al. 2012.)
3. "A bandpass frequency filter from 4.0-45.0Hz was applied."
4. "The data was averaged to the common reference."
5. "The EEG channels were reordered so that they all follow the Geneva order as above."
6. "The data was segmented into 60 second trials and a 3 second pre-trial baseline removed."
7. "The trials were reordered from presentation order to video (Experiment_id) order."

Channels 33–40 (hEOG, vEOG, zEMG, tEMG, GSR, respiration, plethysmograph, temperature) got steps 1, 6 and 7 only.

Channel order, 1–32: Fp1, AF3, F3, F7, FC5, FC1, C3, T7, CP5, CP1, P3, P7, PO3, O1, Oz, Pz, Fp2, AF4, Fz, F4, F8, FC6, FC2, Cz, C4, T8, CP6, CP2, P4, P8, PO4, O2.

Checks against the files [measured], six participants (s01–s04, s31, s32):

- **Common average reference holds.** The mean across the 32 EEG channels is zero at every sample (largest absolute value 9e-14). Re-referencing would change nothing.
- **Band-pass holds.** See the spectrum figures in section 2. No power at 50 Hz, so no notch filter is needed.
- **"Baseline removed" means mean-subtracted, and the 3 s are still in the file.** 8064 samples is 63 s. The mean of the first 384 samples is zero for every trial and channel (largest absolute value 1e-14 for EEG, 3e-9 for the peripheral channels), so each trial has already had its pre-trial mean subtracted. Subtracting the pre-trial mean again is a no-op. The 60 s stimulus segment has a small non-zero mean per channel (0.04 to 0.64 µV on average).
- **Amplitude differs a lot between participants.** EEG standard deviation: 4.4 (s01), 42.2 (s02), 15.2 (s03), 59.9 (s04), 12.3 (s31), 21.5 (s32). Units are not stated in the readme section; the magnitudes are consistent with microvolts.

What is left for a pipeline:

- Select the 32 EEG channels and drop the first 384 samples.
- Normalize. The files are not standardized, and the tenfold spread between participants above is the fact that makes some scaling necessary for a model trained across participants. Options seen in first-party sources: per-channel z-scoring, and braindecode's exponential moving standardization (section 2). Which statistics to use (per window, per trial, per participant, or training-set) is a choice, not something the sources settle; statistics computed per participant or per trial need no training-set state at inference.
- Window. The current code uses 4 s (512 samples), giving 15 non-overlapping windows per trial and 600 per participant. torcheeg's default is 1 s. Windows from one trial share a label and are strongly correlated, so a split that puts windows of the same trial on both sides leaks; `i_dataset` (section 1b) or the trial index is the grouping key.
- Binarize labels. `DEAP_SETUP.md` records that the shipped checkpoints used each participant's median; torcheeg's examples use `Binary(5.0)`.
- A spectral baseline correction using the pre-trial segment (subtracting pre-trial band features from trial band features) appears in the DEAP literature and is why torcheeg returns a `baseline` item. It applies to band-feature inputs, not to raw-window models. Not traced to a primary source here.

**Label anomaly in the local files.** [doc] The readme documents each rating as a "float between 1 and 9". [measured] Across the 32 local files, 110 of 1280 valence values and 53 of 1280 arousal values are below 1 (minimum 0.0); dominance and liking stay within 1–9. Mean valence is 3.47 and mean arousal 3.73, and 22% of valence and 25% of arousal ratings are ≥ 5, which matches the "~24/76" split `DEAP_SETUP.md` reports. The first row of `s01.dat` is `[1.29, 1.40, 6.90, 7.83]`. These files come from a Kaggle mirror, not the official download, and no official copy was available to compare. Whether the mirror altered the valence and arousal columns is unverified. If the two columns were reversed (stored as 9 − x), a median split would still give balanced classes but with high and low swapped.

## 4. Small-model defaults and their input requirements

[installed] `braindecode.models` exports 94 names in 2.0.0. `EEGNet` is the Lawhern et al. 2018 model; `EEGNetv4` and `EEGNetv1` no longer exist as names. The 2.0.0 changelog notes that `EEGNet`'s initial weights differ from 1.x for the same seed, and that whole-model pickles from 1.8.1 do not load for 13 models, so `state_dict` is the save format to use ([whats_new](https://braindecode.org/stable/whats_new.html)).

**Input contract** [installed], shared by every model through `EEGModuleMixin`: a float tensor `(batch, n_chans, n_times)`. The constructor needs `n_chans`, `n_outputs`, and either `n_times` or both `sfreq` and `input_window_seconds`; omitting them raises `ValueError: n_times could not be inferred`. `EEGNet(n_chans=32, n_outputs=2, sfreq=128, input_window_seconds=4)` worked. The window length is fixed at construction: an `EEGNet` built for 512 samples raised on a 256-sample input. Output is `(batch, n_outputs)` raw logits, which is why `EEGClassifier` defaults to `CrossEntropyLoss`. The models are plain `torch.nn.Module`s.

Built with `n_chans=32, n_outputs=2, n_times=512` [installed]:

| Model | Parameters | Extra requirement | CPU s/epoch, 600 windows | MPS s/epoch, 600 windows |
|---|---|---|---|---|
| `EEGNet` | 2,130 | none | 1.86 | 1.10 |
| `ShallowFBCSPNet` | 54,562 | none | 0.46 | 1.85 |
| `EEGNeX` | 56,354 | none | not timed | not timed |
| `TSception` | 209,111 | `sfreq` | not timed | not timed |
| `Deep4Net` | 283,927 | none | not timed | not timed |

Timings: `EEGClassifier`, AdamW, batch 64, five epochs on one participant's 600 standardized windows. Scaled linearly to all 32 participants (19,200 windows) that is roughly 60 s per epoch for `EEGNet` on CPU and 15 s for `ShallowFBCSPNet` on CPU; the scaling is an extrapolation, not a measurement. No accuracy was measured.

Which model the docs treat as the default [doc]: `ShallowFBCSPNet` is the model in the [trialwise tutorial](https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html), the [MNE epochs tutorial](https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html) and the [pure PyTorch tutorial](https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html); `EEGNet` is the model in the [MOABB benchmark example](https://braindecode.org/stable/auto_examples/advanced_training/plot_moabb_benchmark.html). Those tutorials are all motor imagery; braindecode has no emotion-recognition example. `TSception` was published for emotion recognition, a statement not checked against its paper here.

**Training helpers.**

- `braindecode.EEGClassifier` [installed]: signature `(module, *args, criterion=CrossEntropyLoss, cropped=False, callbacks=None, iterator_train__shuffle=True, iterator_train__drop_last=True, aggregate_predictions=True, **kwargs)`. [doc] It "inherits from skorch.classifier.NeuralNetClassifier, so the training logic is the same as in skorch". skorch's own defaults still apply: SGD, `lr=0.01`, `max_epochs=10`, `batch_size=128`. `module` can be a class, an instance, or a name string such as `"EEGNet"`; model arguments go in as `module__<name>`. `fit` accepts numpy arrays, an `mne.Epochs`, or a braindecode dataset with `y=None`, and gives `predict`, `predict_proba`, a `history`, callbacks (the tutorial uses `"accuracy"`, a `CosineAnnealingLR` scheduler and `EarlyStopping`), and scikit-learn compatibility for `cross_val_score` and `GridSearchCV` ([tuning example](https://braindecode.org/stable/auto_examples/model_building/plot_hyperparameter_tuning_with_scikit-learn.html)).
- The tutorial's values for the shallow network: AdamW, `lr = 0.0625 * 0.01`, `weight_decay = 0`, `batch_size = 64`. The page says they were found for motor decoding and asks readers to tune their own.
- One default to know [doc]: `iterator_train__drop_last=True`. In the MNE epochs tutorial's own output, 80 training examples with the default `batch_size=128` meant every batch was dropped and nothing trained.
- Plain PyTorch [doc]: the pure PyTorch tutorial shows a `DataLoader` over the windows dataset yielding `(X, y, _)`, `CrossEntropyLoss`, AdamW and `CosineAnnealingLR`, about 30 lines for train and eval functions. It works because the models are ordinary modules.
- `braindecode.training` [installed] holds only cropped-decoding losses, scoring callbacks and trial-prediction helpers. There is no trainer other than the skorch wrappers.

**Two outputs.** The current classifier returns arousal and valence logits from two heads. Every braindecode model has one output layer of `n_outputs` units, and `EEGClassifier` trains one target. With these models as shipped, the options are a single four-class quadrant target (`n_outputs=4`), two separate models, or a plain loop that splits a wider output; a two-headed variant means custom code around the backbone.

## 5. Probe scripts

The probes were throwaway scripts run with `uv run --directory backend python`. In outline: unpickle a participant; assert the facts in section 3 with numpy and `scipy.signal.welch`; build datasets with `create_from_X_y`, `mne.EpochsArray` and `create_from_mne_epochs`; instantiate the five models and count parameters; time `EEGClassifier.fit` on 600 windows for CPU and MPS.

## 6. Not verified

- The DEAP paper (Koelstra et al. 2012) was not re-read; "EOG artefacts were removed as in [1]" is quoted from the readme without the method's details. The readme was read from a Wayback capture, not the live site.
- Whether the Kaggle mirror's labels match the official files (section 3).
- torcheeg was not installed or run; its compatibility with Python 3.14 and torch 2.15 is unknown.
- The braindecode changelog was read for 2.0.0 only; the release in which `EEGNetv4` was renamed was not located.
- No model was trained to convergence. No accuracy figure for any model on DEAP comes from this note, and the per-epoch times for 32 participants are extrapolated.
- The claim that spectral baseline correction and 1 s windows are common in DEAP papers rests on torcheeg's defaults, not on a literature survey.
