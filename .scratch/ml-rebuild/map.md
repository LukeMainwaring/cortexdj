# Map: Rebuild the signal pipeline and the quadrant classifier

Label: wayfinder:map

## Destination

Settled decisions for one spec: the DEAP signal pipeline and the quadrant classifier rebuilt together, with training code back in the repo. One classifier pipeline the user can explain end to end, reported with an honest number (balanced accuracy against the majority baseline, on a stated split), even if that number stays near chance. The map is done when nothing is left to decide before `to-spec`.

## Notes

- Domain: EEG emotion decoding on DEAP. Read `GLOSSARY.md` (Segment, Quadrant, Session) and `backend/src/cortexdj/ml/` before a ticket. Grilling tickets use the `grilling` and `domain-modeling` skills; research tickets use `research`.
- The app is a decoder calibrated to one person, and DEAP's 32 participants stand in for that person: one recipe, one model per participant, judged within-subject by trial (`docs/adr/0005-one-quadrant-classifier-per-participant.md`). A Muse headband and the user's own rated sessions are where this heads, not part of this map.
- Simplicity outranks accuracy. Prefer the approach that is easier to read and explain over one that gains a point or two.
- One code path. Training is one plain Python function per participant that runs on any machine; a full run of all 32 is a local loop of about a minute. Modal is not in this effort: the function is shaped so a thin Modal wrapper can fan participants out later, with the bake-off. No multiplying of device, config, or backend branches: few knobs, one model, one preprocessing path.
- Exactly one classifier pipeline survives; the other is deleted along with the `isinstance` branch in `ml/predict.py`.
- The app contract is fixed by default: 4-second stored segments, arousal and valence scores from 0 to 1, a quadrant, band powers; no migrations, no frontend changes. Reopen it only if a ticket finds the segment timing or the scores are plainly wrong. The model's input window may differ from the stored segment.
- Input stays DEAP's preprocessed `.dat` files (128Hz, already filtered, eye artifacts removed).
- The contrastive encoder is untouched: `contrastive_best.pt` must keep working, so `trial_to_eeg_windows` and `CBRAMOD_SCALE_FACTOR` as retrieval uses them do not change.
- Starting facts: the libraries are current (torch 2.15, braindecode 2.0, mne 1.13, scipy 1.18); the old training code exists only in git history (`legacy/backend/src/cortexdj/ml/train.py`, `legacy/backend/scripts/modal_train.py`, `legacy/backend/autoresearch/`, removed in 2378eff); the CBraMod checkpoint records about 52% leave-one-subject-out.
- `after-port 01` (every seeded segment is classified calm) is folded into this effort: the new checkpoint fixes it, it is not patched on its own. Other open tickets may go stale the same way; say so when a decision makes one obsolete.

## Decisions so far

- [Modal's current setup for small fan-out training runs](issues/03-modal-setup-for-small-fan-out-runs.md): a 30-line wrapper around a plain `train_fold`, CPU by default, folds via `.map`, data on a read-only Volume; about $0.67 per 32-fold run at an assumed 5 minutes a fold; untested on Modal.
- [Current braindecode and MNE practice for DEAP](issues/01-current-braindecode-mne-practice-for-deap.md): no loader to adopt, so we keep wrapping the arrays; DEAP already filtered, re-referenced and baseline-corrected, leaving normalization, windowing and labels to us; the delta band is empty; small braindecode models are single-output; the local labels look altered.
- [What recent benchmarks report for DEAP arousal and valence](issues/02-what-deap-benchmarks-report.md): trial-safe results sit near chance for every model family, foundation models included (about 0.50 to 0.57 balanced accuracy); numbers above 90% come from leaking window-level splits; and the local labels differ from the published ratings in 439 trials.
- [Are the local DEAP labels the official ones?](issues/09-are-the-local-deap-labels-official.md): no, the Kaggle mirror re-saved the files with valence and arousal changed in 439 trials; the effort keeps the local EEG (unverified against an official copy, accepted) and takes labels from `participant_ratings.csv` in `backend/data/deap/`, joined on participant and `Experiment_id`; the official site is down.
- [A baseline run on band features](issues/04-baseline-run.md): logistic regression on differential entropy scores chance cross-subject (0.50 balanced accuracy, about 0.53 with per-participant z-scoring) and 0.60 valence, 0.55 arousal within-subject by trial; the label rule barely moves balanced accuracy; the whole run takes 40 seconds on a laptop.
- [Labels, split and metric](issues/05-labels-split-and-metric.md): one model per participant, scored within-subject by trial (stratified 5-fold); high is a rating `>= 5`; the headline is per-window balanced accuracy for each axis, their mean the single number (baseline 0.57); the app shows only held-out predictions; cross-subject is quoted from a kept baseline script.
- [Which libraries and which features for band features on DEAP](issues/10-signal-libraries-and-features-for-band-features.md): no industry-standard library for band features; MNE stops at the spectrum, torcheeg (the only one with differential entropy) does not install beside scipy 1.18; differential entropy is log band power and nothing trial-safe on DEAP beats it; the four band edges are standard; keep plain scipy.
- [What preprocessing we own](issues/06-what-preprocessing-we-own.md): the model reads differential entropy in four bands (theta, alpha, beta, gamma; no delta) from 4-second windows with no overlap, the same slice as a stored segment; one standardization fitted on the training fold; the pre-trial is dropped; plain numpy and scipy, no MNE.
- [Which model survives](issues/07-which-model-survives.md): a linear layer in PyTorch (`nn.Linear(128, 2)`, arousal and valence as its two outputs) on the band features, fixed untuned settings; both current classifiers and `EEG_MODEL_BACKEND` are deleted; scikit-learn is not a dependency; no model was run, so 0.57 stays the baseline's figure until the first training run.
- [Where training code lives, the checkpoint, and how Modal fits](issues/08-training-code-home-and-modal.md): `cortexdj/training/` with one `train-classifier` command and a `train_participant` function shaped for Modal, which waits for the bake-off; one locally made checkpoint of all 160 fold models that carries its training recipe and fails at load on a mismatch; seeding scores each segment with the fold model that did not see its trial; tested on a made-up participant.
- [Band powers the app shows](issues/11-band-powers-the-app-shows.md): delta is dropped for DEAP; the displayed band powers are the model's own four bands (the variance the model takes the log of), one `FREQ_BANDS`, Welch deleted; the chart draws only the bands present in the data; `seed-sessions` replaces a participant's session, not adds a second.
- [What a re-seeded session shows at about 0.57](issues/12-what-a-re-seeded-session-shows.md): scores are the plain sigmoid with one fixed, strong weight decay keeping them off the ends; the trajectory, dominant quadrant and card label are computed per track, not per segment, backend only; the assistant hedges in one clause by a tier of the checkpoint's number ("a rough read" below 0.60); the canned brain-wave explanations are rewritten.

## Not yet specified

- What goes stale: `after-port 01`, `docs/ROADMAP.md`, the glossary's Segment and Quadrant entries, the architecture guide pages, the checkpoint download script, `backend/data/DEAP_SETUP.md` (its "same preprocessed data" and "~24/76" claims, and the missing ratings-file step), the recorded 52% figure, `AGENTS.md` (the `ml/`, `eeg_model.py` and `scripts/` paragraphs, `EEG_MODEL_BACKEND`), the classifier files on the GitHub Release, the system prompt's band list and quadrant list in `agents/assistant.py`, the four explanations in `features/sessions/emotion.py`, the Session Narrative instructions in the sessions capability (per-segment transitions and timestamps).

## Out of scope

- Retraining the contrastive EEG↔CLAP encoder: blocked on `audio-decoding 01` and a long run; a fresh effort later.
- Bigger models, raw-window architectures (a real EEGNet) and fine-tuning CBraMod as the classifier: a later effort once the app works end to end, starting with a bake-off on Modal, where the Modal wrapper is added and becomes the default training path (`docs/ROADMAP.md`, Phase 4; [Which model survives](issues/07-which-model-survives.md)).
- An autoresearch loop that changes the training recipe and reruns: tuning on the folds the headline number is reported on would make it dishonest, and there is little to win; it goes with the bake-off, through `train_participant`'s `recipe` argument ([Is an autoresearch loop in this effort?](issues/13-is-autoresearch-in-this-effort.md)).
- Going back to DEAP's raw recordings.
- Other datasets than DEAP, the user's own Muse recordings included.
