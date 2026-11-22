# DEAP binary arousal/valence benchmarks, split by evaluation protocol

Researched 2026-11-22. Every figure below was read that day from the paper's own text or tables (arXiv HTML, PMC full text, or the PDF), not from memory or a secondary summary. Companion to [datasets-analysis.md](datasets-analysis.md) and [pretrained-models-analysis.md](pretrained-models-analysis.md), which already cover the dataset catalog, the model catalog and NeuralBench's overall ranking; none of that is repeated here.

**Confidence note.** Table cells were extracted from arXiv HTML by script. Where a table's header had to be aligned to its columns by position (LibEER Tables V and VI), that is marked. Numbers that exist only in figures (Moreno-Alcayde, NeuralBench per-task) are given as the authors' prose states them. "What could not be verified" is at the end.

## Summary

1. **Under any split that keeps a trial's windows together, DEAP binary accuracy sits at or within a few points of the majority-class baseline, for every model family.** Cross-subject: 46-58% accuracy, 0.48-0.53 balanced accuracy. Within-subject with trial-level splits: 51-66% accuracy against a 58-64% majority baseline.
2. **Model family does not change that picture.** Band-power or DE features with an SVM, EEGNet-class conv nets, and ten EEG foundation models (CBraMod, LaBraM, REVE included) all land in the same band. In the one benchmark that runs foundation models on DEAP (OmniEEG-Bench), the best cross-subject balanced accuracy is 0.515 for arousal and 0.533 for valence, and the authors name DEAP-arousal among the "especially difficult" tasks.
3. **Window-level splits produce 86-99% on the same data.** Khan et al. show the same kNN going from about 52% (leave-one-trial-out) to over 90% with 6 segments per trial and nearly 100% with 60, and reproduce the effect on EEG recorded from a watermelon with random labels. 58 of 101 reviewed DEAP papers had this flaw.
4. **The label rule moves the majority baseline by 5 to 16 points**, so an accuracy figure without its rule and its baseline is not interpretable. Fixed threshold at 5: 55-59% pooled, 58-64% per participant. Threshold at 4.5: 63-67%. Per-participant median: 50-51%.
5. **The repo's local DEAP files do not carry the published valence and arousal ratings.** In `backend/data/deap/s*.dat`, exactly the 439 trials where both ratings exceed 5 have valence and arousal replaced by `9 - rating`; dominance and liking are untouched. Labels built from those files agree with labels built from the published ratings on 43-66% of trials. The recorded "about 52% under leave-one-subject-out" was measured against those labels. See [Local label check](#local-label-check).
6. **An honest expectation for a small, simple model** is therefore: cross-subject, indistinguishable from chance to about 0.55 balanced accuracy; within-subject with trial-level folds, 0.52-0.57 balanced accuracy, with per-participant credible intervals that mostly include 0.5.

## Majority-class baselines by label rule

Computed for this note from DEAP's `participant_ratings.csv` (32 participants x 40 trials). The file came from a public GitHub mirror, not the login-gated official download; two mirrors were byte-identical (sha256 `7bc0c19d...8ea1b8`), and the file reproduces the published statistics in the last two rows, which is the evidence that it is the authentic one.

| Label rule (high class) | Valence: high share | Valence: pooled majority | Valence: per-participant majority, mean | Arousal: high share | Arousal: pooled majority | Arousal: per-participant majority, mean |
|---|---|---|---|---|---|---|
| rating > 5 | 55.3% | 55.3% | 58.1% | 57.6% | 57.6% | 64.3% |
| rating >= 5 | 56.6% | 56.6% | 58.6% | 58.9% | 58.9% | 64.4% |
| rating > 4.5 | 63.1% | 63.1% | 64.2% | 63.8% | 63.8% | 66.9% |
| > participant's own median | 49.5% | 50.6% | 50.6% | 48.8% | 51.2% | 51.2% |
| >= participant's own median | 50.9% | 50.9% | 50.9% | 51.2% | 51.2% | 51.2% |

- "Pooled majority" is what a constant classifier scores cross-subject (the training pool's majority class applied to a held-out participant). Its spread across participants is wide: the high share per participant has a standard deviation of 9 points for valence and 16 for arousal, and for arousal ranges from 20% to 85%. A leave-one-subject-out accuracy of 52% +/- 12 is not distinguishable from this.
- "Per-participant majority" is the within-subject baseline, and is an upper bound because it uses all 40 of the participant's labels.
- 1.25% of valence ratings and 1.33% of arousal ratings are exactly 5, so `>` versus `>=` moves the baseline by about 1.3 points. The repo's `ml/dataset.py` uses `>= 5.0`.
- Balanced accuracy and macro-F1 have a chance level of 0.5 under every rule.
- Published checks: Koelstra et al. report the high class as "arousal 59%(15%), valence 57%(9%)" (mean and SD over participants) with the threshold "placed in the middle", and a majority-class accuracy of 0.644 for arousal and 0.586 for valence (Table 7). EEGain reports 56.5%/43.5% high/low valence and 58.9%/41.1% arousal at threshold 5. Both match the `>= 5` and per-participant rows above.

How the literature binarizes: EEGain's review counts 7 different threshold definitions for DEAP, with more than 30 studies using 5 and more than 10 using 4, and notes studies disagree on which side a rating of exactly 5 falls. EEGain's own default is low = rating <= 4.5. Koelstra, TSception, Khan et al. and Li et al. use 5. None of the benchmarks read for this note uses a per-participant median.

## Cross-subject (held-out participants)

| Source | Split | Model | Arousal | Valence | Metric | Label rule | Baseline reported |
|---|---|---|---|---|---|---|---|
| EEGain, Table IV | Leave-one-subject-out; 80/20 subject-level train/val inside each fold; 4 s windows, no overlap | EEGNet | 0.53 +/- 0.13 (F1 0.53, wF1 0.50) | 0.53 +/- 0.10 (F1 0.56, wF1 0.45) | Accuracy; F1 of the positive class; weighted F1 | low = rating <= 4.5 | Trivial: 0.59 +/- 0.15 arousal, 0.57 +/- 0.09 valence (wF1 0.53 / 0.51) |
| same | same | TSception | 0.54 +/- 0.09 (wF1 0.53) | 0.51 +/- 0.08 (wF1 0.47) | same | same | same |
| same | same | DeepConvNet | 0.57 +/- 0.14 (wF1 0.49) | 0.51 +/- 0.11 (wF1 0.41) | same | same | same |
| same | same | ShallowConvNet | 0.56 +/- 0.13 (wF1 0.49) | 0.53 +/- 0.08 (wF1 0.45) | same | same | same |
| LibEER, Table VI | Subjects split 0.6/0.2/0.2 train/val/test (not LOSO); 1 s windows; DE features in 5 bands | SVM on DE | 51.48 (7.76), F1 50.95 | 49.58 (6.77), F1 48.49 | Accuracy, F1 (averaging not stated) | Not stated | None |
| same | same | EEGNet | 48.94 (7.72), F1 48.94 | 52.36 (6.81), F1 49.74 | same | Not stated | None |
| same | same | TSception | 45.90 (8.16), F1 45.56 | 54.44 (6.33), F1 48.94 | same | Not stated | None |
| same | same | DGCNN | 49.91 (9.27), F1 47.09 | 49.91 (7.48), F1 47.09 | same | Not stated | None |
| same | same | Best of 18 methods | 51.48 (the SVM) | 57.78 (CDCN) | same | Not stated | None |
| Li et al. 2018 | Leave-one-subject-out; one feature vector per trial (18 feature types, 4 bands) | Linear SVM, no feature selection | not run | 55.31 (8.39) | Accuracy | > 5 high, < 5 low | None |
| same | same | SVM, best L1 feature selection | not run | 59.06 (8.68), AUC 0.605 | Accuracy, AUROC | same | None. Reported as an "upper bound": the setting was chosen on the test results |
| OmniEEG-Bench, Suppl. Table 5 | Subjects split 8:1:1; 10 s windows; frozen backbone + linear probe; up to 40 samples per subject per class | CBraMod | 0.515 +/- 0.021 | 0.519 +/- 0.020 | Balanced accuracy, 3 runs | Not stated ("high/low", 2 classes) | 0.500 (BENDR collapses to it) |
| same | same | LaBraM | 0.478 +/- 0.007 | 0.507 +/- 0.033 | same | same | same |
| same | same | REVE | 0.508 +/- 0.029 | 0.521 +/- 0.008 | same | same | same |
| same | same | BrainOmni | 0.505 +/- 0.024 | 0.530 +/- 0.011 | same | same | same |
| same | same | Best of 10 | 0.515 (CBraMod) | 0.533 (EEGMamba) | same | same | same |

Reading these:

- EEGain's own conclusion for the valence/arousal datasets: "the improvements over trivial baselines ... are small, if not non-existent." On DEAP no conv net beats the trivial accuracy on either axis.
- LibEER observes that several deep models "did not outperform the baseline SVM", and that DEAP and MAHNOB-HCI are harder than SEED, which it attributes to label quality: SEED's class is assigned per film clip, DEAP's is a self-rating.
- OmniEEG-Bench did not include DEAP in its full fine-tuning table (Suppl. Table 7), so there is no published fine-tuned foundation-model number for DEAP in the sources read. Its text names "arousal classification (DEAP-arousal)" among the "especially difficult" tasks.
- The other two foundation-model benchmarks do not include DEAP. EEG-FM-Bench (arXiv 2508.17742v2) covers SEED, SEED-V and SEED-VII, and splits those within-subject by trial "to ensure that our metrics are comparable with other baselines" (SEED: 15 trials split 9:3:3). NeuralBench-EEG's one emotion task is FACED (9 classes, 123 subjects); its per-task numbers are in figures only.
- For scale, EEG-FM-Bench's SEED results (3 classes, chance 33.3%, within-subject trial split, full fine-tuning, balanced accuracy): CBraMod 62.59, LaBraM 61.59, EEGPT 70.83, REVE 67.90. With a frozen backbone (multi-task setting, Table 4) CBraMod scores 44.32. SEED's label is a function of the video, so these are not evidence about self-rated DEAP labels; arXiv 2610.28422 argues that on SEED and FACED part of such accuracy is stimulus memorization, and that DEAP is not exposed to that shortcut because "the label is not fixed by the video."

## Within-subject, trial-level splits (no window of a test trial is seen in training)

| Source | Split | Model | Arousal | Valence | Metric | Label rule | Baseline reported |
|---|---|---|---|---|---|---|---|
| Koelstra et al. 2012, Table 7 | Leave-one-trial-out per participant; one feature vector per trial; Fisher feature selection | Gaussian naive Bayes on band power | 0.620 (F1 0.583) | 0.576 (F1 0.563) | Accuracy; F1 averaged over both classes | Threshold "in the middle" of the 9-point scale | Majority 0.644 / 0.586 accuracy (F1 0.389 / 0.368); class-ratio voting F1 0.500 |
| arXiv 2005.02619 | Leave-one-out per participant ("one sample"; read here as one trial, as in Koelstra) | SVM on band-power feature sets, mean over sets | 0.637 accuracy, 0.521 balanced | 0.604 accuracy, 0.544 balanced | Accuracy and balanced accuracy | Not captured | "Average class bias" 0.64 arousal, 0.59 valence |
| same | same | SVM, best single feature set (beta power) | not given | 0.5732 balanced | Balanced accuracy | same | Mean lower bound of the 95% credible interval is not above 0.5 |
| TSception paper, Table III | Per participant, 10 folds over trials; 4 s non-overlapping windows; 80/20 train/val inside the training folds | SVM | 60.37 (F1 57.33) | 55.19 (F1 57.87) | Accuracy, F1 | Threshold 5 | None |
| same | same | EEGNet (2,162 parameters) | 58.29 (F1 60.60) | 54.56 (F1 57.61) | same | same | None |
| same | same | ShallowConvNet | 61.19 (F1 61.19) | 59.42 (F1 62.26) | same | same | None |
| same | same | DeepConvNet | 61.03 (F1 62.58) | 59.92 (F1 62.04) | same | same | None |
| same | same | TSception (12,563 parameters) | 61.57 (F1 63.24) | 59.14 (F1 62.33) | same | same | None |
| EEGain, Table III | Leave-one-trial-out per participant | TSception, re-run in EEGain | 60.67 (F1 61.40) | 59.32 (F1 62.49) | Accuracy, F1 of the positive class | Threshold 5 | Trivial: 62.73 / 50.31 accuracy |
| LibEER, Table V | Per participant, trials split 0.6/0.2/0.2; 1 s windows; DE features | SVM on DE | 63.49 (16.72), F1 53.31 | 54.17 (18.67), F1 49.73 | Accuracy, F1 | Not stated | None |
| same | same | EEGNet | 61.30 (15.88), F1 53.26 | 51.50 (11.57), F1 47.85 | same | Not stated | None |
| same | same | TSception | 57.49 (11.86), F1 50.75 | 51.52 (9.54), F1 47.33 | same | Not stated | None |
| same | same | Best of 17 methods | 66.09 | 59.23 | same | Not stated | None |
| Khan et al. 2025, Fig. 3A | Leave-one-trial-out per participant; raw DEAP; PSD in 5 bands | kNN (k=5) | not run | about 52%, flat from 1 to 60 segments per trial | Accuracy | <= 5 low, > 5 high | None |
| OmniEEG-Bench, Suppl. Table 6 | Each participant's trials split 8:1:1, then pooled across participants; linear probe | CBraMod | 0.513 +/- 0.004 | 0.516 +/- 0.016 | Balanced accuracy | Not stated | 0.500 |
| same | same | LaBraM | 0.495 +/- 0.010 | 0.506 +/- 0.007 | same | same | same |
| same | same | Best of 10 | 0.514 (REVE) | 0.531 (FEMBA) | same | same | same |

Reading these:

- Arousal accuracies of 58-64% are at or below the 64% per-participant majority baseline. Valence accuracies of 54-60% straddle the 58% baseline. This is why Koelstra and the 2005.02619 tutorial both report a class-balanced metric, and why the tutorial's 0.637 arousal accuracy becomes 0.521 balanced.
- The small conv nets do not beat the SVM on arousal and beat it by about 4 points on valence in the TSception paper; in LibEER, with a validation split for model selection, the SVM is ahead of EEGNet and TSception on both axes.
- Standard deviations across participants are 7-19 points, with 40 trials per participant. A single test trial is 2.5% of a participant's data.

## Within-subject, window-level splits (windows of one trial on both sides)

| Source | Split | Model | Arousal | Valence | Metric | Label rule |
|---|---|---|---|---|---|---|
| Khan et al. 2025, Fig. 3A | 5-fold over all of a participant's segments | kNN (k=5) on PSD | not run | about 53% unsegmented, over 90% at 6 segments per trial, nearly 100% at 60 | Accuracy | Threshold 5 |
| LibEER, Table IV (replication in the original papers' settings) | "dependent, 9:1", 1 s windows | CDCN | 92.92 reported, 91.99 +/- 12.20 reproduced | 92.24 reported, 92.30 +/- 11.33 reproduced | Accuracy | Not stated |
| same | "dependent, 9:1", 3 s DE windows | ACRNN | 93.38 reported, 88.31 +/- 7.77 reproduced | 93.72 reported, 86.03 +/- 9.20 reproduced | Accuracy | Not stated |
| Moreno-Alcayde et al. 2024 | Share of the test participants' or test videos' segments moved into training, from 0 to all | One conv net; also kNN on hand-crafted features | not run | "from about random (~50%) to nearly 90% using the very same ML model" | Accuracy | > 5 high |

What the literature says about the leak:

- **Mechanism.** Adjacent windows of one 60 s trial are highly correlated and all carry the trial's single label. TSception paper: shuffling segments across the split "will give high classification results. But the accuracy will drop when the highly correlated segments are never seen by the model." With 40 trials per participant, recognizing which trial a window came from is enough to recover its label.
- **Size on DEAP.** Same library, same model, only the split changed: CDCN is 92% in LibEER's replication of its original setting and 57.71 (valence) / 63.37 (arousal) under LibEER's trial-level split; ACRNN goes from 86-88% to 53.52 / 61.83. Khan et al.: 52% to nearly 100%. That is 30-48 points.
- **It needs no brain.** Khan et al. recorded EEG from a watermelon, assigned random binary labels, and got the same inflation curve under segment-level k-fold.
- **Prevalence.** Khan et al. reviewed 101 DEAP papers: 58 had invalid segmentation and 4 were undeterminable; 33 tuned hyperparameters on test data; 27 selected features on the whole dataset; 13 had none of the four flaws ("nearly 87%" had at least one). Only 38 reported F1 or another class-aware metric. In their own experiments feature selection on the whole dataset added 14-18 points and tuning on the test set 21-27.
- **Overlap makes it worse.** Moreno-Alcayde et al.: "overlapping segments can be seen as one of the strongest forms of data leakage." With no test-subject data in training, their model "is essentially random."
- **LibEER's caveat on the originals.** It names ACRNN (its ref. 14) among the "trial-dependent configurations" that "can introduce data leakage risks". For CDCN it records the 9:1 setting without saying whether the split is by window; the 92% versus 58% gap is the evidence.
- **Other inflators the reviews name:** no validation set, so the reported number is the best test epoch (LibEER: this "greatly exaggerates the performance"); reporting accuracy alone on imbalanced labels (EEGain: 98.8% of analyzed experiments report accuracy, 21.9% any F1).

## Facts that bear on choosing a split and a label rule

- Any split must group by trial. Window-level numbers measure trial identification, and no figure from that family is a usable comparison point.
- Leave-one-subject-out is the protocol with the most published trial-safe DEAP numbers for small conv nets (EEGain), and the result there is chance. Within-subject trial-level (leave-one-trial-out or trial-grouped k-fold) is the protocol under which the original paper and the tutorial show a small above-chance effect, and it is what a per-user product would correspond to.
- Fixed threshold at 5 is the rule most of the literature uses, so it is the comparable one. It needs balanced accuracy or macro-F1 beside accuracy, and the per-participant baseline is as high as 64% for arousal.
- Per-participant median gives every participant about 20/20, so accuracy and balanced accuracy coincide and the baseline is 50-51%. No benchmark read here uses it, so its numbers have no external comparison. The median is computed from all 40 of a participant's ratings, which in a within-subject split includes the test trials' ratings.
- Metrics in use: accuracy almost everywhere; positive-class F1 and weighted F1 (EEGain); class-averaged F1 (Koelstra); balanced accuracy (OmniEEG-Bench, EEG-FM-Bench, NeuralBench, Khan et al., the 2005.02619 tutorial); AUROC rarely (Li et al.). Balanced accuracy is the one the foundation-model benchmarks share.
- EEGain's trivial baselines are the cheapest honesty check: a majority-class predictor for accuracy and a class-distribution sampler for F1, both fit on the training folds.

## Local label check

Run for this note against `backend/data/deap/s01.dat` to `s32.dat` in the main checkout (the Kaggle mirror named in `backend/data/DEAP_SETUP.md`), compared with the `participant_ratings.csv` described above, matched on participant and `Experiment_id`.

- Dominance and liking are identical in all 1,280 trials.
- Valence and arousal are identical in 841 trials and differ in 439. The 439 are exactly the trials where the published valence and arousal are both above 5, and in every one the local value is `9 - published` for both columns (participant 1, video 1: published 7.71 / 7.60, local 1.29 / 1.40). Local valence and arousal therefore range 0-9 on a 1-9 scale.
- Effect: the high-arousal/high-valence quadrant is empty in the local files (published share 34.3%), and low/low holds 55.7% (published 21.4%). The local high share at `>= 5` is 22.3% valence and 24.6% arousal, which is the "~24/76" split `DEAP_SETUP.md` attributes to DEAP; the published split is 57/43 and 59/41. The local valence-liking correlation is -0.07; the published one is 0.62, and Koelstra et al. report "high positive correlations between liking and valence."
- Label agreement between local and published ratings: 65.7% under `>= 5` on each axis; under the per-participant median rule, 43.3% for valence and 50.2% for arousal.

The cause was not investigated (the mirror, or a later edit to the files). Any accuracy measured with these files, including the recorded leave-one-subject-out figure, was scored against labels that differ from DEAP's on a third of trials.

## What could not be verified

- **LibEER column alignment.** Tables V and VI were aligned to method names by header order (SVM, EEGNet, CDCN, TSception, ACRNN, ...). The SVM, EEGNet, CDCN and TSception cells are the first four and are the least at risk; "DGCNN" is the eighth. LibEER's DEAP binarization threshold and F1 averaging are not stated in the paper text read.
- **OmniEEG-Bench.** In the arXiv HTML, the first data row of Suppl. Tables 5 and 6 carries identical values under differently ordered model headers, so a header misprint in one of them cannot be ruled out; the DEAP rows are quoted as printed. Its DEAP label rule is not in the paper, and its repository (`ncclab-sustech/omni-eegbench`) could not be listed. Its linear probe subsamples up to 40 windows per subject per class, which rebalances classes, so its numbers are not comparable to accuracy figures.
- **Fine-tuned foundation models on DEAP.** No source read reports one. The only foundation-model DEAP numbers are OmniEEG-Bench's frozen-backbone linear probes.
- **Logistic regression specifically.** No trial-safe DEAP result with logistic regression was found; the classic-model rows are SVM, naive Bayes and kNN.
- **EEGain's trivial baseline versus its stated threshold.** EEGain states low = rating <= 4.5 for its LOSO runs, under which the ratings file gives a 63-64% majority; its table shows 0.57 / 0.59, which matches a threshold of 5. Which was actually used is unclear.
- **TSception's axis order.** The TSception paper's Table III reads arousal 61.57 / valence 59.14; LibEER's Table IV quotes the same two numbers with the axes swapped. The TSception paper is followed here.
- **arXiv 2005.02619's unit of "one sample"** in its leave-one-out is read as one trial because it cites agreement with Koelstra; its label rule was not captured.
- **Moreno-Alcayde et al. and NeuralBench** give their per-condition numbers only in figures.
- **The ratings file** is a GitHub mirror, not the official download. Its match to the paper's published class shares and majority baselines is the only authentication.
- **Li et al. 2018** drops ratings of exactly 5 and reports valence only.

## Sources

- Koelstra et al., "DEAP: A Database for Emotion Analysis Using Physiological Signals", IEEE Trans. Affective Computing 3(1), 2012. Read from the University of Geneva open archive copy: <https://archive-ouverte.unige.ch/unige:47405>
- EEGain: "Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework", arXiv 2505.18175v1. <https://arxiv.org/abs/2505.18175>
- LibEER: "A Comprehensive Benchmark and Algorithm Library for EEG-based Emotion Recognition", arXiv 2410.09767. <https://arxiv.org/abs/2410.09767>
- OmniEEG-Bench: "A Standardized Evaluation Benchmark for EEG Foundation Models", arXiv 2606.00815. <https://arxiv.org/abs/2606.00815>
- EEG-FM-Bench, arXiv 2508.17742v2. <https://arxiv.org/abs/2508.17742>
- NeuralBench: "A Unifying Framework to Benchmark NeuroAI Models", arXiv 2605.08495v1. <https://arxiv.org/abs/2605.08495>
- Khan et al., "The Role of Review Process Failures in Affective State Estimation: An Empirical Investigation of DEAP Dataset", arXiv 2508.02417. <https://arxiv.org/abs/2508.02417>
- Moreno-Alcayde, Traver, Leiva, "Sneaky emotions: impact of data partitions in affective computing experiments with brain-computer interfacing", Biomedical Engineering Letters 14(1), 2024. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10769959/>
- Ding et al., "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition", arXiv 2104.02935. <https://arxiv.org/abs/2104.02935>
- Li et al., "Exploring EEG Features in Cross-Subject Emotion Recognition", Frontiers in Neuroscience 12:162, 2018. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5867345/>
- Mowla et al., "Affective Brain-Computer Interfaces: A Tutorial to Choose Performance Measuring Metric", arXiv 2005.02619. <https://arxiv.org/abs/2005.02619>
- "How Much of Cross-Subject EEG Emotion Recognition Is Stimulus Memorization?", arXiv 2610.28422. <https://arxiv.org/abs/2610.28422>
- DEAP `participant_ratings.csv`, public mirrors: `tomdamelio/arousal_decoding` (`data/`) and `hsiehjackson/Emotion-Recognition-DEAP_GSR` (`zip/`) on GitHub.
