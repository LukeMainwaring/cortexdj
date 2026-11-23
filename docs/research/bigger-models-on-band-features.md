# Bigger models on one window's band features: what beats linear on a trial-safe split

Researched 2026-11-23. Every figure below was read that day from the paper's own text or tables (arXiv HTML parsed by script, or the PDF through `pdftotext`) or from the benchmark's source code on GitHub, not from memory or a secondary summary. Builds on [deap-benchmarks-by-protocol.md](deap-benchmarks-by-protocol.md) (branch `research/ml-rebuild-deap-benchmarks`) and [2026-11-22-signal-libraries-and-features.md](2026-11-22-signal-libraries-and-features.md) (branch `research/ml-rebuild-signal-libraries`); the protocol tables, the leak evidence and the feature comparison there are not repeated.

**The question.** The quadrant classifier is `nn.Linear(128, 2)` on one 4-second window's differential entropy (32 channels, 4 bands), one model per participant, 32 training trials (480 windows) per fit, scored 0.583 mean balanced accuracy (arousal 0.558, valence 0.609). Which models that read the same one-window feature vector and could be a torch module beat a linear classifier under a split that keeps a trial's windows together, by how much, and with what regularization?

**Confidence note.** No model was trained for this note. No source read compares a plain MLP with a linear model on DEAP band features under a trial-safe split, so the DEAP answer rests on one benchmark (LibEER) whose baseline is an RBF SVM, not a linear model. The list at the end says what could not be verified.

## Summary

1. **The honest answer for DEAP is that nothing reliably beats a simple classifier here.** In the one benchmark that runs feature-vector models on DEAP differential entropy with a trial-safe split and a validation set (LibEER, 17 methods), the best gain over the SVM baseline in macro-F1 is 2.0 points (arousal) and 1.7 points (valence), the standard deviation across participants is 14 to 20 points, and more than half the bigger models score below the SVM (5 of 9 on valence, 7 of 9 on arousal).
2. **MLP-family models gain nothing on DEAP.** The deep belief network (two hidden layers, 300 and 400 units) moves macro-F1 by -1.1 (valence) and -0.7 (arousal) against the SVM. The two MLPs with a domain-alignment loss (DAN, DANN) move it by -4.7 to +1.2.
3. **Graph convolutions over the channel-by-band matrix are the only family with a repeatable gain, and only on SEED-type data.** DGCNN gains 7.5 accuracy points over the SVM on SEED and 4.6 on SEED-IV under LibEER's trial-safe split. On DEAP the same model gains -0.7 to +0.6 macro-F1. SEED gives each model about 2,000 training windows from 9 four-minute trials and a label assigned per film clip; DEAP gives one-minute trials and a self-rating.
4. **The classic "deep beats linear" result is small and did not reproduce.** Zheng and Lu 2015 (SEED, 9 training trials, 6 test trials) report a deep belief network at 86.08% against a linear SVM at 83.99% and L2 logistic regression at 82.70%, with no validation split. LibEER reproduced that network in the original setting at 81.18%, below the linear SVM's published figure.
5. **Large gains for feature-vector models on DEAP come from splits that are leaking or unstated.** An MLP on exactly our input shape (2,400 windows by 128 features per participant) reports 94% F1 with 5-fold over windows. A single-hidden-layer network (GELM) reports 69.67% against a linear SVM's 47.57% on DEAP quadrants with a 5-fold whose unit the paper does not state.
6. **No regularization recipe is shown to make a bigger model work at this sample size.** What the trial-safe sources used: dropout 0.5 to 0.7 on the last fully connected layer, weight decay 1e-4, one or two hidden layers of 64 to 400 units, and the epoch picked on held-out validation trials. Picking the epoch on a validation set was worth 1 to 3 accuracy points on average in LibEER's own ablation, against a spread of about 20, and it costs trials: LibEER trains on 24 of 40.
7. **A gain of the size the literature shows cannot be detected with 40 trials a person.** The reported gains on DEAP (0 to 2 points) are smaller than the bake-off's 0.03 threshold and smaller than the standard error of a 32-participant mean.

## The one trial-safe DEAP comparison: LibEER

LibEER (arXiv 2410.09767v3, 22 July 2025; code at `XJTU-EEG/LibEER`) re-implements 17 published methods in PyTorch and runs them under one protocol. The version read here has more methods and different columns than the one the earlier note read.

Protocol, from the paper (section III) and the repository:

- **Features:** differential entropy in five bands (0.5 to 4, 4 to 8, 8 to 14, 14 to 30, 30 to 50 Hz) from 1-second windows with no overlap, then smoothed across windows with a linear dynamic system. On DEAP that is 160 features a window and 60 windows a trial. The smoothing uses neighbouring windows, which is outside our recipe.
- **Split, subject-dependent:** each participant's trials are divided 0.6 / 0.2 / 0.2 into training, validation and test, with all of a trial's windows on one side. For DEAP that is 24 / 8 / 8 trials, drawn once per participant (`docs/PresetSetting.md`).
- **Model selection:** the reported test score is from the epoch with the best macro-F1 on the validation trials.
- **Labels and metric:** the DEAP run commands pass `-bounds 5 5`, a threshold at 5 (`DGCNN_train.py`, `DBN_train.py`). F1 is macro-F1 (`utils/metric.py`), the closest reported metric to our balanced accuracy.
- **The baseline is not linear.** It is `svm.SVC(kernel='rbf', C=1.0, gamma='scale')`, untuned (`models/SVM.py`).

Which of the 17 read one window's feature vector: the SVM; the "DNN" group (DBN, a deep belief network; DAN and DANN, MLPs trained with a domain-alignment loss; PR-PL, prototype learning on an MLP); and the graph group (DGCNN, GCBNet, GCBNet_BLS, RGNN, NSAL-DGAT), which treat the window as a channels-by-bands matrix and learn or fix an adjacency between channels. The CNN, RNN and Transformer groups read raw signal or sequences of windows and are out of scope.

LibEER Table V, subject-dependent, mean over participants with the standard deviation across participants in parentheses. Gains are against the SVM column.

| Model | DEAP valence: acc | DEAP valence: macro-F1 | F1 gain | DEAP arousal: acc | DEAP arousal: macro-F1 | F1 gain |
|---|---|---|---|---|---|---|
| SVM (RBF) | 54.17 (18.67) | 49.73 (18.92) | | 63.49 (16.72) | 53.31 (14.39) | |
| DBN | 56.08 (17.38) | 48.61 (19.33) | -1.1 | 64.60 (19.42) | 52.61 (19.85) | -0.7 |
| DAN | 57.89 (19.18) | 50.97 (18.88) | +1.2 | 60.51 (17.32) | 50.96 (16.53) | -2.4 |
| DANN | 57.70 (19.93) | 50.23 (20.97) | +0.5 | 62.38 (16.78) | 48.57 (16.96) | -4.7 |
| PR-PL | 59.23 (14.69) | 47.42 (17.67) | -2.3 | 62.15 (15.90) | 53.22 (17.99) | -0.1 |
| DGCNN | 56.07 (17.15) | 49.08 (17.50) | -0.7 | 62.68 (19.66) | 53.94 (20.10) | +0.6 |
| GCBNet | 56.49 (18.17) | 50.36 (19.57) | +0.6 | 65.95 (17.61) | 55.34 (17.86) | +2.0 |
| GCBNet_BLS | 57.02 (15.07) | 51.40 (17.25) | +1.7 | 61.07 (16.56) | 50.43 (16.36) | -2.9 |
| RGNN | 55.90 (16.24) | 47.25 (17.55) | -2.5 | 66.09 (13.91) | 49.27 (12.86) | -4.0 |
| NSAL-DGAT | 52.21 (16.19) | 48.87 (18.34) | -0.9 | 60.31 (17.29) | 50.21 (18.91) | -3.1 |

Reading it:

- Macro-F1 runs from 47.2 to 51.4 on valence and 48.6 to 55.3 on arousal for every model. The SVM is in the middle of both ranges.
- Accuracy gains look larger (up to +5.1 on valence for PR-PL) but come with a lower macro-F1, which is the pattern of a model leaning on the majority class. The per-participant majority baseline at this threshold is 58.6% valence and 64.4% arousal (earlier note), so no accuracy in the table is clearly above it.
- No model is ahead on both axes by more than 0.6 macro-F1 (GCBNet: +0.6 and +2.0 is the best pair).
- LibEER's own reading: several deep models do not outperform the baseline SVM, which it attributes to how little EEG data there is (Observation 3).

The same table on the other datasets shows where a bigger model does help:

| Dataset (classes; training trials) | SVM acc / F1 | DBN | DGCNN | GCBNet | Best feature-vector model |
|---|---|---|---|---|---|
| SEED (3; 9 of 15 a session) | 75.08 / 70.82 | 71.88 / 67.39 | 82.55 / 79.89 | 80.56 / 77.29 | DGCNN, +7.5 / +9.1 |
| SEED-IV (4; about 14 of 24) | 47.80 / 40.17 | 45.56 / 37.61 | 52.39 / 45.94 | 53.28 / 46.26 | PR-PL 54.59 / 48.24, +6.8 / +8.1 |
| MAHNOB-HCI valence (2; 12 of 20) | 64.83 / 55.99 | 62.03 / 52.84 | 67.83 / 54.78 | 66.84 / 54.61 | GCBNet_BLS 70.60 / 58.96, +5.8 / +3.0 |
| MAHNOB-HCI arousal (2; 12 of 20) | 63.61 / 50.99 | 68.51 / 57.18 | 67.29 / 58.04 | 64.89 / 57.43 | DGCNN by F1, +3.7 / +7.1 |

- On SEED and SEED-IV, DGCNN and GCBNet are 4.6 to 7.5 accuracy points over the SVM; RGNN is +1.5 and -2.4. The MLP-family models (DBN, DAN, DANN) are below the SVM on both. The standard deviations are 15 to 24 points.
- MAHNOB-HCI is the closest relative of DEAP (self-rated valence and arousal, 20 trials a person). Results there are mixed: DBN is 3 F1 points below the SVM on valence and 6 above on arousal.
- LibEER explains the SEED-to-DEAP difference by label quality: SEED's class is fixed per film clip, DEAP's and MAHNOB's are self-ratings. A SEED trial is also about four minutes, so 9 training trials are roughly 2,000 one-second windows.

## SEED, original papers: the gain over a linear model and whether it holds

These are trial-safe in the sense that matters here (first 9 trials of a session for training, last 6 for testing), but none has a validation split, so each tuned setting was chosen with the test trials in view.

| Source | Model on one window's DE vector (310 features) | Accuracy | Gain over linear | Regularization and tuning reported |
|---|---|---|---|---|
| Zheng and Lu 2015, Table II and text | Linear SVM | 83.99 (9.72) | | C searched over 2^-10 to 2^10 |
| same | L2 logistic regression | 82.70 (10.38) | | Penalty searched over 1.5 to 10 |
| same | Deep belief network, 2 hidden layers | 86.08 (8.34) | +2.1 over SVM, +3.4 over LR | Hidden sizes searched in 200 to 500 and 150 to 500; unsupervised pretraining; no dropout or weight decay reported |
| Song et al. 2018, as quoted in the RGNN paper's Table 1 | DGCNN | 90.40 (8.49) | +6.4 over SVM | Not read (see below) |
| Zhong et al. 2020 (RGNN), Table 1 and section 5.3 | RGNN | 94.24 (5.95) | +10.3 over SVM | 2 graph layers, dropout 0.7 on the output layer, batch 16, L1 on the adjacency, L2 and label-noise level tuned per experiment |

What LibEER found when it re-ran them (Table IV, original settings, 9 : 6 split, no validation):

- DBN: 81.18 +/- 8.13, against the published figure, a drop of more than 5 points and below the linear SVM's published 83.99.
- DGCNN: 89.48 +/- 8.49, within a point of the published 90.40.
- RGNN: 84.66 +/- 10.74, against 94.24 published, a drop of 9.6 points.
- LibEER notes that none of the original settings uses a train, validation and test split, and that reporting the best test epoch exaggerates performance. Under its own three-way split the same models fall further: DGCNN 82.55, RGNN 76.55, DBN 71.88, with the RBF SVM at 75.08.

Two more facts from Zheng and Lu's own Table III: on single-band inputs (62 features) the SVM is ahead of the network in four of five bands (alpha 66.64 against 64.01, beta 80.76 against 78.92), and on the asymmetry features the SVM is ahead on the full input too. The network's advantage appears only on the largest input.

## Results from window-level or unstated splits

| Source | Model | Reported | Split | Why it is not evidence |
|---|---|---|---|---|
| Kumar and Molinas, CEUR-WS Vol-3368 paper 4 | MLP, hidden layers 256, 256, 128, 64, on DEAP DE from 1 s windows; per participant the input is 2,400 windows by 128 features (our 32 channels by 4 bands) | F1 94.5% arousal, 94% valence | 5-fold: "the dataset is divided into five equal sets", per participant; no grouping by trial is described | Windows of one trial are on both sides. This is the window-level leak the earlier note documents (kNN on DEAP going from about 52% under leave-one-trial-out to over 90% under segment-level folds). |
| Zheng, Zhu and Lu, arXiv 1601.02197, Table 3 and section 5.1 | GELM, an extreme learning machine: one random hidden layer 10 times the input width, a ridge penalty and a graph penalty on the output weights, solved in closed form; DEAP DE, four quadrants | 69.67% against linear SVM 47.57%, logistic regression 40.86%, RBF SVM 39.21% | "5-fold cross-validation"; the unit of a fold is not stated; features smoothed across windows | A 22-point gap between a one-hidden-layer network and a linear SVM is what trial identification looks like. The split cannot be confirmed as trial-safe. |
| same, section 5.2 | GELM on SEED DE | 91.07 (7.54) against logistic regression 84.08 (8.77) and linear SVM 83.26 (9.08) | "5-fold cross-validation", unit not stated | Same. The same section shows smoothing mattering more than the classifier: 70.82 without smoothing, 91.07 with it. |
| torcheeg `KFoldGroupbyTrial` and `KFoldPerSubjectGroupbyTrial` | Any model in torcheeg's DEAP examples | (splitter, not a result) | Splits the windows inside each trial: for the first fold, the first 80% of every trial's windows train and the last 20% test | Despite the name, every trial is on both sides. A paper that says "grouped by trial" with torcheeg is using a within-trial split. |

## Regularization: what the trial-safe sources used

No source shows a regularization setting that turns a losing bigger model into a winning one at this sample size. What was used where the split was trial-safe:

| Source | Model | Width and depth | Dropout | Weight decay | Epoch selection | Other |
|---|---|---|---|---|---|---|
| LibEER (`config/model_param/DBN.yaml`, `DBN_train.py`) | DBN on DEAP | 300 then 400 hidden units | none | none found | Best validation macro-F1 over 100 epochs | lr 0.001, batch 256 |
| LibEER (`config/model_param/DGCNN.yaml`, `DGCNN_train.py`) | DGCNN on DEAP | One graph-convolution layer of 64 or 128 channels (the file's comment gives 128 for DEAP), Chebyshev order 2 | 0.5 on the fully connected layer | AdamW, 1e-4 | Best validation macro-F1 over 150 epochs | lr 0.01, batch 512 |
| RGNN paper | RGNN on SEED | 2 graph layers | 0.7 on the output layer | L2, tuned per experiment | None (no validation split) | L1 on the adjacency, soft labels with a tuned noise level, batch 16 |
| Zheng and Lu 2015 | DBN on SEED | 2 hidden layers, 150 to 500 units, searched | none | none | None | Unsupervised layer-wise pretraining |

On early stopping, LibEER's ablation (Table VII, SEED-IV and four-class MAHNOB-HCI, subject-dependent) compares its validation-picked epoch with training on training plus validation trials and taking the last epoch:

- Mean accuracy over five methods: 43.90 against 43.04 on SEED-IV, 47.49 against 44.55 on MAHNOB-HCI. Mean macro-F1: 35.91 against 35.71, and 34.32 against 32.27.
- It is not uniform. For the DBN on SEED-IV the last-epoch run is ahead, 49.50 against 45.56.
- Standard deviations across participants are 19 to 28 points in every cell, so these differences are not separable from noise.
- The same section's seed experiment: changing the random seed alone moves a method's result by about 10 to 20 points (the seed also redraws the split).

So the published evidence for early stopping at this scale is a 1 to 3 point average edge that costs a fifth of the trials. The repo's choice (ticket 07) of a fixed epoch count and no held-out trials is not contradicted by anything read here.

## Why the answer is "not reliably", in numbers

- **Trials, not windows, are the sample size.** A fit sees 32 trials. The 15 windows of a trial share one label and are near-copies, so 480 windows do not constrain a model 15 times better than 32 trials do. A one-hidden-layer MLP with 64 units on 128 features has 8,386 weights against the linear model's 258 (own arithmetic).
- **The noise floor.** Varoquaux (arXiv 1706.07581) shows cross-validated accuracy on about 100 samples carries an error bar of about plus or minus 10 points, and that the spread across folds understates it. One participant here has 40 trials.
- **Against the bake-off's threshold.** With a between-participant standard deviation of 14 to 20 points (LibEER) and 32 participants, the standard error of a mean is about 2.5 to 3.5 points (own arithmetic; a paired comparison on the same folds is tighter, and no source reports the paired figure). The largest trial-safe gain reported on DEAP is 2.0 macro-F1 points. A true gain of that size would not clear a 0.03 bar, and a measured 0.03 would be about one standard error.
- **The wider EEG literature agrees on direction.** Lotte et al. 2018, the standard review of EEG classifiers, concludes that shrinkage-regularized linear classifiers are the ones suited to small training sets (it gives 20 to 100 trials per class as typical) and that deep networks had not shown convincing gains. The MOABB benchmark (arXiv 2404.15319), on motor imagery from raw signal rather than band features, finds deep pipelines ranking below Riemannian and linear ones and needing more than about 150 trials per class to reach satisfactory scores, where the linear and Riemannian pipelines do well with fewer than 50. Neither is about emotion or DEAP.

## Per model: the short answer

| Model | Trial-safe gain over a simple baseline on DEAP | Elsewhere, trial-safe | Regularization used | Leaking results exist? |
|---|---|---|---|---|
| Small MLP on the feature vector | No direct result. Nearest: DBN, -1.1 / -0.7 macro-F1 against an RBF SVM (LibEER) | SEED: +2.1 over linear SVM in the original paper, not reproduced; -3.2 against RBF SVM in LibEER | Width 300 to 400; validation-picked epoch | Yes: 94% with window-level 5-fold (CEUR-WS) |
| MLP with a domain-alignment loss (DAN, DANN) | -4.7 to +1.2 macro-F1 | SEED: 13 to 15 points below the SVM | Not read | Not checked |
| Extreme learning machine with graph penalty (GELM) | None found | None found | Ridge and graph penalty, closed form | Split unit unstated on both DEAP (+22) and SEED (+7) |
| Graph convolution over channels by bands (DGCNN, GCBNet) | -0.7 to +2.0 macro-F1 | SEED +5.5 to +7.5 accuracy, SEED-IV +4.6 to +5.5, MAHNOB-HCI +1.3 to +3.7 | Dropout 0.5, weight decay 1e-4, one layer of 64 to 128 channels, validation-picked epoch | Not on DE in the sources read |
| RGNN | -2.5 / -4.0 macro-F1 (LibEER); 57.90% valence accuracy under trial-wise 10-fold with nested validation (LGGNet paper, Table III), level with EEGNet 56.38 and TSception 57.46 on raw signal | SEED +1.5 against RBF SVM in LibEER; published +10.3 fell 9.6 points on reproduction | Dropout 0.7, L1 and L2, label noise | No |

## What could not be verified

- **A plain MLP against a linear model on DEAP band features, trial-safe.** No source read reports it. The DEAP conclusion for MLPs is by proxy: a deep belief network and two domain-alignment MLPs against an RBF SVM.
- **A linear baseline in LibEER.** Its SVM is RBF with default settings, untuned. Whether a linear model would sit above or below it on DEAP is not known from the sources; Zheng, Zhu and Lu report linear ahead of RBF on SEED (83.26 against 78.21) under an unstated split.
- **Balanced accuracy.** LibEER reports accuracy and macro-F1, not balanced accuracy, and at 1-second windows with cross-window smoothing and five bands, not our 4-second, four-band, unsmoothed features. Its numbers are not directly comparable to 0.583.
- **LibEER hyperparameters.** Learning rates, batch sizes and epoch counts were read from example commands in comments at the top of the training scripts, not from a results log. The paper says unreported settings were set by tuning and experience. DGCNN's DEAP layer width (64 active in the config file, 128 in its comment) is ambiguous. No weight decay was found for the DBN, but its trainer file (`Trainer/DBNTraining.py`) was not read.
- **DAN and DANN within one participant.** How LibEER applies a domain-alignment loss in the subject-dependent task (which windows are the target domain) was not read; if unlabeled test windows are used, those two rows are transductive.
- **LibEER's reported DBN figure.** Its Table IV gives the published DBN accuracy as 86.91; Zheng and Lu's paper says 86.08. The reproduction figure (81.18) is quoted as printed.
- **The DGCNN paper itself** (Song et al., IEEE Trans. Affective Computing, 2018) was not read; its SEED figure is taken from the RGNN paper's table and LibEER's Table IV, which agree. Its regularization is known only through LibEER's config comment ("0.5 set by official").
- **Zheng, Zhu and Lu's fold unit.** The paper says 5-fold cross-validation for both DEAP and SEED and does not say whether folds are over windows or trials. It is listed under unstated splits for that reason, not because a leak was shown.
- **The CEUR-WS paper's split** is inferred from its wording (five equal sets of a participant's 2,400 windows) and its 94% result; it does not say "windows are shuffled".
- **RGNN's input in the LGGNet comparison.** The LGGNet paper does not say in the text read which features its RGNN baseline used on DEAP.
- **Zheng and Lu 2015** was read from the authors' submitted manuscript on the SJTU site, not the IEEE version. It describes no validation split; how the searched settings were chosen is not stated.
- **Paired statistics.** No source reports a paired test of a feature-vector model against a simple baseline on DEAP, so "not distinguishable" rests on the unpaired spreads.

## Sources

- LibEER: "A Comprehensive Benchmark and Algorithm Library for EEG-based Emotion Recognition", arXiv 2410.09767v3 (22 July 2025). <https://arxiv.org/abs/2410.09767>. Code read on `main` (last push 2026-03-18): <https://github.com/XJTU-EEG/LibEER> (`LibEER/models/SVM.py`, `LibEER/models/DBN.py`, `LibEER/models/DGCNN.py`, `LibEER/config/model_param/DBN.yaml`, `LibEER/config/model_param/DGCNN.yaml`, `LibEER/DBN_train.py`, `LibEER/DGCNN_train.py`, `LibEER/utils/metric.py`, `LibEER/utils/args.py`, `docs/PresetSetting.md`).
- Zheng and Lu, "Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks", IEEE Trans. Autonomous Mental Development 7(3), 2015. Authors' manuscript: <https://bcmi.sjtu.edu.cn/home/zhengweilong/pdf/TAMD2014_zwl_final_Submitted.pdf>
- Zheng, Zhu and Lu, "Identifying Stable Patterns over Time for Emotion Recognition from EEG", arXiv 1601.02197. <https://arxiv.org/abs/1601.02197>
- Zhong, Wang and Miao, "EEG-Based Emotion Recognition Using Regularized Graph Neural Networks", arXiv 1907.07835. <https://arxiv.org/abs/1907.07835>
- Ding et al., "LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface", arXiv 2105.02786. <https://arxiv.org/abs/2105.02786>
- Kumar and Molinas, MLP and CNN on differential entropy for SEED and DEAP, CEUR Workshop Proceedings Vol-3368, paper 4. <https://ceur-ws.org/Vol-3368/paper4.pdf>
- torcheeg, `torcheeg/model_selection/k_fold_groupby_trial.py` and `k_fold_per_subject_groupby_trial.py` on `main`. <https://github.com/torcheeg/torcheeg>
- Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 2018, arXiv 1706.07581. <https://arxiv.org/abs/1706.07581>
- Lotte et al., "A Review of Classification Algorithms for EEG-based Brain-Computer Interfaces: A 10-year Update", J. Neural Engineering 15(3), 2018. Open copy: <https://hal.science/hal-01846433>
- Chevallier et al., "The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark", arXiv 2404.15319. <https://arxiv.org/abs/2404.15319>
- Earlier notes in this repo: `docs/research/deap-benchmarks-by-protocol.md` (branch `research/ml-rebuild-deap-benchmarks`, commit d5674d6) for the majority baselines, the TSception, EEGain and Khan et al. figures, and the leak evidence.
