# Pilot: spatial samples versus device diversity, all PKC

Authorized in chat: create this comparison with physics and send it to Kaggle sgarizado. This standalone implementation pilot does not assert a new published managed protocol or a result.

## Question and scope
With the repaired r15 physical system and common data pretraining, does allocating a training update to more devices or more spatial samples yield the better validation-error/runtime tradeoff? This concerns stochastic PKC refinement, not training from scratch, a permanently coarsened mesh, or a universal statement about PINNs.

Dataset: existing base_m6.h5; 1384 train, 175 validation, 172 test. Original fixed 250 spatial nodes per device; every voltage retained. Test is untouched. One seed 0: exploratory selection only, no significance claim or training-seed confidence interval. Repeat selected configurations with independent seeds before drawing a paper-level conclusion.

## Factorial design

| Case | Devices/update | Spatial nodes/device | Product before voltage expansion |
|---|---:|---:|---:|
| A | 64 | 75 | 4800 |
| B | 64 | 150 | 9600 |
| C | 128 | 75 | 9600 |
| D | 128 | 150 | 19200 |

Every case uses data + Poisson + both current relations + both continuities (PKC), with Auger and corrected current transforms. No data-only control is needed for this within-PKC sampling question. Same 256x5 float64 network, optimizer hyperparameters and gradient clipping. Only device count and spatial sample count vary between cases.

B versus C is the main allocation comparison. Its point budget is equal before voltage expansion; devices have different voltage counts, so actual row counts and elapsed times are recorded rather than claiming exact equal compute. A/B and C/D isolate the spatial-sample factor; A/C and B/D isolate device count. Interaction may prevent either factor having a universal winner.

One shared data-only pretraining of 300 stochastic updates, 254 devices, 150 nodes. Save and load exactly the same checkpoint in every case. Compute Eq.39 weights once at that checkpoint using 128 training devices and all 250 nodes; freeze and share them. Then 200 Adam updates per case, logs and validation J-V metrics every 50 updates plus initial/final. No L-BFGS in this pilot: it would add a second changing sampling/optimization stage and obscure the Adam comparison. Results are not the planned 2000/500 production run.

## Sampling control
Nested device batches at a common update (64 is a subset of 128); independent per-device RNG makes 75 selected nodes a subset of the 150 for common devices. All voltages use the same spatial draw in an update. Draws change across updates. Region sizes in the stored pool: 25 left / 150 junction / 25 right / 50 bulk. Quotas are 15/90/15/30 at 150 and 7/45/8/15 at 75. Uniform draws without replacement within each region. For region size N_r and quota q_r at k total samples, use weight k*N_r/(250*q_r) on each sampled squared error. This inverse-inclusion correction makes both estimators target the identical 250-node spatial mean, despite the odd edge quota at 75. Same weighting for data and all physical residuals. Accumulation weights chunks by row count. Longer voltage curves retain their existing greater training weight.

Unlike the preceding timing smoke, this scientifically controlled comparison does not force region-extreme samples: doing so changes their inclusion probability as resolution changes. Full stored-grid validation still checks those positions. This choice applies equally to all four cases.

## Evaluation and decision
Primary: median per-device validation J-V curve RMSE (mA/cm2), with per-material medians and per-device values retained. Secondary: PCE error in percentage points, Jsc, Voc, FF, invalid-Voc counts, and all five physical MSEs. Dense final evaluation uses every validation device, every voltage and all 250 stored nodes, independent of training resolution. No test evaluation.

Report at equal update count, and show J-V error versus measured cumulative training time and actual row evaluations at saved checkpoints. For time-matched comparisons use observed checkpoints only; do not interpolate a fictional trained model. Count device visits and distinct spatial nodes observed. A 75-node draw can eventually cover all 250 stored positions, so do not call it a 75-node permanent grid.

Predeclared exploratory screen for reducing nodes at a fixed device count: final curve RMSE <=1.05 times its 150-node counterpart; PCE error increase <=0.2 percentage points; no increase in invalid-Voc count; no individual physical MSE >1.5 times the corresponding dense-run value (numerical floor 1e-12). These are pragmatic pilot tolerances, not published standards or a statistical noninferiority test. Show all metrics even if the screen fails. For B versus C report the direct paired per-device errors and the time/error tradeoff; conflicting metrics or unstable training imply inconclusive rather than a forced winner. One seed and 200 updates cannot establish asymptotic convergence or generalization outside this dataset.

## Cost, execution and verification
Two T4 GPUs: GPU0 cases A,D; GPU1 cases B,C, balancing projected work. Shared pretraining is performed once first. Estimated 4-6 hours wall time including pretraining and dense evaluation, extrapolated from the previous 254x150 PKC timing (about65s/update), not a measured guarantee. First remote smoke: same runner with 1 pretraining update, 2 PKC updates and 4 devices per split; then the 300/200 full pilot. Local checks cover nested samples, exact regional estimator mass, streamed versus direct gradients, and all four case executions with dense validation.

## External rationale (verified in this session, 2026-11-16)
Nabian, Gladstone and Meidani (2021), Efficient training of physics-informed neural networks via importance sampling: https://arxiv.org/abs/2104.12325 . Supports distinguishing sampling distributions from their weighted empirical objective; does not establish the optimum batch size for this solar dataset.
Krishnapriyan et al. (NeurIPS2021), Characterizing possible failure modes in physics-informed neural networks: https://arxiv.org/abs/2109.01050 . Motivates treating optimization success and physics residuals as measurements, rather than assuming additional points necessarily improve learning. Neither source supplies a standard protocol or universal tolerance for this private dPV parameter sweep; the operational criteria above are declared pilot choices.
