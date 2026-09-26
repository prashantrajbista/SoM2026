# Task 2 literature review — multi-modal-enhanced channel prediction

Scope: Task 2 predicts CSI over the remaining subcarriers from RGB image +
2D UE location + CSI over the first `Nc` subcarriers (inputs may be
missing/noisy), scored by NMSE, under the challenge's constraint that the
Task 1/2/3 models share >80% of weights off one wireless-foundation-model
backbone, and are additionally scored on average parameter count and FLOPs.
See [[challenege_statement]].

## What we've already found in this repo (read before acting on anything below)

From `notes/results.md` and `experiments/task2_*/summary.json` (500 labelled
samples, 400 train / 100 val split, dB = `10*log10(NMSE)`, lower is better):

| approach | NMSE (dB) | note |
|---|---|---|
| persistence (copy `X_prev`) | -19.96 | zero-training baseline |
| **angle-delay neighbourhood ridge, radius 3** | **-29.5 to -31.4** (5-fold) | best result on this branch, linear, CSI-only |
| linear rank-1 (`linear_r1`) | -29.07 | also CSI-only, linear |
| WiFo2 default (mask 0.5, rgb+pos) | -5.89 | far worse than persistence |
| WiFo2 + physics ridge overlay (`task2_physics`, with rgb+pos) | -29.54 | matches linear, `alpha≈0.002` — the learned WiFo2 correction contributes almost nothing |
| WiFo2 residual, masked | -29.05 | ties linear only when scored on the *masked* subset |
| WiFo2 residual, full | -19.05 | collapses back toward persistence on the full set |
| WiFo2 + MAE pretrain, best (30 epochs) | -9.31 | MAE helps WiFo2's own path but stays far below linear |
| WiFo2 + MAE 30ep + unfreeze 2 encoder blocks | -10.11 | best WiFo2-path result, still 19+ dB worse than linear |
| unfreeze 0/1/2 encoder blocks (no MAE) | -7.4 to -8.2 | unfreezing alone doesn't close the gap |

**The load-bearing fact for this whole review:** on this dataset, a plain
linear/ridge model on CSI alone beats every WiFo2-backbone variant tried by
~20 dB, and adding RGB+position to the linear model changes almost nothing
(`alpha≈0.002`, `-29.536` vs `-29.540` dB CSI-only). So "make the multi-modal
fusion fancier" has *not* been the bottleneck here — the deep backbone itself
is the bottleneck for this task, and RGB/location have not yet been shown to
carry predictive signal beyond what's already in the leading CSI subcarriers.
That reframes the literature question: not "how do others fuse RGB+CSI"
in the abstract, but "why would a foundation-model path ever beat a linear
extrapolator here, and under what conditions does vision/location actually
help."

## Papers reviewed

15 papers, arXiv, downloaded to `papers/task2_channel_prediction/`.

| paper | arXiv id | cluster | modalities | fusion mechanism | backbone | reported metric | few-shot | code |
|---|---|---|---|---|---|---|---|---|
| [WiFo: Wireless Foundation Model for Channel Prediction](../papers/task2_channel_prediction/WiFo%20Wireless%20Foundation%20Model%20for%20Channel%20Prediction.pdf) | 2412.08908 | foundation model | CSI only (space-time-freq) | N/A (single modality) | MAE-style transformer, masked pretraining | NMSE, multi-scenario | yes (few/zero-shot) | [liuboxun/WiFo](https://github.com/liuboxun/WiFo) |
| [WiFo-2: a generalist foundation model unifies heterogeneous wireless system design](../papers/task2_channel_prediction/WiFo-2%20a%20generalist%20foundation%20model%20unifies%20heterogeneous%20wireless%20system%20design.pdf) | 2511.22222 | foundation model | CSI-centric, heterogeneous configs | N/A (single modality, cross-config) | transformer, scaled STF tokenizer | task-specific (multiple downstream) | yes | [PKU-PCNI/WiFo-2](https://github.com/PKU-PCNI/WiFo-2) |
| [Tiny-WiFo: Lightweight WFM via Multi-Component Adaptive Knowledge Distillation](../papers/task2_channel_prediction/Tiny-WiFo%20A%20Lightweight%20Wireless%20Foundation%20Model%20for%20Channel%20Prediction%20via%20Multi-Component%20Adaptive%20Knowledge%20Distillation.pdf) | 2511.04015 | efficiency | CSI only | cross-attention-based knowledge selection (teacher→student distillation, not multi-modal fusion) | distilled transformer | NMSE vs teacher WiFo | zero-shot | not found |
| [Multimodal Wireless Foundation Models](../papers/task2_channel_prediction/Multimodal%20Wireless%20Foundation%20Models.pdf) | 2511.15162 | foundation model, multimodal | multiple wireless-signal modalities (task/condition-dependent) | masked-autoencoder-style joint pretraining, modality selection at inference | transformer | not directly comparable | not stated | not found |
| [WiFo-MiSAC: WFM for Multimodal Sensing and Communication Integration via SoM](../papers/task2_channel_prediction/WiFo-MiSAC%20A%20Wireless%20Foundation%20Model%20for%20Multimodal%20Sensing%20and%20Communication%20Integration%20via%20Synesthesia%20of%20Machines.pdf) | 2604.18255 | SoM foundation model | heterogeneous comms + sensing signals tokenized into unified space | shared-specific disentangled Mixture-of-Experts (SS-DMoE), early + late fusion paths | transformer + MoE | not directly comparable | few-shot | not found |
| [A Multi-Modal Foundational Model for Wireless Communication and Sensing](../papers/task2_channel_prediction/A%20Multi-Modal%20Foundational%20Model%20for%20Wireless%20Communication%20and%20Sensing.pdf) | 2602.04016 | foundation model, multimodal | multiple PHY sensing modalities | mixture-of-experts transformer, task-agnostic | transformer + MoE | claimed generalization across scenarios | not stated | not found |
| [Environment-Aware Channel Inference via Cross-Modal Flow](../papers/task2_channel_prediction/Environment-Aware%20Channel%20Inference%20via%20Cross-Modal%20Flow%20From%20Multimodal%20Sensing%20to%20Wireless%20Channels.pdf) | 2512.04966 | multimodal fusion, pilot-free | environmental sensing → CSI | conditional diffusion / flow model conditioned on sensing embedding | flow-matching / diffusion | NMSE, pilot-free channel inference | not stated | [gm-leung](https://github.com/gm-leung/) |
| [Vision-Aided Channel Prediction via Image Segmentation (Street Intersection)](../papers/task2_channel_prediction/Vision-Aided%20Channel%20Prediction%20Based%20on%20Image%20Segmentation%20at%20Street%20Intersection%20Scenarios.pdf) | 2501.15726 | vision-aided | RGB (segmented) + channel history | segmentation-derived environment features concatenated with channel features | not stated (lightweight CNN/regressor) | not directly comparable | not stated | not found |
| [Vision Aided Channel Prediction for Vehicular Comms (RGB → received power)](../papers/task2_channel_prediction/Vision%20Aided%20Channel%20Prediction%20for%20Vehicular%20Communications%20A%20Case%20Study%20of%20Received%20Power%20Prediction%20Using%20RGB%20Images.pdf) | 2501.18618 | vision-aided | RGB only → received power (proxy for channel quality) | CNN feature extraction → regression head | CNN | not directly comparable (predicts power, not full CSI/NMSE) | not stated | uses off-the-shelf detection tools (ultralytics, labelme), not the model itself |
| [Cross-Environment Transfer Learning for Location-Aided Beam Prediction](../papers/task2_channel_prediction/Cross-Environment%20Transfer%20Learning%20for%20Location-Aided%20Beam%20Prediction%20in%205G%20and%20Beyond%20Millimeter-Wave%20Networks.pdf) | 2503.14287 | location-aided, transfer | UE location → beam index | location features → classifier, pretrain in one environment / fine-tune in another | not stated (compact NN) | beam prediction accuracy | yes (cross-environment fine-tuning) | not found |
| [Resource-Efficient Beam Prediction with Multimodal Realistic Simulation](../papers/task2_channel_prediction/Resource-Efficient%20Beam%20Prediction%20in%20mmWave%20Communications%20with%20Multimodal%20Realistic%20Simulation%20Framework.pdf) | 2504.05187 | multimodal, efficiency | RGB + other sensing modalities → beam index | multimodal sensing fusion, efficiency-focused architecture | not stated (compact NN) | beam prediction accuracy vs compute cost | not stated | not found |
| [LLM4CP: Adapting LLMs for Channel Prediction](../papers/task2_channel_prediction/LLM4CP%20Adapting%20Large%20Language%20Models%20for%20Channel%20Prediction.pdf) | 2406.14440 | few-shot / cross-domain transfer | CSI only (time-series), LLM backbone | frozen pretrained LLM + tailored preprocessor/embedding/output adapters | pretrained LLM (frozen backbone, small trainable adapters) | NMSE, "SOTA on full-sample, few-shot, and generalization tests" | yes (explicit few-shot evaluation) | [liuboxun/LLM4CP](https://github.com/liuboxun/LLM4CP) |
| [Physics Equivariance for Robust Generalization in Wireless Foundation Models](../papers/task2_channel_prediction/Physics%20Equivariance%20for%20Robust%20Generalization%20in%20Wireless%20Foundation%20Model.pdf) | 2606.28847 | robustness / physics-informed | CSI only | physics-equivariance constraints baked into masked-autoencoder pretraining | transformer (MAE-style) | not directly comparable | not stated | not found |
| [M3F-UAV: Missing-Modality Multimodal Foundation Model for Low-Altitude Wireless Sensing](../papers/task2_channel_prediction/M3F-UAV%20A%20Missing-Modality%20Multimodal%20Foundation%20Model%20for%20Low-Altitude%20Wireless%20Sensing.pdf) | 2607.13678 | missing-modality robustness | vision + geometric/wireless sensing (occlusion/sensor-failure setting) | mixture-of-experts with modality-dropout-robust routing | transformer + MoE | not directly comparable | not stated | not found |
| [SynthSoM: a synthetic multi-modal sensing-communication dataset for SoM](../papers/task2_channel_prediction/SynthSoM%20A%20synthetic%20intelligent%20multi-modal%20sensing-communication%20dataset%20for%20Synesthesia%20of%20Machines.pdf) | 2501.07459 | dataset | RGB, radar, CSI, geometry (simulated: AirSim + WaveFarer + Wireless InSite) | N/A (dataset paper) | N/A | N/A | dataset+code, [ZiweiHuang96/SynthSoM](https://github.com/ZiweiHuang96/SynthSoM) |

## Synthesis by cluster

**Foundation models (WiFo/WiFo-2/Tiny-WiFo/Physics-Equivariance).** All are
CSI-only, pretrained with a masked-autoencoder-style objective, then
fine-tuned per task. None of them fuse RGB or location — the multi-modal
angle is entirely absent from the WiFo lineage itself. Tiny-WiFo and the
physics-equivariance paper both target exactly the axes this challenge
scores on (FLOPs/params, robustness) but purely on the CSI side. Established:
masked pretraining + light fine-tuning transfers across CSI configurations.
Missing: none of them show CSI-only pretraining beating a plain linear
extrapolator on a *short-horizon, structured* prediction task like ours —
which matches what we measured locally.

**Multimodal / SoM foundation models (WiFo-MiSAC, "Multi-Modal Foundational
Model", Multimodal WFMs).** These are the closest match to the challenge's
framing (shared backbone, multiple modalities, MoE-based sharing). Common
mechanism: mixture-of-experts with a shared-vs-modality-specific split,
letting one backbone serve several tasks/modalities — structurally similar
to what the >80%-shared-weight constraint is asking for. Established: MoE
routing is the field's answer to "one backbone, several modalities/tasks."
Missing: none report a channel-prediction NMSE we can directly compare
against our -29 to -31 dB; they evaluate on their own multi-scenario
benchmarks, not this challenge's data.

**Vision/location-aided (image segmentation, RGB→power, location→beam,
resource-efficient beam prediction).** Established: environment features
(segmented scene, location) help *classification/selection* tasks (beam
index, coarse power level) where the mapping from environment to outcome is
geometric and low-dimensional. Missing: none of these predict full
subcarrier-resolution CSI from vision; the RGB signal is consistently used
for a coarser output than what Task 2 asks for. This is a real gap between
"vision helps in the literature" and "vision helps for *this exact*
input→output shape."

**Cross-modal flow / diffusion for channel inference.** The most directly
relevant single paper (Environment-Aware Channel Inference via Cross-Modal
Flow) — pilot-free CSI inference conditioned on environmental sensing, with
code available. Established: generative conditioning (flow/diffusion) is an
alternative to regression-style fusion for sensing→CSI. Missing: not
benchmarked in a few-shot, >80%-shared-backbone, RGB+location+partial-CSI
setting like ours; and it's a heavier model class, cutting against the
FLOPs/param scoring axis.

**Missing-modality robustness (M3F-UAV).** Established: MoE-based routing
that degrades gracefully when a modality is absent/corrupted, evaluated in
occlusion/sensor-failure conditions structurally similar to this challenge's
"noisy/missing RGB or CSI" requirement. Missing: not evaluated on
channel-prediction-style regression, and not on the +80% shared-weight
multi-task constraint specifically.

**Few-shot/cross-domain transfer (LLM4CP, cross-environment beam transfer).**
Established: freezing a large pretrained backbone and fine-tuning small
adapters transfers well in low-data regimes for CSI-shaped time series —
directly relevant to the few-shot requirement. Missing: LLM4CP is CSI-only
(no RGB/location fusion), so it doesn't address the multi-modal half.

## Gap analysis tied to this challenge's constraints

- **>80% shared-backbone constraint.** WiFo-MiSAC and the MoE-based
  multimodal foundation models are the only reviewed works structurally
  aligned with this (shared trunk + small modality/task-specific heads).
  Our current `task2_wifo2`/`task2_physics` experiments already share the
  WiFo2 backbone with Task 1 (same `weights/model_best.pkl`), so this
  constraint is *already satisfied structurally* — the open question is
  whether the shared backbone can be made to actually help Task 2 rather
  than being carried passively while a separate linear/ridge model does the
  real work (currently the case: `alpha≈0.002` in `task2_physics`).

- **Noisy/missing-modality robustness.** M3F-UAV and the MoE-disentangled
  papers address this directly for vision+sensing; nothing we've tried
  locally has tested Task 2 under corrupted/missing RGB or location yet —
  this is an **open gap**, not something covered by `experiments/task2_*`.

- **Param/FLOPs efficiency scoring.** Tiny-WiFo (knowledge distillation) and
  the physics-equivariance paper are the direct efficiency-oriented
  analogues. Our best Task 2 result is a **linear ridge model** — already
  far cheaper in params/FLOPs than any transformer path in the literature,
  and 20 dB better on NMSE. The efficiency literature is solving a problem
  (shrinking a big transformer) that a much smaller model already beats
  outright here. This is worth stating explicitly as a finding, not just a
  gap: the challenge's own efficiency scoring rewards exactly the direction
  our simplest model already points in, provided it stays "foundation-model
  adjacent" enough to satisfy the >80%-shared-weight rule.

- **Whether vision/location help at all.** This is the biggest open
  question the literature doesn't resolve for us: vision-aided papers
  operate on coarser outputs (beam index, power level, not per-subcarrier
  CSI), so "vision helps" in the literature doesn't yet transfer as evidence
  for this exact task shape. Locally, RGB+position moved NMSE by <0.02 dB
  over CSI-only. Untested locally: RGB/location under the *few-shot* subset
  size actually used for grading (500 train samples is not necessarily the
  regime that ships), and whether RGB helps specifically on samples where
  the leading-subcarrier CSI is least informative (e.g. low correlation
  between first-`Nc` and remaining subcarriers) rather than on average.

## Recommendations

1. **Diagnose before adding fusion complexity.** Before trying any paper's
   fusion mechanism, check *why* RGB+location contributed ~0 dB in
   `task2_physics` — e.g. does an oracle model with RGB/location as the only
   input (no CSI) beat persistence at all? If not, no fusion architecture
   from this literature will help; the modalities may simply not carry
   independent signal for this dataset. (Distinct from the augmentation
   finding in Task 1 — a different kind of "this input doesn't help.")

2. **Try conditioning at the linear model, not swapping it for a
   transformer.** Given the linear ridge already wins by ~20 dB, the
   MoE-style "shared trunk + tiny modality-specific head" pattern
   (WiFo-MiSAC, M3F-UAV) suggests testing a *small* gating/conditioning term
   on top of the ridge (e.g. RGB/location-conditioned per-cell ridge weights
   rather than a global `alpha`) instead of routing more capacity through
   WiFo2's transformer path, which has consistently underperformed here.

3. **Test missing-modality robustness explicitly**, since it's untested
   locally and directly scored: rerun the best Task 2 config (ridge, and
   ridge+overlay) with RGB and/or location zeroed/noised at inference, and
   check whether the >80%-shared-backbone submission degrades gracefully or
   catastrophically — this is a concrete, cheap experiment none of our
   `experiments/task2_*` runs have covered.

4. **If pursuing the foundation-model path further, borrow LLM4CP's adapter
   pattern (freeze backbone, small trainable input/output adapters) rather
   than WiFo2's own fine-tuning path**, since it's the one reviewed method
   demonstrating explicit few-shot gains on CSI-shaped data with a large
   pretrained backbone — closer to what `task2_unfreeze`'s sweep was
   probing, but with adapters instead of unfreezing encoder blocks (which
   only got to -8.2 dB, nowhere near linear).

5. **Consider the cross-modal flow/diffusion result (2512.04966) only as a
   ceiling reference, not a first move** — it's the one paper solving
   sensing→CSI inference directly and has code, but it's a heavier model
   class that cuts against the FLOPs scoring axis; worth reading for the
   conditioning mechanism, not for wholesale adoption.
