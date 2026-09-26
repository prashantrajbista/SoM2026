# SoM Challenge 2026 — Challenge Statement

## What it is

Synesthesia of Machines (SoM) Challenge 2026: "Wireless Foundation Model-Empowered
Multi-Modal Sensing and Communications." Goal is to design/adapt a **wireless
foundation model** (e.g. WiFo / WiFo-2, or a new architecture pretrained from
scratch) and use **one shared backbone** to solve three different few-shot
downstream tasks, instead of training separate task-specific models.

Constraint: training separate models per task, or using purely parametric
(non-learned) models, is prohibited — the three task-specific models must
share **>80% of their weights**.

## The three tasks (all few-shot fine-tuning from the shared backbone)

1. **Task 1 — LoS/NLoS Scenario Classification**
   - Input: 3D CSI sample (OFDM symbols × antennas × subcarriers).
   - Output: LoS or NLoS label.
   - Metric: Precision, Recall, F1.

2. **Task 2 — Multi-modal-enhanced Channel Prediction**
   - Input: RGB image, 2D UE location, CSI over the first `Nc` subcarriers
     (single-antenna UE, multi-antenna BS). Inputs may be missing/noisy.
   - Output: predicted CSI over the remaining subcarriers.
   - Metric: NMSE against ground-truth CSI.

3. **Task 3 — Multi-modal-enhanced Depth Map Estimation**
   - Input: CSI (subcarriers × TX antennas) + front-view RGB image from a
     vehicle camera. Both subject to noise and partial data loss.
   - Output: dense depth map.
   - Metric: MAE against ground-truth depth map.

## Practical conditions being tested

- Noisy CSI/RGB inputs and partial modality loss (robustness).
- Trade-off across task accuracy, average parameter count, and average FLOPs
  across all three tasks (deployability, not just accuracy).

## Data

- **Pretraining (open):** WiFo pretraining datasets, LH-CSI zero-shot splits
  (WiFo-2), SynthSoM datasets on Hugging Face. No restriction on pretraining
  dataset type/scale.
- **Downstream (gated):** per-task datasets with train / public-test /
  hidden-challenge-test splits (test splits have no ground truth). Requires
  challenge registration + dataset access password via the Submit tab.

## Benefits / stakes

- CHF 12,000 total prize pool; first prize RMB 30,000.
- Open access to WiFo-2-Tiny weights + fine-tuning baselines.
- ITU-issued certificates and recognition at the 3rd AI4COMM4AI Forum for
  winning teams.

## Evaluation Metrics

Five per-model scores, summed into a final score in `[0, 5]`.

1. **Task 1 (LoS/NLoS classification):** `score_1 = F1`, computed from
   Precision `= TP/(TP+FP)`, Recall `= TP/(TP+FN)`,
   `F1 = 2·Precision·Recall / (Precision+Recall)`.

2. **Task 2 (channel prediction):** let `NMSE` be the normalized MSE between
   predicted and ground-truth CSI over the remaining subcarriers, and
   `NMSE_m` the best achievable NMSE (in dB) under consideration.
   `score_2 = max(0, 10·log10(NMSE) / NMSE_m)`.

3. **Task 3 (depth estimation):** let `MAE` be the mean absolute error
   between predicted and ground-truth depth maps, and `MAE_m` the maximum
   allowable error (score is 0 if `MAE` exceeds it).
   `score_3 = max(0, 1 - MAE/MAE_m)`.

4. **Parameters:** let `P` be the average total parameter count across the
   three task-specific models, with predefined bounds `P_min`, `P_max`.
   `score_p = (log P_max - log P) / (log P_max - log P_min)` — fewer
   parameters score higher.

5. **FLOPs:** let `F` be the average FLOPs for one forward pass at batch
   size 1 across the three task-specific models, with predefined bounds
   `F_min`, `F_max`.
   `score_f = (log F_max - log F) / (log F_max - log F_min)` — fewer FLOPs
   score higher.

**Final score:** `Score = score_1 + score_2 + score_3 + score_p + score_f ∈ [0, 5]`.

## Relation to prior year

SoM Challenge 2025 used a fixed pretrained WiFo model fine-tuned for
LoS/NLoS classification, channel prediction, and vision-aided localization.
2026 opens up the foundation-model architecture/pretraining choice and adds
noise, partial-modality-loss, and multi-modal conditions, plus the
depth-estimation task and the >80% shared-weight constraint.
