# CLARYON Model Guide

A practical guide for nuclear medicine researchers using CLARYON's classical and quantum machine learning models.

---

## 1. Choosing Complexity

Every model in CLARYON can be run at one of five complexity levels. Higher complexity means longer training but potentially better results. The complexity level controls hyperparameters like the number of training epochs, learning rate, and batch size.

| Preset       | Purpose                              | Typical Runtime | When to Use                              |
|--------------|--------------------------------------|-----------------|------------------------------------------|
| `quick`      | Smoke test, verify pipeline works    | Seconds         | Debugging, testing config files          |
| `small`      | Fast exploratory analysis            | Minutes         | Initial data exploration                 |
| `medium`     | Balanced quality and speed (default) | 10-30 min       | Most experiments, draft results          |
| `large`      | High quality for publication         | 1-3 hours       | Final results, preparing manuscripts     |
| `exhaustive` | Maximum quality, no time constraint  | Hours to days   | Benchmark studies, definitive comparisons|

You can set complexity globally for all models:

```yaml
experiment:
  complexity: medium
```

Or override per model using the `preset` field:

```yaml
models:
  - name: xgboost
    preset: large
  - name: kernel_svm
    preset: small
```

Explicit parameters in a model's `params` block always take priority over presets. This means you can use a preset as a starting point and override individual parameters as needed.

**Resolution priority** (highest wins):
1. Explicit `params` in your YAML config
2. Model-level `preset`
3. Global `complexity` setting
4. Category default for `medium`

> **Important**: Never publish results generated with `quick` or `small` presets. These are designed for testing and exploration only. CLARYON will print a warning if you use these presets with quantum models.

---

## 2. Auto Mode

Setting `complexity: auto` tells CLARYON to automatically choose the best preset for each model based on your dataset size, number of features, and time budget.

```yaml
experiment:
  complexity: auto
  max_runtime_minutes: 120
```

### How it works

After preprocessing (once the final feature count is known), CLARYON estimates the runtime for each model at every preset level. It then selects the highest-quality preset that fits within your time budget, distributed evenly across models, folds, and seeds.

Classical models (XGBoost, LightGBM, etc.) nearly always get `large` or `exhaustive` because they train in seconds. Quantum models, which are computationally more expensive, may be assigned lower presets to stay within budget.

The resolved configuration is saved to `Results/<experiment>/auto_resolved_config.yaml` so you can see exactly what was chosen.

### When to trust auto mode

- You have a reasonable time budget (60+ minutes)
- Your dataset has fewer than 1000 samples and fewer than 32 features
- You want a quick comparison across many models without tuning each one

### When to override

- You need reproducible, publication-quality results (set `large` or `exhaustive` explicitly)
- You have a specific model you care about most (give it a higher preset manually)
- Your dataset is very large or very small (auto mode's estimates may be inaccurate at extremes)

---

## 3. Understanding Quantum Models

### angle_pqk_svm (Angle-Encoded Projected Quantum Kernel SVM)

The recommended quantum model for tabular radiomic data. Each feature is encoded into a dedicated qubit via an angle rotation RY(bandwidth * x_i), then single-qubit Pauli observables (X, Y, Z) are measured on every qubit to produce a classical feature vector. An RBF kernel is computed on these "Pauli vectors" and fed to a standard SVM.

Unlike amplitude-encoded models, angle encoding does not L2-normalize the data, so per-feature magnitude information is preserved. This is why `angle_pqk_svm` closes approximately 90% of the quantum-classical performance gap on radiomic datasets. It also uses only O(N) circuit evaluations (one per sample), making it substantially faster than O(N^2) fidelity kernel models.

Key hyperparameter: `bandwidth` (default 0.5) controls how strongly features rotate the qubits. Lower bandwidth preserves finer distinctions.

Uses `type: tabular` in config (z-score IS applied; the model handles encoding internally).

### kernel_svm (Quantum Kernel SVM)

Computes similarity between patients using a quantum fidelity kernel: the overlap between two amplitude-encoded quantum states. The resulting kernel matrix is fed to a standard SVM classifier. Uses amplitude encoding, so features are L2-normalized. Runtime scales with the square of the number of samples (O(N^2) kernel matrix). Available for comparative evaluation and NIfTI imaging workflows.

### projected_kernel_svm (Projected Quantum Kernel SVM)

Same Pauli measurement approach as `angle_pqk_svm`, but uses amplitude encoding instead of angle encoding. Proved empirically that amplitude encoding (not the kernel measurement) is the bottleneck for quantum performance on tabular data: `projected_kernel_svm` achieved the same BACC as the fidelity `kernel_svm`. Available for comparative evaluation.

### qcnn_muw (Quantum Convolutional Neural Network --- Moradi et al. Architecture)

Inspired by classical convolutional neural networks, this model applies layers of quantum operations that progressively reduce the quantum state to a single prediction. It uses a circuit design from Moradi et al. developed for nuclear medicine imaging biomarker classification.

Trains iteratively (like a neural network) with amplitude encoding. Needs >= 100 epochs to converge. For tabular data, consider `angle_pqk_svm` as a faster alternative.

### qcnn_alt (Alternative QCNN Architecture)

An alternative quantum convolutional neural network with a different circuit topology. Performance differences between the two QCNN architectures are dataset-dependent, so comparing both is recommended.

### qdc_hadamard (Quantum Distance Classifier --- Hadamard Test)

Classifies patients by measuring the quantum distance between a new patient and the average quantum state of each class. Uses the Hadamard test with an ancilla qubit. Fast and deterministic (no training loop), making it a good baseline quantum model. Uses amplitude encoding.

### quantum_gp (Quantum Gaussian Process)

A Gaussian Process classifier with a quantum (amplitude-encoded) fidelity kernel. Provides calibrated uncertainty estimates in addition to predictions. The most robust amplitude-encoded quantum model on real tabular data. Runtime scales quadratically with sample count.

### qnn (Quantum Neural Network)

A quantum neural network trained with a contrastive margin-based loss function. Uses amplitude encoding. Best suited for small datasets (under 200 samples) with complex decision boundaries.

---

## 4. Parameter Reference

### Classical Tabular Models

#### XGBoost

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `n_estimators`  | 500      | Number of boosting rounds. More = better but slower.  |
| `max_depth`     | 8        | Maximum tree depth. Higher captures more interactions. |
| `learning_rate` | 0.02     | Step size. Lower = more stable but needs more rounds.  |
| `subsample`     | 0.8      | Fraction of samples per tree. Lower reduces overfitting.|
| `colsample_bytree` | 0.8  | Fraction of features per tree.                         |

#### LightGBM

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `n_estimators`  | 500      | Number of boosting rounds.                            |
| `max_depth`     | 8        | Maximum tree depth (-1 for no limit).                 |
| `learning_rate` | 0.02     | Step size per boosting round.                         |
| `num_leaves`    | 31       | Maximum leaves per tree. Controls model complexity.   |

#### CatBoost

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `iterations`    | 500      | Number of boosting rounds.                            |
| `depth`         | 8        | Maximum tree depth.                                   |
| `learning_rate` | 0.02     | Step size per boosting round.                         |
| `verbose`       | 0        | Set to 0 to suppress CatBoost output.                |

#### MLP (Multi-Layer Perceptron)

| Parameter        | Default  | What It Controls                                     |
|------------------|----------|------------------------------------------------------|
| `hidden_layers`  | [64, 32] | Sizes of hidden layers.                             |
| `epochs`         | 100      | Number of training passes over the data.            |
| `lr`             | 0.01     | Learning rate for optimizer.                         |
| `batch_size`     | 16       | Samples per training batch.                          |
| `dropout`        | 0.1      | Dropout rate for regularization.                     |

#### TabPFN

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `N_ensemble_configurations` | 16 | Number of ensemble members. More = better but slower. |

### Imaging Models

#### CNN 2D / CNN 3D

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `epochs`        | 50       | Number of training passes over the data.              |
| `batch_size`    | 4        | Images per training batch (limited by GPU memory).    |
| `lr`            | 0.001    | Learning rate for optimizer.                          |
| `weight_decay`  | 1e-4     | L2 regularization strength.                          |

### Quantum Models

#### angle_pqk_svm

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `bandwidth`     | 0.5      | Feature scaling before angle rotation. Lower = finer distinctions. |
| `gamma`         | `"auto"` | RBF kernel bandwidth. `"auto"` uses 1/(d * var(V)).  |
| `C`             | 1.0      | SVM regularization parameter.                         |

#### kernel_svm / projected_kernel_svm

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `shots`         | `null`   | Number of measurement shots. `null` = exact simulation.|
| `gamma`         | `"auto"` | RBF bandwidth (projected_kernel_svm only).            |

#### qcnn_muw / qcnn_alt

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `epochs`        | 100      | Number of training iterations.                        |
| `lr`            | 0.01     | Learning rate for parameter optimization.             |
| `init_scale`    | 0.1      | Scale of random initial circuit parameters.           |
| `batch_size`    | 16       | Samples per training batch.                           |

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `num_layers`    | 4        | Number of variational circuit layers. More = more expressive. |
| `epochs`        | 100      | Number of training iterations.                        |
| `lr`            | 0.01     | Learning rate.                                        |
| `batch_size`    | 16       | Samples per training batch.                           |

| Parameter         | Default  | What It Controls                                    |
|-------------------|----------|-----------------------------------------------------|
| `classical_layers`| [32, 16] | Classical network layer sizes before quantum circuit.|
| `num_layers`      | 2        | Quantum circuit layers.                              |
| `epochs`          | 100      | Number of training iterations.                       |
| `lr`              | 0.01     | Learning rate.                                       |

#### qdc_hadamard

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `shots`         | `null`   | Measurement shots. `null` = exact simulation.         |

> Note: QDC Hadamard has no training loop. It computes distances directly from encoded data.

#### quantum_gp

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `shots`         | `null`   | Measurement shots for kernel evaluation.              |
| `noise_level`   | 1e-4     | Regularization noise added to kernel diagonal.        |

#### qnn

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `num_layers`    | 4        | Number of quantum circuit layers.                     |
| `epochs`        | 100      | Number of training iterations.                        |
| `lr`            | 0.005    | Learning rate.                                        |
| `margin`        | 0.15     | Contrastive loss margin. Higher pushes classes further apart. |
| `batch_size`    | 16       | Samples per training batch.                           |

---

## 5. Runtime Expectations

Estimated per-fold runtimes on a modern CPU workstation (no GPU, exact quantum simulation). Quantum model runtimes depend heavily on qubit count, which equals ceil(log2(features)).

### Classical Models

| Model     | 100 samples | 500 samples | 1000 samples |
|-----------|-------------|-------------|--------------|
| XGBoost   | < 1s        | 1-2s        | 2-5s         |
| LightGBM  | < 1s        | 1-2s        | 2-3s         |
| CatBoost  | 1-2s        | 3-5s        | 5-10s        |
| MLP       | 2-5s        | 5-15s       | 15-30s       |
| TabPFN    | 1-3s        | 3-10s       | 10-30s       |

### Quantum Models — Angle-Encoded (8 features = 8 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| angle_pqk_svm  | < 5s        | 5-10s       | 10-20s       |

### Quantum Models — Amplitude-Encoded (8 features = 3 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| kernel_svm     | 5s          | 2 min       | 8 min        |
| qcnn_muw       | 15s         | 1 min       | 2 min        |
| qcnn_alt       | 15s         | 1 min       | 2 min        |
| qdc_hadamard   | 3s          | 1 min       | 5 min        |
| quantum_gp     | 5s          | 2 min       | 8 min        |
| qnn            | 20s         | 1.5 min     | 3 min        |

### Quantum Models — Amplitude-Encoded (16 features = 4 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| kernel_svm     | 10s         | 4 min       | 16 min       |
| qcnn_muw       | 30s         | 2 min       | 5 min        |
| qcnn_alt       | 30s         | 2 min       | 5 min        |
| qdc_hadamard   | 8s          | 3 min       | 12 min       |
| quantum_gp     | 10s         | 4 min       | 16 min       |
| qnn            | 40s         | 3 min       | 7 min        |

> **Tip**: `angle_pqk_svm` is O(N) — one circuit per sample. Amplitude-encoded kernel models (`kernel_svm`, `quantum_gp`, `qdc_hadamard`) are O(N²) — one circuit per sample pair.

---

## Important: Preprocessing and Quantum Models

**Amplitude-encoded models** (`type: tabular_quantum`): z-score normalization is automatically skipped. Amplitude encoding L2-normalizes the feature vector, and prior z-score distorts quantum kernel geometry by 30-40%.

**Angle-encoded models** (`angle_pqk_svm`, `type: tabular`): z-score normalization IS applied. Angle encoding benefits from normalized feature scales because each feature independently controls a qubit rotation.

CLARYON handles this automatically based on the model's `type` field. If you override preprocessing manually, be aware of the distinction.

---

## 6. Common Pitfalls

### Do not publish with quick or small presets

The `quick` and `small` presets use very few training epochs and aggressive shortcuts. They are designed for testing your pipeline configuration, not for generating scientific results. Always use `medium` or higher for any results you plan to report.

### Reduce features before quantum models

Quantum simulation cost grows exponentially with qubit count. Each qubit doubles the memory and compute time. Use mRMR feature selection (`preprocessing.feature_selection.method: mrmr`) to reduce your feature set to 16 or fewer features before running quantum models.

### Kernel models scale quadratically with samples

Amplitude-encoded models that compute a kernel matrix (`kernel_svm`, `qdc_hadamard`, `quantum_gp`) must compare every pair of samples. Doubling your sample count quadruples the runtime. For datasets with more than 500 samples, prefer `angle_pqk_svm` (O(N) circuits) or training-based models like `qcnn_muw`.

### Set a realistic time budget for auto mode

If you set `max_runtime_minutes: 10` with 10 quantum models, auto mode will assign `quick` to most of them. Give enough time for meaningful training --- at least 60 minutes for a small dataset, 120+ minutes for moderate ones.

### Seed your experiments

Always set `experiment.seed` and `cv.seeds` explicitly for reproducibility. Different seeds can produce noticeably different results with quantum models due to the stochastic nature of parameter initialization.

---

## 7. When to Use Quantum Models

Quantum machine learning on classical simulators is computationally expensive and does not provide a guaranteed advantage over classical methods. Here is honest guidance for when quantum models are most likely to add value in your research:

**Best case for quantum models:**
- Features: 16 or fewer (after feature selection)
- Samples: 500 or fewer
- Decision boundary: highly nonlinear
- You want to compare quantum and classical approaches on your data

**When classical models are likely sufficient:**
- Your dataset is large (1000+ samples) --- classical models will outperform or match quantum ones
- Features are linearly separable --- a simple SVM or logistic regression will work
- You need fast iteration --- classical models train in seconds

**Recommended workflow:**
1. Run classical models first with `medium` or `large` preset
2. Add `angle_pqk_svm` (tabular) --- it runs in seconds and gives the best quantum performance on radiomic data
3. Optionally add amplitude-encoded quantum models for comparative evaluation
4. Compare with appropriate statistical tests across cross-validation folds

**On the GDQ score**: The geometric difference score (Huang et al., 2021) evaluates whether a fidelity (amplitude-encoded) quantum kernel captures structure inaccessible to classical kernels. It does not apply to projected quantum kernels (`angle_pqk_svm`) or training-based models (`qnn`, `qcnn_muw`). Use it for research into quantum kernel properties, not as a prerequisite for running quantum models.
