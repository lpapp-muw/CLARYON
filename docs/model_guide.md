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

### kernel_svm (Quantum Kernel SVM)

This model computes similarity between patients using a quantum circuit instead of a traditional mathematical formula. Each patient's features are encoded into a quantum state, and the overlap between two quantum states measures how similar those patients are. This produces a "kernel matrix" that is fed into a standard Support Vector Machine classifier.

Quantum Kernel SVM works best when you have a small-to-moderate number of features (up to 16) and the relationships between features are complex and nonlinear. Because it computes pairwise similarities between all patients, runtime scales with the square of the number of samples --- it becomes slow beyond a few hundred patients.

### sq_kernel_svm (Squashed Quantum Kernel SVM)

A variant of the Quantum Kernel SVM that applies a squashing function to the quantum kernel values. This helps prevent the "exponential concentration" problem, where quantum kernels can become nearly identical for all pairs of patients as the number of qubits increases, making the model unable to distinguish between classes.

Use this instead of the standard `kernel_svm` when you have more than 8 features. The squashing function preserves meaningful differences between patients while suppressing noise. It has the same runtime characteristics as the standard quantum kernel SVM.

### qcnn_muw (Quantum Convolutional Neural Network --- Moradi-Papp Architecture)

Inspired by classical convolutional neural networks, this model applies layers of quantum operations that progressively reduce the quantum state to a single prediction. It uses a specific circuit design from Papp et al. that was developed for nuclear medicine imaging biomarker classification.

This is the recommended first-choice quantum model for tabular nuclear medicine data. It trains iteratively (like a neural network) and handles moderate feature counts well. Typically outperforms kernel-based quantum models on structured clinical data with 4-16 features.

### qcnn_alt (Alternative QCNN Architecture)

An alternative quantum convolutional neural network with a different circuit topology. It uses a different arrangement of quantum gates that may capture different types of feature interactions compared to the standard QCNN.

Try this as a second quantum model alongside `qcnn_muw` to see if the alternative architecture captures patterns that the primary one misses. Performance differences between the two architectures are dataset-dependent, so comparing both is recommended.

### vqc (Variational Quantum Classifier)

A general-purpose quantum classifier that uses a parameterized quantum circuit trained via gradient descent. The circuit consists of repeated layers of rotation gates and entangling gates, with parameters optimized to minimize classification loss.

The VQC is the most flexible quantum model in CLARYON but can be harder to train. It works well when the number of features is small (under 10) and the decision boundary is highly nonlinear. For larger feature counts, prefer `qcnn_muw` which has built-in feature reduction through its convolutional structure.

### hybrid (Hybrid Classical-Quantum Model)

Combines a classical neural network front-end with a quantum circuit back-end. The classical layers first reduce the input features to a small number of dimensions, which are then processed by a variational quantum circuit for final classification.

This is the best choice when you have many features but still want to use quantum processing. The classical front-end handles dimensionality reduction, so the quantum circuit only needs a few qubits. It bridges the gap between classical deep learning and quantum computing.

### qdc_hadamard (Quantum Distance Classifier --- Hadamard Test)

Classifies patients by measuring the quantum distance between a new patient and the average quantum state of each class. It uses the Hadamard test, a quantum subroutine that measures the overlap between two quantum states using one additional auxiliary qubit.

This model is fast and deterministic (no training loop), making it a good baseline quantum model. It works best when the two classes are well-separated in feature space. However, it reduces each class to a single average state, so it cannot capture within-class variation.

### qdc_swap (Quantum Distance Classifier --- SWAP Test)

Similar to `qdc_hadamard` but uses the SWAP test instead of the Hadamard test to measure quantum state overlap. The SWAP test requires more qubits (2n+1 instead of n+1) but can provide more accurate distance estimates for certain types of data.

Use with caution: because it doubles the qubit count, it becomes very expensive for more than 8 features. Prefer `qdc_hadamard` unless you have specific reason to believe the SWAP test will perform better. CLARYON will warn you if the memory requirements become excessive.

### quantum_gp (Quantum Gaussian Process)

A Gaussian Process classifier that uses a quantum kernel instead of a classical one. Gaussian Processes provide not just predictions but also uncertainty estimates, telling you how confident the model is for each patient.

This is the only quantum model in CLARYON that provides calibrated uncertainty estimates. Use it when knowing the model's confidence matters --- for example, when flagging uncertain cases for manual review. Like all kernel models, runtime scales quadratically with sample count.

### qnn (Quantum Neural Network)

A quantum neural network trained with a contrastive margin-based loss function. It learns to map patients into a quantum feature space where same-class patients are close together and different-class patients are far apart.

Best suited for small datasets (under 200 samples) with complex decision boundaries. The margin-based training can be more stable than standard cross-entropy loss for quantum circuits. Requires careful tuning of the margin parameter.

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

#### kernel_svm / sq_kernel_svm

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `shots`         | `null`   | Number of measurement shots. `null` = exact simulation.|
| `gamma`         | 1.0      | Feature map scaling parameter.                        |

#### qcnn_muw / qcnn_alt

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `epochs`        | 100      | Number of training iterations.                        |
| `lr`            | 0.01     | Learning rate for parameter optimization.             |
| `init_scale`    | 0.1      | Scale of random initial circuit parameters.           |
| `batch_size`    | 16       | Samples per training batch.                           |

#### vqc

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `num_layers`    | 4        | Number of variational circuit layers. More = more expressive. |
| `epochs`        | 100      | Number of training iterations.                        |
| `lr`            | 0.01     | Learning rate.                                        |
| `batch_size`    | 16       | Samples per training batch.                           |

#### hybrid

| Parameter         | Default  | What It Controls                                    |
|-------------------|----------|-----------------------------------------------------|
| `classical_layers`| [32, 16] | Classical network layer sizes before quantum circuit.|
| `num_layers`      | 2        | Quantum circuit layers.                              |
| `epochs`          | 100      | Number of training iterations.                       |
| `lr`              | 0.01     | Learning rate.                                       |

#### qdc_hadamard / qdc_swap

| Parameter       | Default  | What It Controls                                      |
|-----------------|----------|-------------------------------------------------------|
| `shots`         | `null`   | Measurement shots. `null` = exact simulation.         |

> Note: QDC models have no training loop. They compute distances directly from encoded data.

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

### Quantum Models (8 features = 3 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| kernel_svm     | 5s          | 2 min       | 8 min        |
| sq_kernel_svm  | 5s          | 2 min       | 8 min        |
| qcnn_muw       | 15s         | 1 min       | 2 min        |
| qcnn_alt       | 15s         | 1 min       | 2 min        |
| vqc            | 10s         | 45s         | 1.5 min      |
| hybrid         | 10s         | 45s         | 1.5 min      |
| qdc_hadamard   | 3s          | 1 min       | 5 min        |
| qdc_swap       | 10s         | 5 min       | 20 min       |
| quantum_gp     | 5s          | 2 min       | 8 min        |
| qnn            | 20s         | 1.5 min     | 3 min        |

### Quantum Models (16 features = 4 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| kernel_svm     | 10s         | 4 min       | 16 min       |
| sq_kernel_svm  | 10s         | 4 min       | 16 min       |
| qcnn_muw       | 30s         | 2 min       | 5 min        |
| qcnn_alt       | 30s         | 2 min       | 5 min        |
| vqc            | 20s         | 1.5 min     | 3 min        |
| hybrid         | 20s         | 1.5 min     | 3 min        |
| qdc_hadamard   | 8s          | 3 min       | 12 min       |
| qdc_swap       | 1 min       | 30 min      | 2 hours      |
| quantum_gp     | 10s         | 4 min       | 16 min       |
| qnn            | 40s         | 3 min       | 7 min        |

### Quantum Models (32 features = 5 qubits)

| Model          | 100 samples | 500 samples | 1000 samples |
|----------------|-------------|-------------|--------------|
| kernel_svm     | 20s         | 8 min       | 30 min       |
| qcnn_muw       | 1 min       | 5 min       | 10 min       |
| qdc_swap       | 5 min       | 2 hours     | 8+ hours     |

> **Warning**: With 32+ features, `qdc_swap` requires 11 qubits and becomes extremely slow. Use mRMR feature selection to reduce features before quantum models.

---

## 6. Common Pitfalls

### Do not publish with quick or small presets

The `quick` and `small` presets use very few training epochs and aggressive shortcuts. They are designed for testing your pipeline configuration, not for generating scientific results. Always use `medium` or higher for any results you plan to report.

### Reduce features before quantum models

Quantum simulation cost grows exponentially with qubit count. Each qubit doubles the memory and compute time. Use mRMR feature selection (`preprocessing.feature_selection.method: mrmr`) to reduce your feature set to 16 or fewer features before running quantum models.

### Kernel models scale quadratically with samples

Models that compute a kernel matrix (`kernel_svm`, `sq_kernel_svm`, `qdc_hadamard`, `qdc_swap`, `quantum_gp`) must compare every pair of samples. Doubling your sample count quadruples the runtime. For datasets with more than 500 samples, prefer training-based models like `qcnn_muw` or `vqc`.

### Watch memory with SWAP test

The `qdc_swap` model uses 2n+1 qubits, where n is the number of qubits for other models. With 8 features (3 qubits), SWAP test needs 7 qubits. With 16 features (4 qubits), it needs 9 qubits. Memory grows as 2^qubits, so SWAP test can easily exhaust available RAM. CLARYON will warn you and skip the model if memory is insufficient.

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
- Geometric difference score (g) > 1.0

**When classical models are likely sufficient:**
- Your dataset is large (1000+ samples) --- classical models will outperform or match quantum ones
- Features are linearly separable --- a simple SVM or logistic regression will work
- You need fast iteration --- classical models train in seconds

**When to include quantum models in your study:**
- You are specifically investigating quantum vs. classical performance
- You want to report geometric difference analysis (Huang et al. 2021)
- Your features come from quantum-compatible sources (e.g., radiomic features with known nonlinear interactions)
- You have sufficient compute budget for the additional training time

**Recommended workflow:**
1. Run classical models first with `medium` or `large` preset
2. Enable geometric difference analysis (`evaluation.geometric_difference: true`)
3. If g > 1.0, run quantum models
4. Compare with appropriate statistical tests across cross-validation folds

**Reference**: Huang, H.-Y. et al. "Power of data in quantum machine learning." Nature Communications 12, 2631 (2021). doi:10.1038/s41467-021-22539-9
