# CLARYON Contributor Guide

How to set up, test, and contribute to CLARYON.

---

## 1. Development Setup

### Clone and install in editable mode

```bash
git clone https://github.com/lpapp-muw/CLARYON.git
cd claryon
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

The `[all]` extra installs all optional dependencies (quantum, imaging, explainability). The `[dev]` extra installs testing and linting tools.

### Verify your setup

```bash
python -m pytest tests/ -x -q --timeout=300
claryon list-models
```

All tests should pass and `list-models` should show all registered models.

---

## 2. Running Tests

### Full test suite

```bash
python -m pytest tests/ -q --timeout=300
```

### Run a specific test file

```bash
python -m pytest tests/test_preprocessing/test_zscore.py -v
```

### Run tests matching a keyword

```bash
python -m pytest tests/ -k "feature_selection" -v
```

### Stop on first failure

```bash
python -m pytest tests/ -x -q
```

### Test organization

```
tests/
  test_preprocessing/
    test_zscore.py
    test_binary_grouping.py
    test_feature_selection.py
    test_state.py
    test_image_norm.py
  test_models/
    test_presets.py
    test_auto_complexity.py
  test_evaluation/
    test_geometric_difference.py
  test_integration/
    test_inference.py
  test_safety.py
  test_verification.py
```

### Writing tests

- Use `pytest` (not unittest)
- Use small synthetic datasets (iris, random data) to keep tests fast
- Set fixed random seeds for reproducibility
- Mock external dependencies when testing in isolation
- Tests should complete in under 30 seconds each

---

## 3. Adding a New Model

Follow this template to add a model to CLARYON.

### Step 1: Create the model builder

Create a file in the appropriate subdirectory under `claryon/models/`. For a classical tabular model:

```python
# claryon/models/classical/my_model.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from claryon.models.base import ModelBuilder, register

logger = logging.getLogger(__name__)


@register("my_model", category="tabular")
class MyModelBuilder(ModelBuilder):
    """Brief description of the model."""

    def build(self, params: Dict[str, Any]) -> None:
        """Initialize model with parameters.

        Args:
            params: Resolved parameter dictionary (after preset resolution).
        """
        from some_library import SomeClassifier

        self.n_estimators = params.get("n_estimators", 100)
        self.max_depth = params.get("max_depth", 6)
        self.model = SomeClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=params.get("seed", 42),
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            task_type: One of "binary", "multiclass".
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X_test: Test features, shape (n_samples, n_features).

        Returns:
            Predicted class labels, shape (n_samples,).
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Args:
            X_test: Test features, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X_test)

    def save(self, path: str) -> None:
        """Save model to directory.

        Args:
            path: Directory path.
        """
        joblib.dump(self.model, Path(path) / "model.joblib")

    def load(self, path: str) -> None:
        """Load model from directory.

        Args:
            path: Directory path containing saved model.
        """
        self.model = joblib.load(Path(path) / "model.joblib")
```

### Step 2: Register the import

Add an import to `claryon/models/__init__.py` (or the appropriate sub-package `__init__.py`) so the `@register` decorator runs at import time:

```python
from claryon.models.classical import my_model  # noqa: F401
```

### Step 3: Add presets

Add entries to `claryon/models/presets.yaml`:

```yaml
my_model:
  quick:      { n_estimators: 10, max_depth: 3 }
  small:      { n_estimators: 50, max_depth: 4 }
  medium:     { n_estimators: 200, max_depth: 6 }
  large:      { n_estimators: 500, max_depth: 8 }
  exhaustive: { n_estimators: 1000, max_depth: 10 }
```

### Step 4: Add a method description

Add an entry to `claryon/reporting/method_descriptions.yaml`:

```yaml
my_model:
  short: "My Model (SomeLibrary)"
  description: >
    A brief description of the model suitable for a methods section.
    Include the key algorithmic approach and any relevant citations.
  citation: "Author2024"
```

### Step 5: Write tests

```python
# tests/test_models/test_my_model.py
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris


def test_my_model_fit_predict():
    """Test basic fit/predict cycle."""
    from claryon.models.classical.my_model import MyModelBuilder

    X, y = load_iris(return_X_y=True)
    # Convert to binary
    y = (y > 0).astype(int)

    builder = MyModelBuilder()
    builder.build({"n_estimators": 10, "max_depth": 3, "seed": 42})
    builder.fit(X[:120], y[:120], "binary")
    preds = builder.predict(X[120:])

    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1})


def test_my_model_save_load(tmp_path):
    """Test model save and load round-trip."""
    from claryon.models.classical.my_model import MyModelBuilder

    X, y = load_iris(return_X_y=True)
    y = (y > 0).astype(int)

    builder = MyModelBuilder()
    builder.build({"n_estimators": 10, "seed": 42})
    builder.fit(X, y, "binary")
    preds_before = builder.predict(X)

    builder.save(str(tmp_path))

    builder2 = MyModelBuilder()
    builder2.load(str(tmp_path))
    preds_after = builder2.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)
```

### Step 6: Test end-to-end

Create a test config that includes your model and run:

```bash
claryon -v run -c configs/test_my_model.yaml
```

---

## 4. Adding a New Metric

### Step 1: Implement the metric function

Add to `claryon/evaluation/metrics.py`:

```python
def matthews_corrcoef(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> float:
    """Compute Matthews Correlation Coefficient.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (unused for MCC).

    Returns:
        MCC score in [-1, 1].
    """
    from sklearn.metrics import matthews_corrcoef as sklearn_mcc
    return float(sklearn_mcc(y_true, y_pred))
```

### Step 2: Register it

In the same file, add to the registry dictionary:

```python
METRIC_REGISTRY["matthews_corrcoef"] = matthews_corrcoef
```

### Step 3: Test

```python
def test_matthews_corrcoef():
    from claryon.evaluation.metrics import METRIC_REGISTRY
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    score = METRIC_REGISTRY["matthews_corrcoef"](y_true, y_pred)
    assert score == 1.0
```

The metric is now available in config files under `evaluation.metrics`.

---

## 5. PR Checklist

Before submitting a pull request, verify the following:

- [ ] All existing tests pass: `python -m pytest tests/ -x -q --timeout=300`
- [ ] New code has tests with reasonable coverage
- [ ] New models implement all `ModelBuilder` abstract methods
- [ ] New models have preset entries in `presets.yaml`
- [ ] New models have method descriptions in `method_descriptions.yaml`
- [ ] All files have `from __future__ import annotations` as the first import
- [ ] All public functions have type hints and Google-style docstrings
- [ ] No `print()` statements --- use `logging.getLogger(__name__)` instead
- [ ] Predictions are written through `io/predictions.py` with semicolon separator
- [ ] Random operations use the seed from config for reproducibility
- [ ] No hardcoded file paths
- [ ] Resource-intensive operations have appropriate warnings via `safety.py`
- [ ] Integration test passes: `claryon -v run -c configs/iris_full_preprocess.yaml`

---

## 6. Code Style Rules

### Imports

Every Python file must start with:

```python
from __future__ import annotations
```

This enables PEP 604 union types (`X | Y`) and forward references everywhere.

### Type hints

All function signatures must have type hints:

```python
def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ...
```

Use `from __future__ import annotations` to enable modern syntax (`list[str]` instead of `List[str]`, `dict[str, Any]` instead of `Dict[str, Any]`).

### Docstrings

Use Google-style docstrings:

```python
def amplitude_encode(features: np.ndarray, pad_length: int | None = None) -> np.ndarray:
    """Encode classical features into quantum amplitude vectors.

    Normalizes the feature vector to unit norm and pads to the nearest
    power of 2 length for quantum circuit compatibility.

    Args:
        features: Input features, shape (n_samples, n_features).
        pad_length: Target length after padding. If None, uses the
            nearest power of 2 >= n_features.

    Returns:
        Amplitude vectors, shape (n_samples, pad_length), with unit
        norm along axis 1.

    Raises:
        ValueError: If features contain NaN values.
    """
```

### Logging

Never use `print()`. Always use the logging module:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Training %s on %d samples", model_name, n_samples)
logger.warning("Low sample count: %d", n_samples)
logger.error("Model %s failed: %s", model_name, error)
```

### Model registration

All model classes must use the `@register` decorator:

```python
@register("model_name", category="tabular")
class MyModelBuilder(ModelBuilder):
    ...
```

Valid categories: `tabular`, `tabular_quantum`, `imaging`.

### Deterministic seeding

All random operations must be seeded:

```python
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
```

Never rely on global random state. Pass seeds explicitly.

### CSV output

All prediction files use semicolons as separators, written through `claryon/io/predictions.py`:

```python
from claryon.io.predictions import write_predictions
write_predictions(pred_dir, y_true, y_pred, y_prob)
```

### Error handling

Never let a model crash the entire pipeline. Catch exceptions and continue:

```python
try:
    model.fit(X_train, y_train, task_type)
except MemoryError:
    logger.error("OUT OF MEMORY during %s training. Skipping.", model_name)
    continue
except Exception as e:
    logger.error("FAILED %s: %s", model_name, e)
    continue
```

### File organization

- One model per file
- Tests mirror the source layout (`claryon/models/foo.py` -> `tests/test_models/test_foo.py`)
- Configuration files go in `configs/`
- No business logic in `__init__.py` (imports and registration only)
