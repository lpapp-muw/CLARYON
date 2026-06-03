"""Tests for claryon.io.base — Dataset, LabelMappers."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import (
    BinaryLabelMapper,
    Dataset,
    MultiClassLabelMapper,
    RegressionTarget,
    TaskType,
)


class TestBinaryLabelMapper:
    def test_fit_transform_inverse(self):
        mapper = BinaryLabelMapper.fit(["High", "Low", "High", "Low"])
        ints = mapper.transform(["High", "Low", "High"])
        assert ints.dtype == np.int64
        assert list(ints) == [0, 1, 0]  # sorted alphabetically: High=0, Low=1
        back = mapper.inverse_transform(ints)
        assert back == ["High", "Low", "High"]

    def test_fit_numeric(self):
        mapper = BinaryLabelMapper.fit([0, 1, 1, 0])
        ints = mapper.transform([1, 0])
        assert list(ints) == [1, 0]

    def test_fit_not_binary_raises(self):
        with pytest.raises(ValueError, match="exactly 2"):
            BinaryLabelMapper.fit([0, 1, 2])

    def test_json_roundtrip(self):
        mapper = BinaryLabelMapper.fit(["cat", "dog", "cat"])
        d = mapper.to_json()
        restored = BinaryLabelMapper.from_json(d)
        assert restored.classes == mapper.classes
        np.testing.assert_array_equal(
            restored.transform(["cat", "dog"]),
            mapper.transform(["cat", "dog"]),
        )


class TestMultiClassLabelMapper:
    def test_fit_transform_inverse(self):
        mapper = MultiClassLabelMapper.fit(["a", "b", "c", "a"])
        assert mapper.n_classes == 3
        ints = mapper.transform(["c", "a", "b"])
        assert list(ints) == [2, 0, 1]
        back = mapper.inverse_transform(ints)
        assert back == ["c", "a", "b"]

    def test_fit_too_few_raises(self):
        with pytest.raises(ValueError, match="≥2"):
            MultiClassLabelMapper.fit([1, 1, 1])

    def test_json_roundtrip(self):
        mapper = MultiClassLabelMapper.fit([0, 1, 2])
        d = mapper.to_json()
        restored = MultiClassLabelMapper.from_json(d)
        assert restored.classes == mapper.classes


class TestRegressionTarget:
    def test_fit_and_stats(self):
        target = RegressionTarget.fit([1.0, 2.0, 3.0])
        assert target.mean == pytest.approx(2.0)
        assert target.std == pytest.approx(np.std([1.0, 2.0, 3.0]))

    def test_transform_inverse(self):
        target = RegressionTarget.fit([1.0, 2.0, 3.0])
        arr = target.transform([1.5, 2.5])
        assert arr.dtype == np.float64
        back = target.inverse_transform(arr)
        assert back == pytest.approx([1.5, 2.5])

    def test_json_roundtrip(self):
        target = RegressionTarget.fit([10.0, 20.0])
        d = target.to_json()
        restored = RegressionTarget.from_json(d)
        assert restored.mean == pytest.approx(target.mean)
        assert restored.std == pytest.approx(target.std)


class TestDataset:
    def test_basic_properties(self):
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        ds = Dataset(
            X=X,
            y=y,
            keys=[f"S{i}" for i in range(10)],
            feature_names=[f"f{i}" for i in range(5)],
            task_type=TaskType.BINARY,
        )
        assert ds.n_samples == 10
        assert ds.n_features == 5

    def test_inference_dataset_no_y(self):
        X = np.random.randn(5, 3)
        ds = Dataset(X=X)
        assert ds.y is None
        assert ds.n_samples == 5
