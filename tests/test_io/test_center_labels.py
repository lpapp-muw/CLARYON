"""Tests for claryon.io.center_labels — center label loading and attachment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from claryon.io.center_labels import attach_center_ids, load_center_labels

FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "center_labels_test.csv"


def test_load_center_labels():
    """Read CSV fixture, verify mapping."""
    mapping = load_center_labels(FIXTURE_CSV)
    assert len(mapping) == 9
    assert mapping["case_001"] == "DE"
    assert mapping["case_002"] == "CH"
    assert mapping["case_003"] == "AT"


def test_load_center_labels_file_not_found():
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_center_labels(Path("/nonexistent/path.csv"))


def test_attach_center_ids_all_matched():
    """All keys found in map."""
    mapping = load_center_labels(FIXTURE_CSV)
    keys = ["case_001", "case_005", "case_009"]
    result = attach_center_ids(keys, mapping)
    np.testing.assert_array_equal(result, ["DE", "CH", "AT"])


def test_attach_center_ids_missing_key():
    """Raise ValueError for unmatched key."""
    mapping = load_center_labels(FIXTURE_CSV)
    keys = ["case_001", "case_999"]
    with pytest.raises(ValueError, match="not found in center map"):
        attach_center_ids(keys, mapping)


def test_attach_center_ids_preserves_order():
    """Output order matches input keys."""
    mapping = load_center_labels(FIXTURE_CSV)
    keys = ["case_003", "case_001", "case_005", "case_007"]
    result = attach_center_ids(keys, mapping)
    np.testing.assert_array_equal(result, ["AT", "DE", "CH", "DE"])
