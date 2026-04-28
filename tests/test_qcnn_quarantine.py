"""Tests for the qCNN WIP quarantine introduced in CLARYON v0.13.0.

These tests lock in the quarantine behaviour so any future regression that
re-registers a qcnn model (intentionally or accidentally) trips a red CI
build. The two qcnn variants (qcnn_muw, qcnn_alt) MUST remain unregistered
until a validated implementation replaces them.

If you are intentionally lifting the quarantine (i.e. publishing a
validated qCNN), delete this whole file and re-enable the corresponding
@register decorators in claryon/models/quantum/qcnn_*.py and the auto-import
entries in claryon/pipeline.py and claryon/inference.py.
"""
from __future__ import annotations

import warnings

import pytest

QCNN_MODULE_PATHS = (
    "claryon.models.quantum.qcnn_muw",
    "claryon.models.quantum.qcnn_alt",
)

QCNN_REGISTRY_NAMES = (
    "qcnn_muw",
    "qcnn_alt",
)


class TestQCNNNotInPipelineAutoImport:
    """The pipeline's auto-import list must not pull in any qcnn module."""

    def test_pipeline_module_string_excludes_qcnn(self):
        """Static check: pipeline.py source should mention no qcnn module string."""
        from pathlib import Path

        import claryon.pipeline as _pipe

        src = Path(_pipe.__file__).read_text(encoding="utf-8")
        # Find the _import_model_modules function body.
        start = src.find("def _import_model_modules(")
        assert start != -1, "pipeline._import_model_modules not found"
        # Take a generous window after the function start.
        body = src[start : start + 3000]
        for path in QCNN_MODULE_PATHS:
            # The path may appear in a comment; check there's no live string entry.
            # A live entry would look like:  "claryon.models.quantum.qcnn_..."
            assert f'"{path}"' not in body, (
                f"{path} appears as a live string in pipeline._import_model_modules. "
                "qCNN must remain quarantined — see tests/test_qcnn_quarantine.py."
            )

    def test_inference_module_string_excludes_qcnn(self):
        """Same static check for inference.py."""
        from pathlib import Path

        import claryon.inference as _inf

        src = Path(_inf.__file__).read_text(encoding="utf-8")
        start = src.find("def _import_model_modules(")
        assert start != -1, "inference._import_model_modules not found"
        body = src[start : start + 3000]
        for path in QCNN_MODULE_PATHS:
            assert f'"{path}"' not in body, (
                f"{path} appears as a live string in inference._import_model_modules. "
                "qCNN must remain quarantined — see tests/test_qcnn_quarantine.py."
            )


class TestQCNNNotInRegistry:
    """After auto-import, the registry must contain no qcnn entry."""

    def test_pipeline_auto_import_leaves_registry_clean(self):
        from claryon.pipeline import _import_model_modules
        from claryon.registry import list_registered

        _import_model_modules()
        models = list_registered("model")
        leaked = [m for m in models if "qcnn" in m]
        assert leaked == [], (
            f"qCNN model(s) leaked into registry via pipeline auto-import: {leaked}. "
            "Re-quarantine via the @register decorator and pipeline.py auto-import."
        )

    def test_explicit_manual_import_leaves_registry_clean(self):
        """Belt-and-braces: even an explicit manual import must not register."""
        import importlib

        from claryon.registry import list_registered

        for path in QCNN_MODULE_PATHS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                importlib.import_module(path)

        models = list_registered("model")
        leaked = [m for m in models if "qcnn" in m]
        assert leaked == [], (
            f"qCNN model(s) registered via manual import: {leaked}. "
            "Confirm @register decorators are commented out in qcnn_*.py."
        )

    @pytest.mark.parametrize("name", QCNN_REGISTRY_NAMES)
    def test_get_qcnn_by_name_raises(self, name):
        """Looking up any qcnn name in the registry must raise KeyError."""
        from claryon.registry import get

        with pytest.raises(KeyError):
            get("model", name)


class TestQCNNImportEmitsWarning:
    """Each qcnn module must emit a UserWarning at import time."""

    @pytest.mark.parametrize("path", QCNN_MODULE_PATHS)
    def test_module_emits_user_warning(self, path):
        import importlib
        import sys

        # Force a fresh import so the warning fires (warnings are deduped per
        # source location across already-imported modules).
        if path in sys.modules:
            del sys.modules[path]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(path)

        wip_warnings = [w for w in caught if issubclass(w.category, UserWarning) and "WORK IN PROGRESS" in str(w.message)]
        assert wip_warnings, (
            f"Importing {path} did not emit the WIP UserWarning. "
            "Confirm the warnings.warn(...) block at the top of the module is intact."
        )


class TestQCNNClassesStillImportable:
    """The class definitions must remain importable for ongoing maintainer dev work."""

    def test_qcnn_muw_class_importable_and_instantiable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            from claryon.models.quantum.qcnn_muw import QCNNMuwModel
        m = QCNNMuwModel()
        assert m.name == "qcnn_muw"

    def test_qcnn_alt_class_importable_and_instantiable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            from claryon.models.quantum.qcnn_alt import QCNNAltModel
        m = QCNNAltModel()
        assert m.name == "qcnn_alt"


class TestPyprojectAllExtraExcludesRadiomics:
    """Regression guard for the Phase A.1 install fix.

    pyradiomics MUST NOT be in the [all] extra; otherwise pip install -e ".[all]"
    fails on a clean environment due to pyradiomics' broken setup.py
    (PEP 517 build isolation + missing numpy).
    """

    def test_radiomics_not_in_all_extra(self):
        from pathlib import Path

        try:
            import tomllib  # py>=3.11
        except ImportError:  # pragma: no cover - py<3.11 fallback
            import tomli as tomllib  # type: ignore

        # Walk up from tests/ to find pyproject.toml at repo root.
        here = Path(__file__).resolve()
        for parent in (here.parent, *here.parents):
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                pyproject = candidate
                break
        else:  # pragma: no cover - sanity guard
            pytest.skip("pyproject.toml not found — running outside source checkout?")

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        all_entries = data["project"]["optional-dependencies"]["all"]
        joined = " ".join(all_entries).lower()
        assert "radiomics" not in joined, (
            "pyradiomics has been re-added to the [all] extra. This breaks "
            'pip install -e ".[all]" on clean environments due to pyradiomics\' '
            "broken setup.py (numpy import in build env). Use scripts/install.sh "
            "or pip install -e \".[radiomics]\" --no-build-isolation instead."
        )
