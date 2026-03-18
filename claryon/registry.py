"""Decorator-based plugin registry for models, metrics, explainers, and encodings."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Global registry: (namespace, name) → class/function
_REGISTRY: Dict[Tuple[str, str], Any] = {}


def register(namespace: str, name: str) -> Any:
    """Register a class or function in the global registry.

    Args:
        namespace: Category (e.g., "model", "metric", "explainer", "encoding").
        name: Unique name within the namespace.

    Returns:
        Decorator that registers the target and returns it unchanged.

    Raises:
        ValueError: If (namespace, name) is already registered.

    Example::

        @register("model", "xgboost")
        class XGBoostModel(ModelBuilder): ...
    """
    def decorator(obj: Any) -> Any:
        key = (namespace, name)
        if key in _REGISTRY:
            raise ValueError(
                f"Duplicate registration: ({namespace!r}, {name!r}) "
                f"already registered to {_REGISTRY[key]!r}"
            )
        _REGISTRY[key] = obj
        logger.debug("Registered %s/%s → %r", namespace, name, obj)
        return obj
    return decorator


def get(namespace: str, name: str) -> Any:
    """Retrieve a registered class or function.

    Args:
        namespace: Category (e.g., "model").
        name: Registered name.

    Returns:
        The registered object.

    Raises:
        KeyError: If (namespace, name) is not registered.
    """
    key = (namespace, name)
    if key not in _REGISTRY:
        available = [n for ns, n in _REGISTRY if ns == namespace]
        raise KeyError(
            f"Nothing registered as ({namespace!r}, {name!r}). "
            f"Available in {namespace!r}: {available}"
        )
    return _REGISTRY[key]


def list_registered(namespace: Optional[str] = None) -> Dict[str, Any]:
    """List all registered items, optionally filtered by namespace.

    Args:
        namespace: If given, only return items in this namespace.

    Returns:
        Dict mapping name → registered object.
    """
    if namespace is not None:
        return {n: obj for (ns, n), obj in _REGISTRY.items() if ns == namespace}
    return {f"{ns}/{n}": obj for (ns, n), obj in _REGISTRY.items()}


def clear(namespace: Optional[str] = None) -> None:
    """Remove all registrations, optionally only for a given namespace.

    Args:
        namespace: If given, only clear this namespace.
    """
    if namespace is None:
        _REGISTRY.clear()
    else:
        keys = [k for k in _REGISTRY if k[0] == namespace]
        for k in keys:
            del _REGISTRY[k]
