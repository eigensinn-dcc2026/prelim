"""Minimal, rule-aware data access utilities for DCC 2026."""

from .manifest import ManifestRow, build_manifest, load_manifest, write_manifest

__all__ = ["ManifestRow", "build_manifest", "load_manifest", "write_manifest"]
