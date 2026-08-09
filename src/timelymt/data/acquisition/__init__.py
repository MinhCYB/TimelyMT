"""Curated talk acquisition interfaces."""

from .core import (
    AcquisitionResult,
    Candidate,
    Discovery,
    acquire_candidates,
    artifact_directory,
    load_manifest,
    validate_artifacts,
)
from .ted import TedAdapter

__all__ = [
    "AcquisitionResult",
    "Candidate",
    "Discovery",
    "TedAdapter",
    "acquire_candidates",
    "artifact_directory",
    "load_manifest",
    "validate_artifacts",
]
