"""Compatibility wrapper for moved evaluate_classifiers.

This module forwards to `utils.evaluate_classifiers`. Keeping this
file allows users or scripts that import or run
`python3 -m evaluate_classifiers` to continue working.
"""

from utils import evaluate_classifiers  # noqa: F401
