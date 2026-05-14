"""Reusable modules for the DOBOT VLM/VLA project.

The command-line files in ``client/``, ``server/``, and ``scripts/`` are kept
as entry points. Shared robot, camera, task, and inference behavior lives here
so a new teammate can reuse the pipeline without copying script internals.
"""

__all__ = ["application", "domain", "infrastructure"]
