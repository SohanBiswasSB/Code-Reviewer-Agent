"""Gather raw evidence through the active Surface's EvidenceSource port.

Previously this imported `tools.github_client` directly, which meant the
`EvidenceSource` port and the Surface's own `source_name` were both bypassed —
plug-and-play was decorative here (G2/T5). Now the provider is whatever the
surface names and config resolves, so GitHub -> GitLab is a config change.
"""
from __future__ import annotations

import config
from nodes._render import guard_payload
from nodes._surface import active_surface


def fetch_evidence(state: dict, source=None) -> dict:
    """`source` is injectable for tests; otherwise resolved from the surface."""
    target = state["target"]              # PR ref | issue url | YYYY-MM-DD

    if source is None:
        surface = active_surface(state)
        if surface is None:
            raise ValueError(
                "fetch_evidence: no active surface. resolve_tenant should have "
                "put `_surface_name` in state before this node ran."
            )
        source = config.get_source(config.source_name_for(surface))

    # The approver's own token when present; a shared credential is a fallback.
    # `codebase` is the tenant's binding, so a source that reads a repo (git_local)
    # knows WHICH repo without a second, competing configuration of its own.
    raw = source.gather(target, token=state.get("github_token"),
                        codebase=(state.get("_policy") or {}).get("codebase") or {})
    # T25: redacted HERE, before the first channel write, because the checkpoint
    # keeps every super-step -- see nodes/_render.py::guard_payload.
    clean, report = guard_payload(raw, tenant=state.get("tenant") or "_default")
    return {"_raw": clean, "_guard": {**(state.get("_guard") or {}), "evidence": report}}
