"""Gather raw evidence through the active Surface's EvidenceSource port.

Previously this imported `tools.github_client` directly, which meant the
`EvidenceSource` port and the Surface's own `source_name` were both bypassed —
plug-and-play was decorative here (G2/T5). Now the provider is whatever the
surface names and config resolves, so GitHub -> GitLab is a config change.
"""
from __future__ import annotations

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
        import config
        source = config.get_source(surface.source_name)

    # The approver's own token when present; a shared credential is a fallback.
    raw = source.gather(target, token=state.get("github_token"))
    return {"_raw": raw}
