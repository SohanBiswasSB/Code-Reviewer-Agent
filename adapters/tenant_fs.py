"""Filesystem tenant store: tenants/<tenant>/{policy.json, prompts/<surface>/<node>.md}.

Resolution = tenant value overrides _default (prompts: file fallback; policy: deep-merge).
Swap for a DB/remote adapter later without touching nodes — same TenantStore port.
"""
from __future__ import annotations

import json
import os

from registry import adapter

DEFAULT = "_default"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        nested = isinstance(v, dict) and isinstance(base.get(k), dict)
        out[k] = _deep_merge(base[k], v) if nested else v
    return out


@adapter("tenant", "fs")
class FsTenantStore:
    def __init__(self, root: str | None = None):
        self.root = root or os.getenv("TENANTS_DIR", "tenants")

    def prompt(self, tenant: str, surface: str, node: str) -> str:
        for t in (tenant, DEFAULT):
            p = os.path.join(self.root, t, "prompts", surface, f"{node}.md")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return f.read()
        raise FileNotFoundError(f"no prompt for {tenant}/{surface}/{node} (and no default)")

    def _raw_policy(self, tenant: str) -> dict:
        p = os.path.join(self.root, tenant, "policy.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    def policy(self, tenant: str, surface: str) -> dict:
        merged = _deep_merge(self._raw_policy(DEFAULT), self._raw_policy(tenant))
        pol = dict(merged.get(surface, {}))
        pol["enabled_surfaces"] = merged.get("enabled_surfaces", [])
        pol["codebase"] = merged.get("codebase", {})   # the repo this tenant reviews
        return pol

    def tenants(self) -> list[str]:
        return [d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d)) and d != DEFAULT]
