"""Filesystem tenant store: tenants/<tenant>/{policy.json, prompts/<surface>/<node>.md}.

Policy resolves by deep merge over `_default`. Prompts **compose**: the org
contract is always included, and a tenant file is a delta on top of it.

## Why prompts compose rather than override (T16)

This used to fall back at *file* level — a tenant file was returned instead of
`_default`, never merged. Every tenant override in this repo is a short emphasis
note that says "Org contract applies (grounded-only, untrusted-input,
no-verdict, structured JSON)" as prose, referring to a file the model never saw.

So once prompts were actually loaded, the three overriding tenants would each
have sent a model a short priority list carrying **no grounding rule, no
untrusted-input rule, no no-verdict rule and no output schema**, while
`_default` — which nobody overrides — kept all of them. The tenants with the
strictest requirements would have got the weakest prompt, silently, because the
override *reads* as though it inherits.

The fix makes that structurally impossible rather than merely documented. The
org contract is prepended unless the tenant explicitly places it with
`{{include _default}}`; either way it is always present. This matches the
invariant in docs/MULTI_TENANT.md: **tenants tune values, they never remove the
boundary.**

Swap for a DB/remote adapter later without touching nodes — same TenantStore port.
"""
from __future__ import annotations

import json
import os

from registry import adapter

DEFAULT = "_default"
INCLUDE_DIRECTIVE = "{{include _default}}"

# Makes it obvious in a captured prompt (and in a trace) where the org contract
# ends and the tenant's delta begins. Debugging a prompt you cannot attribute is
# guesswork.
_DELTA_HEADER = "\n\n---\n# TENANT OVERLAY (applies on top of the org contract above)\n\n"


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

    # ── prompts ──────────────────────────────────────────────────────────────
    def _read_prompt(self, tenant: str, surface: str, node: str) -> str | None:
        path = os.path.join(self.root, tenant, "prompts", surface, f"{node}.md")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    def prompt(self, tenant: str, surface: str, node: str) -> str:
        """The composed prompt: org contract + this tenant's delta.

        A tenant with no override gets the contract verbatim. A tenant with an
        override gets the contract PLUS its delta — never the delta alone.
        """
        org = self._read_prompt(DEFAULT, surface, node)
        delta = self._read_prompt(tenant, surface, node) if tenant != DEFAULT else None

        if org is None and delta is None:
            raise FileNotFoundError(
                f"no prompt for {tenant}/{surface}/{node} (and no {DEFAULT} fallback)"
            )
        if delta is None:
            return org
        if org is None:
            # No org contract for this node at all. Return the tenant's file, but
            # this is a gap worth noticing rather than silently accepting: the
            # guardrail clauses live in _default.
            return delta
        if INCLUDE_DIRECTIVE in delta:
            return delta.replace(INCLUDE_DIRECTIVE, org)
        return org + _DELTA_HEADER + delta

    def prompt_provenance(self, tenant: str, surface: str, node: str) -> dict:
        """Which files contributed, for `doctor` and for audit records."""
        return {
            "org_contract": self._read_prompt(DEFAULT, surface, node) is not None,
            "tenant_delta": (tenant != DEFAULT
                             and self._read_prompt(tenant, surface, node) is not None),
        }

    # ── policy ───────────────────────────────────────────────────────────────
    def _raw_policy(self, tenant: str) -> dict:
        path = os.path.join(self.root, tenant, "policy.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def policy(self, tenant: str, surface: str) -> dict:
        merged = _deep_merge(self._raw_policy(DEFAULT), self._raw_policy(tenant))
        pol = dict(merged.get(surface, {}))
        pol["enabled_surfaces"] = merged.get("enabled_surfaces", [])
        pol["codebase"] = merged.get("codebase", {})   # the repo this tenant reviews
        return pol

    def tenants(self) -> list[str]:
        """Real codebases only.

        An underscore prefix marks a directory that is NOT a project: `_default`
        is the org contract, `_template` is the skeleton to copy. Excluding by
        prefix rather than by name means adding another such directory does not
        silently register itself as a tenant that doctor and the tests iterate.
        """
        return [d for d in sorted(os.listdir(self.root))
                if os.path.isdir(os.path.join(self.root, d)) and not d.startswith("_")]
