"""Ports = the interfaces the core depends on. Adapters implement these; the core
never imports a concrete provider. Swap providers without touching core/nodes.
"""
from __future__ import annotations
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, user: str, *, json: bool = False) -> str: ...


@runtime_checkable
class Retriever(Protocol):
    # returns [{"id": str, "text": str, "score": float}, ...] — score REQUIRED (gate reads it)
    # namespace = per-codebase isolation (G1): NEVER retrieve across tenants.
    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]: ...


@runtime_checkable
class EvidenceSource(Protocol):
    # gather raw evidence for a target (PR ref, issue url, or a date)
    def gather(self, target: str, **opts) -> dict: ...


@runtime_checkable
class ActionSink(Protocol):
    # perform the approved action (post PR comment, write report, open ticket)
    def emit(self, action: dict, *, identity: str | None = None) -> dict: ...


@runtime_checkable
class Detector(Protocol):
    # log-review: turn a parsed log window into anomaly candidates
    def detect(self, window: dict) -> list[dict]: ...


@runtime_checkable
class Checkpointer(Protocol):
    def saver(self) -> Any: ...


@runtime_checkable
class Surface(Protocol):
    """A plug-and-play capability (code_review, log_review, issue_triage).

    Binds an EvidenceSource + query strategy + prompt names + ActionSink to the
    shared graph. Register with @registry.surface("name").
    """
    name: str
    source: EvidenceSource
    sink: ActionSink
    diagnosis_prompt: str     # path under prompts/
    judge_prompt: str
    def build_query(self, evidence: dict) -> str: ...
    def classify(self, evidence: dict) -> str: ...      # rule-based route


@runtime_checkable
class TenantStore(Protocol):
    """Per-tenant prompts + guardrail policy, resolved at request time.

    Nothing is hardcoded: prompt(tenant, surface, node) and policy(tenant, surface)
    are looked up per org/project, falling back to the _default tenant. Backed by
    filesystem now; swap for DB / a config service by writing another adapter.
    """
    def prompt(self, tenant: str, surface: str, node: str) -> str: ...
    def policy(self, tenant: str, surface: str) -> dict: ...
    def tenants(self) -> list[str]: ...


@runtime_checkable
class CredentialProvider(Protocol):
    """G3: resolve a per-tenant secret (repo token, etc.) from a secret REF — never a raw
    token in config. Scope creds to the tenant's codebase."""
    def token_for(self, tenant: str, kind: str) -> str: ...


@runtime_checkable
class InputGuard(Protocol):
    """G5: redact PII/secrets and neutralize injected instructions BEFORE the LLM sees
    evidence. Returns cleaned text + what was redacted."""
    def sanitize(self, text: str, *, tenant: str) -> tuple[str, dict]: ...


@runtime_checkable
class AuditSink(Protocol):
    """G6: one immutable record per run (inputs, route, diagnosis, gate booleans, approver,
    action result). The system takes actions, so this is not optional."""
    def record(self, run: dict) -> None: ...


@runtime_checkable
class Notifier(Protocol):
    """G13: notify a human that a run is paused at the approval gate; carry the approval
    authZ context (who may approve for this tenant)."""
    def notify(self, tenant: str, payload: dict) -> None: ...


# ── accuracy-first ports (cost is no object) ─────────────────────────────────
@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder rerank of a wide candidate set → precise top-n. Ensemble-able."""
    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]: ...


@runtime_checkable
class ToolRunner(Protocol):
    """Run a deterministic analysis tool in a sandbox and return structured findings.
    Code review grounds on these FACTS (linters, mypy, semgrep, tests) — not on the LLM."""
    name: str
    def run(self, workdir: str, diff: str | None = None) -> list[dict]: ...


@runtime_checkable
class OutputGuard(Protocol):
    """Validate a proposed action against schema + policy + safety BEFORE it can execute."""
    def check(self, action: dict, *, tenant: str) -> tuple[bool, list[str]]: ...


@runtime_checkable
class Tracer(Protocol):
    """Span every node/prompt/tool/retrieval (OpenTelemetry / Langfuse / Phoenix)."""
    def span(self, name: str, **attrs): ...


@runtime_checkable
class Evaluator(Protocol):
    """Score a run: RAG-triad (context-relevance, faithfulness, answer-relevance), G-Eval,
    citation precision, hallucination rate, retrieval metrics."""
    def score(self, run: dict) -> dict: ...
