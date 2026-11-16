"""Ports = the interfaces the core depends on. Adapters implement these; the core
never imports a concrete provider. Swap providers without touching core/nodes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """One bound model. The model id is fixed at construction, NOT passed per call.

    That is deliberate: config.get_llm(role) hands back a different instance for
    the diagnosis role and the judge role, so a caller physically cannot reuse
    one model for both and quietly lose the independence property that
    independent_review depends on.

    `json_schema` requests schema-constrained decoding. Passing a schema is not
    the same as asking for JSON in the prompt: free-text instructions are
    advisory and a model can still emit invalid JSON, whereas constrained
    decoding restricts token choice so the output cannot leave the schema.
    """
    model: str
    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str: ...


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
    """Perform the approved action (post a PR comment, write a report, open a ticket).

    `target` is separate from `action` on purpose. The action is the content a
    human approved and `execute` posts it verbatim; the target is only where it
    goes. Folding the destination into the action dict would mean mutating the
    approved payload, and "posted verbatim" has to mean exactly that.

    `identity` is the approver's own credential, so the write is attributable to
    the person who authorised it rather than to a shared service token.

    A partial failure must be REPORTED, not raised: if a comment posts and a
    label then fails on permissions, the comment really did happen and throwing
    it away is worse than reporting both outcomes.

    `context` is run METADATA — tenant, surface, run_id — never content. It is
    separate from `action` for the same reason `target` is: the action is what a
    human approved and it is posted verbatim, so nothing may be folded into it
    afterwards. A sink that writes per-tenant artefacts (the daily report) needs
    to know which tenant; a sink that posts to a URL ignores it.
    """
    def emit(self, action: dict, *, target: str, identity: str | None = None,
             context: dict | None = None) -> dict: ...


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
    source_name: str          # which @adapter("source", ...) to gather with
    sink_name: str            # which @adapter("sink", ...) to emit through
    def build_query(self, evidence: dict) -> str: ...
    def classify(self, evidence: dict) -> str: ...      # rule-based route

    # Optional. A surface that gathers a different SHAPE of evidence must say how
    # to normalise it -- issue evidence and a code diff are not interchangeable,
    # and a surface routing on `touched_areas` gets nothing useful from an
    # IssueEvidence. Surfaces without one fall back to the built-in issue
    # normaliser.
    def normalize(self, raw: dict) -> object: ...


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
class Entailer(Protocol):
    """Does this claim actually FOLLOW from this passage? (T22)

    The third of the three grounding checks the design calls for. The first two
    -- the cited id is real, and its retrieval relevance clears a calibrated bar
    -- together prove only that a real, on-topic snippet was cited. They cannot
    catch a claim that OVERREACHES what its passage supports, which is the most
    common RAG failure and the one a citation makes look trustworthy.

    Returns a SCORE, not a verdict, for the same reason the retriever does:
    `independent_review` thresholds it in Python. A port that returned a boolean
    would be making the decision, and no component outside the gate may.
    """
    def score(self, claim: str, passage: str) -> float: ...


@runtime_checkable
class Embedder(Protocol):
    """Text -> vectors. The port that did not exist while the embedder was pinned.

    `EMBEDDING_MODEL` has been pinned across every profile since the project
    began, `doctor` verifies it is loaded, and its dimensionality is asserted by
    a test -- yet nothing ever called it. Retrieval's "dense" half was Jaccard
    word overlap. That is the same shape as the LLM wiring gap (T1): a configured,
    verified component with no caller.

    Separate from `LLM` on purpose. An embedding endpoint takes a batch and
    returns vectors; `LLM.complete` takes one system/user pair and returns text.
    Forcing one port to cover both would make every implementation of it lie
    about half its surface.

    `dims` is declared rather than inferred so an index can be validated against
    the embedder that is about to query it. A vector store silently queried with
    the wrong dimensionality returns nonsense rather than an error, and every
    `min_relevance_score` in `tenants/*/policy.json` would quietly change meaning.
    """
    model: str
    dims: int
    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """A namespaced store of embedded chunks.

    `namespace` is not optional anywhere in this port. G1 says retrieval must
    never cross codebases, and an index whose isolation is a caller-supplied
    nicety is an index that will eventually be queried without one.

    `manifest` records WHICH embedder built the namespace (model, dims). A vector
    is only meaningful in the space that produced it, so the retriever compares
    the manifest against the embedder about to query and refuses on a mismatch.
    Without it, swapping the embedder would leave every old index answering with
    cosines that look normal and mean nothing.

    `documents` and `similarities` exist for HYBRID retrieval: BM25 needs every
    chunk's text, and a chunk surfaced by BM25 alone still needs a real cosine
    for the gate to read -- otherwise a keyword-only hit would arrive with no
    gateable score (the same reason `tools/retrieval.py::relevance_of` exists).
    """
    def upsert(self, namespace: str, chunks: list[dict],
               manifest: dict | None = None) -> int: ...
    def search(self, namespace: str, vector: list[float], k: int) -> list[dict]: ...
    def similarities(self, namespace: str, vector: list[float],
                     ids: list[str]) -> dict[str, float]: ...
    def documents(self, namespace: str) -> list[dict]: ...
    def namespaces(self) -> list[str]: ...
    def stats(self, namespace: str) -> dict: ...


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


# ── hardware / model selection ────────────────────────────────────────────────
@runtime_checkable
class Platform(Protocol):
    """The ONLY thing that knows about hardware. Nothing in nodes/ may branch on it.

    Two orthogonal axes, deliberately not conflated:
      · runtime()  -> HOW to run (context budget, KV-cache dtype, residency,
                      threads, torch device). Differs per machine by necessity.
      · models()   -> WHICH models to run. Comes from a named profile that is
                      shared across machines, so results stay comparable.

    The default profile must run on EVERY platform. A platform advertises the
    profiles it can actually hold via `profiles`; asking for one it can't fit is
    an error, not a silent downgrade to something else.

    Residency is COMPUTED from the profile's weight cost against a per-platform
    memory budget, not declared — which is why runtime() takes the profile.
    """
    name: str
    profiles: tuple[str, ...]        # profiles this hardware can actually hold
    def runtime(self, profile: str) -> dict: ...
    def models(self, profile: str) -> dict: ...   # role -> model id
