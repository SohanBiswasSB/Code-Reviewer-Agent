"""Ports = the interfaces the core depends on. Adapters implement these; the core
never imports a concrete provider. Swap providers without touching core/nodes.

The dependency is on the SHAPE, not on this module: these Protocols are
structural, and nothing imports this file (not the nodes, the adapters, nor the
tests) or runs an `isinstance` check against them. Adapters conform by having
the right methods, so this file is the written contract -- a signature that
drifts here is not caught by anything but review.

Some ports are DECLARED AHEAD of any adapter, on purpose: they fix the shape of
a planned seam (docs/ARCHITECTURE.md) so the first implementation has a contract
to meet rather than inventing one. Nothing implements them yet, and nothing
calls them -- do not read their presence as a working feature:

    Checkpointer (LangGraph's own savers are used directly), CredentialProvider,
    Notifier, ToolRunner, OutputGuard, Evaluator.

`Reranker` (T13) left that list on 2026-11-17: adapters/rerank_cross_encoder.py
implements it and `IndexedRetriever` calls it -- but only when
RERANKER=cross_encoder. It is OFF by default because, measured, its same- vs
other-file separation was Youden's J 0.31 against the cosine's 0.68
(`config.get_reranker` has the numbers). Built, wired, and opt-in.
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
    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """Run one system/user exchange and return the model's raw text.

        The caller parses it, and must fail safe: constrained decoding fixes the
        structure, not the content, so a model can still loop inside a string
        until the token budget dies (see `_parse_diagnosis`).
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Top-k passages for a query, never mixing codebases (G1)."""

    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]:
        """Return `[{"id": str, "text": str, "score": float, ...}, ...]`, best first.

        `score` is REQUIRED: the gate thresholds exactly that number, so a hit
        without one could not ground anything. It must be a calibrated relevance,
        never a rank-fusion value (T21). A hit may also carry `score_kind`
        ("cosine" or "rerank", T13) and the gate picks its bar by kind; a hit
        without one is held to the cosine bar.

        `namespace` is per-codebase isolation (G1): NEVER retrieve across
        tenants. A retriever that cannot honour a namespace must refuse, not
        ignore the argument.
        """
        ...


@runtime_checkable
class EvidenceSource(Protocol):
    """Where a surface's raw evidence comes from (a PR, a local checkout, a day of logs)."""

    def gather(self, target: str, **opts) -> dict:
        """Gather raw evidence for a target (PR ref, issue url, git range, or a date).

        Raw on purpose: the surface's `normalize` validates and shapes it, so
        provider data fails at that boundary rather than deep in the graph (T11).
        """
        ...


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
             context: dict | None = None) -> dict:
        """Post the approved `action` verbatim to `target`; return what happened.

        The returned dict reports every outcome, partial failures included,
        rather than raising after something irreversible already landed.
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """Log review's first pass: rule-based anomaly detection over the full stream (G4)."""

    def detect(self, window: dict) -> list[dict]:
        """Turn a parsed log window into anomaly candidates.

        Runs over every line before any model sees anything, so it has to be
        cheap enough for a ~2M-line day (see `config.get_detector`).
        """
        ...


@runtime_checkable
class Checkpointer(Protocol):
    """Declared ahead; unused. `app/deps.py` builds LangGraph's savers directly."""

    def saver(self) -> Any:
        """A LangGraph checkpoint saver, built with `app.state.checkpoint_serde()`."""
        ...


@runtime_checkable
class Surface(Protocol):
    """A plug-and-play capability (code_review, log_review, issue_triage).

    Binds an EvidenceSource + query strategy + prompt names + ActionSink to the
    shared graph. Register with @registry.surface("name").

    `source_name` / `sink_name` are the surface's own choices; a deployment may
    override either per surface with `CRA_SOURCE_<SURFACE>` / `CRA_SINK_<SURFACE>`
    (`config.source_name_for`, `config.sink_name_for`).

    A surface may also define `build_queries(evidence) -> list[str]`, not declared
    here because it is optional: `generate_diagnosis` uses it when it exists and
    returns queries (code review makes one per changed file), else `build_query`.
    """
    name: str
    source_name: str          # which @adapter("source", ...) to gather with
    sink_name: str            # which @adapter("sink", ...) to emit through
    def build_query(self, evidence: dict) -> str:
        """The retrieval query for this evidence -- the whole change as one query."""
        ...

    def classify(self, evidence: dict) -> str:
        """The route, decided by rules over the evidence. Zero LLM calls, by design."""
        ...

    def normalize(self, raw: dict, policy: dict | None = None) -> object:
        """Optional: turn this surface's raw evidence into its validated shape.

        A surface that gathers a different SHAPE of evidence must say how to
        normalise it -- issue evidence and a code diff are not interchangeable,
        and a surface routing on `touched_areas` gets nothing useful from an
        IssueEvidence. Surfaces without one fall back to the built-in issue
        normaliser. `policy` is the tenant's resolved policy, for VALUES the
        surface applies while normalising (e.g. extra risky_areas, T27).
        """
        ...


@runtime_checkable
class TenantStore(Protocol):
    """Per-tenant prompts + guardrail policy, resolved at request time.

    Nothing is hardcoded: prompt(tenant, surface, node) and policy(tenant, surface)
    are looked up per org/project and built on the _default tenant -- a policy is
    deep-merged over it, and a prompt COMPOSES with the org contract rather than
    replacing it (T16), so a tenant tunes values but cannot drop a boundary.
    Backed by filesystem now; swap for DB / a config service by writing another
    adapter.
    """
    def prompt(self, tenant: str, surface: str, node: str) -> str:
        """The prompt text for one node: the org contract plus the tenant's delta."""
        ...

    def policy(self, tenant: str, surface: str) -> dict:
        """The tenant's resolved policy for one surface, merged over `_default`."""
        ...

    def tenants(self) -> list[str]:
        """The real codebases this store serves -- not `_default`, the org contract."""
        ...


@runtime_checkable
class CredentialProvider(Protocol):
    """G3: resolve a per-tenant secret (repo token, etc.) from a secret REF — never a raw
    token in config. Scope creds to the tenant's codebase."""
    def token_for(self, tenant: str, kind: str) -> str:
        """The secret of `kind` for `tenant`. Declared ahead; nothing implements it."""
        ...


@runtime_checkable
class InputGuard(Protocol):
    """G5/T12: redact PII/secrets and defang structural injection BEFORE the LLM sees evidence.

    Returns the cleaned text plus a report of what was redacted. Prose that looks
    like an injected instruction is COUNTED in the report, not rewritten; what is
    neutralised is fence forgery -- evidence carrying the prompt's own fence
    markers to close its fence early. The real defence against prose injection is
    that no model output becomes an action without four booleans and a human.
    """
    def sanitize(self, text: str, *, tenant: str) -> tuple[str, dict]:
        """Return (cleaned text, report). The report must never carry a redacted value."""
        ...


@runtime_checkable
class AuditSink(Protocol):
    """G6: one immutable record per run (inputs, route, diagnosis, gate booleans, approver,
    action result). The system takes actions, so this is not optional."""
    def record(self, run: dict) -> None:
        """Append one run's record. Records are never updated in place."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """G13: notify a human that a run is paused at the approval gate; carry the approval
    authZ context (who may approve for this tenant)."""
    def notify(self, tenant: str, payload: dict) -> None:
        """Tell `tenant`'s approvers a run is waiting. Declared ahead; unimplemented."""
        ...


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
    def score(self, claim: str, passage: str) -> float:
        """How well `passage` (the premise) supports `claim` (the hypothesis), 0..1.

        Fails closed: a claim that cannot be checked scores 0.0, not 1.0.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text -> vectors. The port that did not exist while the embedder was pinned.

    `EMBEDDING_MODEL` has been pinned across every profile since the project
    began, `doctor` verifies it is loaded, and its dimensionality is asserted by
    a test -- yet until T10 nothing called it. Retrieval's "dense" half was
    Jaccard word overlap. That is the same shape as the LLM wiring gap (T1): a
    configured, verified component with no caller. (Now `IndexedRetriever`,
    `tools.ingest` and `tools.calibrate` all embed through this port.)

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
    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        """One vector per text, each `dims` wide, in input order.

        `kind` is "query" or "passage": an asymmetric embedder such as
        EmbeddingGemma documents a different task prefix for each, and the
        retriever embeds queries with kind="query" while ingest embeds chunks
        as passages.
        """
        ...


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
               manifest: dict | None = None) -> int:
        """Replace the namespace's contents wholesale; return how many chunks it holds.

        Whole-namespace rather than incremental, so a deleted file's chunks
        cannot linger and stay citable.
        """
        ...

    def search(self, namespace: str, vector: list[float], k: int) -> list[dict]:
        """Top-k chunks by cosine, each with its `score`. [] for an unbuilt namespace."""
        ...

    def similarities(self, namespace: str, vector: list[float],
                     ids: list[str]) -> dict[str, float]:
        """Cosine for specific chunk ids. Unknown ids are omitted, never zeroed."""
        ...

    def documents(self, namespace: str) -> list[dict]:
        """Every chunk's metadata and text, for BM25."""
        ...

    def namespaces(self) -> list[str]:
        """The namespaces that are fully built."""
        ...

    def stats(self, namespace: str) -> dict:
        """The namespace's manifest plus `built`; the retriever checks it before querying."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder rerank of a wide candidate set → precise top-n. Ensemble-able.

    Implemented (adapters/rerank_cross_encoder.py) and OFF by default: measured
    against the cosine on the same pools it separated same- from other-file
    passages worse (Youden's J 0.31 vs 0.68). Opt in with RERANKER=cross_encoder.
    """
    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
        """The best `top_n` candidates, best first, each a COPY with a `rerank` score.

        It scores; it never decides. The retriever makes `rerank` the hit's gate
        score with `score_kind="rerank"`, and the gate holds it to
        `min_rerank_score` -- never to the cosine bar, which was measured for a
        different model's scale.
        """
        ...


@runtime_checkable
class ToolRunner(Protocol):
    """Run a deterministic analysis tool in a sandbox and return structured findings.
    Code review grounds on these FACTS (linters, mypy, semgrep, tests) — not on the LLM."""
    name: str
    def run(self, workdir: str, diff: str | None = None) -> list[dict]:
        """Findings for `workdir`, optionally limited to `diff`. Declared ahead; unused."""
        ...


@runtime_checkable
class OutputGuard(Protocol):
    """Validate a proposed action against schema + policy + safety BEFORE it can execute."""
    def check(self, action: dict, *, tenant: str) -> tuple[bool, list[str]]:
        """Return (allowed, reasons). Declared ahead; nothing implements it yet."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Span the run's work so a bad run can be explained afterwards (T23).

    What is spanned today: every graph node (`app/graph.py`), every LLM call and
    every LLM retry (`adapters/llm_lmstudio.py`). Prompt, tool and retrieval spans
    are not emitted yet. The adapters are append-only JSONL and a null tracer
    (adapters/tracer_jsonl.py); OpenTelemetry / Langfuse / Phoenix would be other
    adapters behind this port, and none exists.
    """
    def span(self, name: str, **attrs):
        """A context manager that times its block and records one span.

        It must never change the block's outcome: a tracing failure is swallowed,
        and the block's own exception always propagates unchanged.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Score a run: RAG-triad (context-relevance, faithfulness, answer-relevance), G-Eval,
    citation precision, hallucination rate, retrieval metrics."""
    def score(self, run: dict) -> dict:
        """Metric name -> value for one run. Declared ahead; `eval/` does not use it."""
        ...


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
    def runtime(self, profile: str) -> dict:
        """HOW to run `profile` on this machine: context budget, KV dtype, residency, device."""
        ...

    def models(self, profile: str) -> dict:
        """role -> model id for `profile`. Raises for a profile this hardware cannot hold."""
        ...
