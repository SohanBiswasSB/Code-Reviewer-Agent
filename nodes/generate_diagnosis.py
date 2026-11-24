"""RAG-grounded diagnosis, retrieved through the Retriever port.

Two things this node stopped doing (T5):

  · importing `tools.retrieval` directly, which bypassed the `Retriever` port and
    made the pluggable backend decorative;
  · retrieving with **no namespace**, which is why G1 was unenforced in practice.
    The port signature had a `namespace` parameter; this caller ignored it, so a
    multi-codebase index would happily let Project A's diagnosis cite Project B's
    code. Grounding is the entire safety story, so that failure would be silent
    and total.

The queries now come from the Surface, so each surface decides what a good query
looks like for its own evidence shape. A surface with `build_queries` (code
review) gets one query per changed file plus the whole-change query, merged by
`merge_hits` (2026-11-16); otherwise one `build_query` for the whole change.
"""
from __future__ import annotations

import json as _json

import config
from nodes._render import combine_guard_reports, render_evidence
from nodes._surface import active_surface, evidence_as_dict, namespace_for
from schemas.diagnosis import (
    Citation,
    Diagnosis,
    Finding,
    diagnosis_request_schema,
)


def _fallback_query(ev) -> str:
    """Title and body, for the built-in issue path or a surface with no query of its own."""
    title = getattr(ev, "title", "") or ""
    body = getattr(ev, "body", "") or ""
    return f"{title}\n{body}".strip()


# Multi-query retrieval: passages per query, and the most kept in total. The cap
# is set by the JUDGE's context (8k tokens on mac_max), which reads every passage
# again; at <= PROMPT_SNIPPET_CHARS each (nodes/_render.py) 8 passages fit with room.
PER_QUERY_K = 3
MAX_HITS = 8

# The keys a model's finding may set: anything else it emits is dropped before
# `Finding(...)` sees it. Computed once, from the schema's own field list, into
# a plain frozenset: pylint infers `Finding.model_fields` as pydantic's
# metaclass PROPERTY object and so reported `k in Finding.model_fields` as an
# unsupported membership test. Iterating it gives the same field names.
_FINDING_FIELDS = frozenset(Finding.model_fields)


def merge_hits(results: list[list[dict]], limit: int = MAX_HITS) -> list[dict]:
    """Several queries' hits -> one list: round-robin by rank, deduplicated.

    Round-robin so every query (every changed file) gets its best passage in
    before any gets a second. A chunk found by several queries keeps its HIGHEST
    retriever score -- its relevance to the part of the change it fits best,
    still the retriever's own number (the cosine, or the rerank score when T13's
    reranker is on), never a model's (T17).
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for rank in range(max((len(r) for r in results), default=0)):
        for hits in results:
            if rank < len(hits):
                h = hits[rank]
                if h["id"] not in best:
                    order.append(h["id"])
                    best[h["id"]] = h
                elif h.get("score", 0) > best[h["id"]].get("score", 0):
                    best[h["id"]] = {**best[h["id"]], "score": h["score"]}
    return [best[i] for i in order[:limit]]


def _retrieve(state: dict, ev, surface, retriever) -> list[dict]:
    """Evidence for the diagnosis: per changed file when the surface can say how
    (code review), else one query for the whole change."""
    ev_dict = evidence_as_dict(ev) if surface is not None else {}
    # G1: always namespaced when the tenant declares a codebase. A retriever that
    # cannot honour a namespace must REFUSE rather than silently retrieve across
    # codebases -- see adapters/retriever_hybrid.py.
    namespace = namespace_for(state)
    queries = surface.build_queries(ev_dict) if hasattr(surface, "build_queries") else []
    if queries:
        return merge_hits([retriever.retrieve(q, k=PER_QUERY_K, namespace=namespace)
                           for q in queries])
    query = (surface.build_query(ev_dict) if hasattr(surface, "build_query") else "") \
        or _fallback_query(ev)
    return retriever.retrieve(query, k=5, namespace=namespace)


def generate_diagnosis(state: dict, llm=None, retriever=None) -> dict:
    """Retrieve, then ask the diagnosis model to explain the change, citing passages.

    Reads `evidence`, `_checks`, the pinned `_prompts["diagnosis"]` and `_policy`
    (for the retrieval namespace), and the evidence half of `_guard`. Writes
    `diagnosis`, `_retrieved` (the hits WITH the retriever's own scores, which the
    gate reads) and the combined `_guard` report. Raises ValueError if
    resolve_tenant did not pin a diagnosis prompt.

    Protects grounding: retrieval is always namespaced to the tenant's codebase
    (invariant 7, G1), evidence is fenced and guarded before the model sees it
    (invariant 6), and citation scores come from the hits, never from the model
    (T17). The model's output is parsed here and never decides anything; a
    response that cannot be parsed becomes a diagnosis that cannot be approved
    (`_unparseable`). `llm` and `retriever` are injectable for tests.
    """
    ev = state["evidence"]
    surface = active_surface(state)
    if retriever is None:
        retriever = config.get_retriever()
    hits = _retrieve(state, ev, surface, retriever)

    # Resolved from config exactly like `retriever` above, and like `entailer`,
    # `source` and `sink` elsewhere. This used to fall through to a hardcoded stub
    # DIAGNOSIS object built right here, which meant `config.get_llm()` had no
    # caller in the whole codebase and the graph could never reach a real model --
    # the T5 shape (a correct component nothing calls), found the first time real
    # models were installed.
    #
    # Offline callers set LLM_PROVIDER=stub (eval/harness.py does) or inject an
    # llm directly. The stub returns TEXT and is parsed by `_parse_diagnosis`
    # below, so the offline path now exercises the parser too -- which the old
    # in-node stub bypassed entirely, and which is exactly where the first live
    # model failed.
    if llm is None:
        llm = config.get_llm("diagnosis")

    # The TENANT's prompt, resolved and pinned by resolve_tenant (T6). Read from
    # state rather than the store, so a resumed run uses the prompt it STARTED
    # with even if the tenant edited the file meanwhile.
    system = (state.get("_prompts") or {}).get("diagnosis")
    if not system:
        raise ValueError(
            "generate_diagnosis: no diagnosis prompt in state. resolve_tenant "
            "should have pinned it before this node ran."
        )
    # Constrained decoding, not just a prompt asking for JSON (T1). A prompt
    # instruction is advisory; a schema restricts token choice so the response
    # cannot leave the shape. The schema deliberately omits citation scores --
    # see schemas.diagnosis.diagnosis_request_schema.
    user_turn, guard_report = render_evidence(
        ev, hits, tenant=state.get("tenant") or "_default", checks=state.get("_checks"))
    raw = llm.complete(system, user_turn, json_schema=diagnosis_request_schema())
    diag = _parse_diagnosis(raw, hits)

    out = {"diagnosis": diag, "_retrieved": hits}
    if guard_report is not None:
        # Into state so it reaches the audit record and the trace (G6/T23). A
        # redaction nobody can see happened is indistinguishable from none.
        # Combined with what fetch_evidence redacted (T25): the diff's secrets
        # are already placeholders by now, so the prompt pass alone would report
        # them as zero. Built from the EVIDENCE breakdown rather than the running
        # total, so a node that runs twice cannot double-count.
        prior = (state.get("_guard") or {}).get("evidence")
        out["_guard"] = combine_guard_reports(prior, guard_report)
    return out


def _parse_diagnosis(raw: str, hits: list[dict]) -> Diagnosis:
    """Turn the model's JSON into a Diagnosis.

    Parsing lives here rather than in the adapter: an adapter that interpreted
    responses would be one step away from making decisions, and the whole design
    rests on the model never doing that.

    Three deliberate choices:

    · **Citation scores are looked up from `hits`, never taken from the model.**
      A model-supplied relevance score would be trivially gameable if anything
      trusted it, and the gate compares against exactly this number (T17).

    · **An unparseable or missing severity defaults to `high`, not `low`.**
      `risk_ok` only passes severities in the allowlist, so defaulting high means
      a malformed response escalates to a human. Defaulting low would auto-approve
      whatever the model failed to express clearly — the wrong direction to fail.

    · **`findings[]` is parsed** (T17). The prompt has always documented a
      per-file, per-line findings array; `Diagnosis` had no field for it, so the
      most useful half of the model's output was read and thrown away while
      anyone reading the prompt would assume it was used.
    """

    # A malformed response must ESCALATE, not crash the node. The graph has no
    # failure edges (G9), so an exception here kills the run -- and a killed run
    # writes no audit record at all, breaking the "every terminal path is
    # recorded" invariant (G6) at exactly the moment a record matters most.
    #
    # Constrained decoding (see the call site) should make this unreachable. It
    # is here because "should" is doing real work in that sentence: a provider
    # that ignores `response_format`, a truncated response from CRA_MAX_TOKENS,
    # or a model emitting a JSON preamble all produce this, and none of them
    # should be able to take down a run.
    try:
        data = _json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    except (ValueError, TypeError) as exc:
        return _unparseable(raw, exc)

    by_id = {h["id"]: h.get("score", 0.0) for h in hits}

    citations = []
    for entry in data.get("citations", []):
        cid = entry.get("snippet_id") if isinstance(entry, dict) else entry
        if cid:
            citations.append(Citation(snippet_id=cid, score=by_id.get(cid, 0.0)))

    findings = [
        Finding(**{k: v for k, v in f.items() if k in _FINDING_FIELDS})
        for f in data.get("findings", []) if isinstance(f, dict)
    ]

    return Diagnosis(
        summary=data.get("summary", ""),
        severity=data.get("severity") or "high",
        proposed_comment=data.get("proposed_comment", ""),
        proposed_label=data.get("proposed_label"),
        citations=citations,
        findings=findings,
    )


# The model's raw text is UNTRUSTED and ends up in an audit record and in front
# of a human, so only a short prefix is kept.
_UNPARSEABLE_EXCERPT = 300


def _unparseable(raw: str, exc: Exception) -> Diagnosis:
    """A diagnosis that cannot be read becomes one that cannot be approved.

    Every route to `approve` is closed deliberately, and by three independent
    booleans rather than one:

      · no citations      -> `groundedness_ok` False
      · severity "high"   -> `risk_ok` False for every shipped policy
      · empty comment     -> `entailment_ok` False (nothing to entail)

    Relying on a single one would make this depend on a tenant's policy values;
    a tenant that allowlisted `high` would otherwise turn a parse failure into an
    approvable action. The gate's `failures` list then tells the human what
    happened, and the audit record keeps it.

    `summary` carries the reason and an excerpt, because a silent escalation
    looks identical to a legitimate one -- and a systematically unparseable model
    would otherwise show up only as a mysteriously rising escalation rate, which
    MODEL_CARD.md already warns is the signal to watch.
    """
    excerpt = (raw or "")[:_UNPARSEABLE_EXCERPT].replace("\n", " ")
    truncated = "..." if raw and len(raw) > _UNPARSEABLE_EXCERPT else ""
    return Diagnosis(
        summary=(f"[unparseable model response] {type(exc).__name__}: {exc}. "
                 f"raw excerpt: {excerpt}{truncated}"),
        severity="high",
        proposed_comment="",
        proposed_label=None,
        citations=[],
        findings=[],
    )
