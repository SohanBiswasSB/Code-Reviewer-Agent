"""Measure the gate's score thresholds instead of guessing them.

    python -m tools.calibrate --tenant merger_app --root ~/Dev/<repo> relevance
    python -m tools.calibrate --tenant merger_app entailment
    python -m tools.calibrate --tenant merger_app relevance --json
    python -m tools.calibrate --tenant merger_app --root ~/Dev/<repo> rerank --commits 15

`min_relevance_score` and `min_entailment_score` (and `min_rerank_score` when
the T13 reranker is on) decide approve-vs-escalate, and the originally shipped
values were calibrated for an embedder this project no longer uses (CLAUDE.md,
"Thresholds are measured, not guessed"). This tool produces the measurement --
the interim 0.75 relevance bar in the merger_app policy and the 0.61
`MIN_RERANK_SCORE` default were both set from its numbers. It never edits a
policy: it prints the numbers and a suggestion, and a human writes the value into
`tenants/<name>/policy.json`.

## What each measurement is -- and what it is a PROXY for

**relevance** (needs the embedder, i.e. LM Studio running). For each of the last
N commits, every changed CODE file the index knows becomes a query built exactly
the way a review builds one (path + that file's diff, `surfaces/code_review.py`).
  · positive  = the best cosine to a chunk of the SAME file
  · negative  = the top cosines to chunks of OTHER files
"Same file" stands in for "relevant". It is noisy both ways -- another file can
be genuinely relevant (the model a view uses), and a long file has unrelated
chunks -- so read the overlap, not just the suggested cut.

**rerank** (needs the embedder AND the reranker model, T13). The same queries,
but measured where a review actually reads them: inside the retriever's real
candidate pool (the top RERANK_CANDIDATES by fusion). For each query, the
cross-encoder scores every candidate, then
  · positive  = the best RERANK score of a same-file candidate
  · negative  = the top rerank scores of other-file candidates
and the cosines of the SAME candidates are measured beside them, so the two
instruments are compared on identical pairs. It also reports how often a
same-file passage reaches the top PER_QUERY_K with and without reranking --
the retrieval metric that says whether the reranker earns its cost.

**entailment** (no model needed for the lexical entailer). Every claim of every
recorded review, scored exactly as the gate scores it now (per claim, against its
own citations plus its own file's diff, re-read from git). There is no label for
"supported", so it reports distributions and the pass rate at the current bar,
not a suggested cut. The sample is only as big as the number of reviews run.

The suggestion is the cut that maximises (positives kept - negatives kept),
i.e. Youden's J, plus the cut that still keeps 90% of positives -- the gate
fails closed, so a threshold that throws away most good evidence escalates
everything, which is the failure mode every real run has shown so far.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

import config
import registry
from adapters.audit_jsonl import iter_records
from adapters.rerank_cross_encoder import passage_text
from adapters.retriever_indexed import RERANK_CANDIDATES, IndexedRetriever
from app.live import JobProgress
from nodes.generate_diagnosis import PER_QUERY_K
from nodes.independent_review import effective_gates, entailment_claims
from schemas.diagnosis import Diagnosis
from tools.codebase import codebase_root
from tools.filekinds import CODE, file_rank
from tools.prepush.change import split_diff

PER_FILE_QUERY_CHARS = 1500     # the same budget surfaces/code_review.py uses
NEGATIVES_PER_QUERY = 5


def _git(root, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=False)
    return proc.stdout if proc.returncode == 0 else ""


def suggest(pos: list[float], neg: list[float]) -> dict:
    """Distribution summary and two candidate cuts for positive vs negative scores."""
    def describe(xs):
        """n, p10, median and p90 of `xs`; just {"n": 0} when it is empty."""
        if not xs:
            return {"n": 0}
        q = statistics.quantiles(xs, n=10) if len(xs) > 1 else [xs[0]] * 9
        return {"n": len(xs), "p10": round(q[0], 4), "median": round(statistics.median(xs), 4),
                "p90": round(q[-1], 4)}
    out = {"positive": describe(pos), "negative": describe(neg)}
    if not pos or not neg:
        out["note"] = "need both positives and negatives to suggest a cut"
        return out
    best, best_j = None, -1.0
    for t in sorted(set(pos + neg)):
        tpr = sum(p >= t for p in pos) / len(pos)
        fpr = sum(n >= t for n in neg) / len(neg)
        if tpr - fpr > best_j:
            best, best_j = t, tpr - fpr
    keep90 = sorted(pos)[int(len(pos) * 0.1)]
    out.update(youden_cut=round(best, 4), youden_j=round(best_j, 3),
               cut_keeping_90pct_positive=round(keep90, 4),
               negatives_passing_that_cut=round(sum(n >= keep90 for n in neg) / len(neg), 3))
    return out


def _no_progress(_done: int, _total: int) -> None:
    """Default progress sink: none. `main` passes the live job's instead.

    Same `(done, total)` shape as `JobProgress.progress`, always called
    positionally; this one reads neither.
    """


def _tenant_target(tenant: str, root_override: str | None) -> tuple[dict, str, object]:
    """(code-review policy, index namespace, repo root on this machine) for a tenant."""
    policy = config.get_tenant_store().policy(tenant, "code_review")
    codebase = policy.get("codebase") or {}
    return policy, codebase.get("repo"), codebase_root(codebase, root_override)


def relevance(tenant: str, root_override: str | None, commits: int,
              progress=_no_progress) -> dict:
    """Same-file vs other-file cosines for per-file queries over the last `commits`.

    Searches the tenant's whole namespace for each query, so every chunk is
    scored, not only the top k a review would see. Returns `suggest`'s summary
    plus the query count, the policy's current `min_relevance_score` and the
    embedder. See the module docstring for what "same file" is a proxy for.
    """
    policy, namespace, root = _tenant_target(tenant, root_override)
    index, embedder = config.get_index(), config.get_embedder()
    metas = index.documents(namespace)
    # A list, not the generator: the progress bar needs its length up front.
    items = list(_commit_queries(root, commits, {m.get("path") for m in metas}))
    pos, neg = _same_vs_other_file(
        lambda query: index.search(namespace, embedder.embed([query], kind="query")[0],
                                   len(metas)),
        items, progress)
    out = suggest(pos, neg)
    out.update(measure="relevance (cosine)", tenant=tenant, queries=len(pos),
               current=policy.get("min_relevance_score"), embedder=embedder.model)
    return out


def _same_vs_other_file(search, items: list[tuple[str, str]],
                        progress) -> tuple[list[float], list[float]]:
    """(positives, negatives) for `relevance`, one query at a time.

    `search(query)` returns every chunk of the namespace scored against the query.
    A positive is the best score of a chunk from the query's OWN file; the
    negatives are the top NEGATIVES_PER_QUERY scores of other files' chunks. A
    query with no chunk on one side has nothing to compare, so it adds neither.
    Kept apart so `relevance` reads as setup, measure, summarise.
    """
    pos, neg = [], []
    for n, (path, query) in enumerate(items, start=1):
        hits = search(query)
        same = [h["score"] for h in hits if h.get("path") == path]
        other = [h["score"] for h in hits if h.get("path") != path]
        if same and other:
            pos.append(max(same))
            neg.extend(other[:NEGATIVES_PER_QUERY])
        progress(n, len(items))
    return pos, neg


def _commit_queries(root, commits: int, paths: set):
    """(path, query) for each changed CODE file the index knows, newest first --
    built exactly the way a review builds a per-file query."""
    for sha in _git(root, "log", f"-{commits}", "--no-merges", "--format=%H").split():
        chunks = split_diff(_git(root, "show", "--format=", "--no-renames", sha))
        for path, diff in chunks.items():
            if path in paths and file_rank(path) == CODE:
                yield path, f"{path}\n{diff[:PER_FILE_QUERY_CHARS]}"


def _top_has_same(hits: list[dict], path: str, n: int) -> bool:
    return any(h.get("path") == path for h in hits[:n])


@dataclass
class _RerankTally:
    """Both instruments' scores on the same pools, accumulated query by query."""

    r_pos: list = field(default_factory=list)
    r_neg: list = field(default_factory=list)
    c_pos: list = field(default_factory=list)
    c_neg: list = field(default_factory=list)
    top_fused: int = 0
    top_reranked: int = 0

    def add(self, pool: list[dict], path: str) -> None:
        """One query's pool, every hit already carrying `rerank` and `cosine`."""
        same = [h for h in pool if h.get("path") == path]
        other = [h for h in pool if h.get("path") != path]
        self.r_pos.append(max(h["rerank"] for h in same))
        self.r_neg.extend(sorted((h["rerank"] for h in other), reverse=True)
                          [:NEGATIVES_PER_QUERY])
        self.c_pos.append(max(h["cosine"] for h in same))
        self.c_neg.extend(sorted((h["cosine"] for h in other), reverse=True)
                          [:NEGATIVES_PER_QUERY])
        self.top_fused += _top_has_same(pool, path, PER_QUERY_K)
        self.top_reranked += _top_has_same(
            sorted(pool, key=lambda h: h["rerank"], reverse=True), path, PER_QUERY_K)

    def summary(self) -> dict:
        """`suggest` for each instrument, plus how often the file's own code made top k."""
        n = max(len(self.r_pos), 1)
        return {"rerank": suggest(self.r_pos, self.r_neg),
                "cosine_same_pairs": suggest(self.c_pos, self.c_neg),
                "queries": len(self.r_pos),
                "same_file_in_top_k": {"k": PER_QUERY_K,
                                       "fusion_order": round(self.top_fused / n, 3),
                                       "reranked": round(self.top_reranked / n, 3)}}


def _tally_pools(pool_retriever, reranker, namespace: str, items: list[tuple[str, str]],
                 progress) -> _RerankTally:
    """Rerank each query's real candidate pool and tally both instruments on it.

    The pool is the top RERANK_CANDIDATES by fusion, retrieved WITHOUT a reranker
    so the cross-encoder scores every candidate here, onto `h["rerank"]`. A pool
    holding only the query's own file, or none of it, has no pair to compare and
    is skipped. Progress is reported before each query and once at the end.
    """
    tally = _RerankTally()
    for n, (path, query) in enumerate(items, start=1):
        progress(n - 1, len(items))
        pool = pool_retriever.retrieve(query, k=RERANK_CANDIDATES, namespace=namespace)
        if not any(h.get("path") == path for h in pool) or \
                all(h.get("path") == path for h in pool):
            continue
        for h, s in zip(pool, reranker.scores(query, [passage_text(h) for h in pool]),
                        strict=True):
            h["rerank"] = s
        tally.add(pool, path)
    progress(len(items), len(items))
    return tally


def rerank(tenant: str, root_override: str | None, commits: int,
           progress=_no_progress) -> dict:
    """Rerank vs cosine on the retriever's real candidate pool. See the docstring."""
    policy, namespace, root = _tenant_target(tenant, root_override)
    reranker = config.get_reranker() or registry.get_adapter("reranker", "cross_encoder")()
    pool_retriever = IndexedRetriever(reranker=None)
    started = time.monotonic()
    items = list(_commit_queries(
        root, commits, {m.get("path") for m in config.get_index().documents(namespace)}))
    out = _tally_pools(pool_retriever, reranker, namespace, items, progress).summary()
    out.update(measure="rerank vs cosine (same candidate pool)", tenant=tenant,
               pool=RERANK_CANDIDATES, reranker=reranker.model,
               current=policy.get("min_rerank_score", "default"),
               seconds_per_query=round((time.monotonic() - started)
                                       / max(out["queries"], 1), 2))
    return out


def _diff_for(root, target: str | None) -> str:
    """The change a recorded run reviewed, re-read from git. `staged` cannot be
    reproduced later, so it yields "" and those claims are scored on citations."""
    if not target or target == "staged":
        return ""
    if ".." in target:
        return _git(root, "diff", "--no-renames", target)
    return _git(root, "show", "--format=", "--no-renames", target)


def _score_records(records: list[dict], text: dict, root, entailer,
                   progress) -> tuple[list[float], list[float]]:
    """(every claim's score, the lowest claim per review) over recorded reviews.

    The lowest is collected per review because that is what decides at the gate.
    A record scored nothing (`_review_scores` is None) adds to neither list.
    Progress is reported before each record and once at the end.
    """
    claims, lowest = [], []
    for n, rec in enumerate(records, start=1):
        progress(n - 1, len(records))
        scored = _review_scores(rec, text, root, entailer)
        if scored is None:
            continue
        claims.extend(scored)
        lowest.append(min(scored, default=0.0))
    progress(len(records), len(records))
    return claims, lowest


def _review_scores(rec: dict, text: dict, root, entailer) -> list[float] | None:
    """Each claim of one recorded review, scored as the gate scores it now.

    The diagnosis is rebuilt from the record, its citations' passages from the
    index's current `text` (by id), and its diff re-read from git (`_diff_for`),
    then `entailment_claims` builds the very claims and premises the gate would.
    A claim or premise that is empty scores 0.0, as at the gate. None for a
    record made before findings were recorded (2026-11-16): it has no claims to
    score in today's shape.
    """
    d = rec.get("diagnosis") or {}
    if "findings" not in d:
        return None
    diag = Diagnosis.model_validate({
        "summary": d.get("summary") or "", "severity": d.get("severity") or "high",
        "proposed_comment": d.get("proposed_comment") or "",
        "citations": d.get("citations") or [], "findings": d.get("findings") or []})
    retrieved = [{"id": r.get("id"), "text": text.get(r.get("id"), "")}
                 for r in rec.get("retrieved") or []]
    return [entailer.score(c, p) if c and p else 0.0 for _, c, p in
            entailment_claims(diag, {"diff": _diff_for(root, rec.get("target"))}, retrieved)]


def entailment(tenant: str, root_override: str | None = None,
               progress=_no_progress) -> dict:
    """Per-claim entailment scores on REAL recorded reviews, computed by the gate's
    own `entailment_claims` -- so the numbers are what the gate would score now.

    There is no ground-truth label for "this finding is supported", so this does
    not suggest a cut. It reports the distribution, the lowest claim per review
    (what decides), and how many reviews the current bar would pass.
    """
    policy, namespace, root = _tenant_target(tenant, root_override)
    text = {m["id"]: m.get("text", "") for m in config.get_index().documents(namespace)}
    entailer = config.get_entailer()
    threshold = effective_gates(policy)["min_entailment_score"]
    records = [r for r in iter_records() if r.get("tenant") == tenant]
    claims, lowest = _score_records(records, text, root, entailer, progress)
    return {"per_claim": suggest(claims, [])["positive"],
            "lowest_claim_per_review": suggest(lowest, [])["positive"],
            "reviews": len(lowest),
            "reviews_passing_current_bar": sum(s >= threshold for s in lowest),
            "measure": "entailment (per claim, the gate's own premises)", "tenant": tenant,
            "entailer": type(entailer).__name__, "current": threshold}


def _measure(args, progress) -> dict:
    """Run the measure the CLI named."""
    if args.measure == "relevance":
        return relevance(args.tenant, args.root, args.commits, progress)
    if args.measure == "rerank":
        return rerank(args.tenant, args.root, args.commits, progress)
    return entailment(args.tenant, args.root, progress)


def _end_job(job: JobProgress, exc: BaseException | None) -> None:
    """finished / interrupted / error, from the exception in flight (if any)."""
    if exc is None:
        job.end("finished")
    elif isinstance(exc, KeyboardInterrupt):
        job.end("interrupted")
    else:
        job.end("error", error=f"{type(exc).__name__}: {exc}"[:300])


def main() -> int:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Measure the gate thresholds.")
    ap.add_argument("measure", choices=("relevance", "rerank", "entailment"))
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--root", help="the tenant's repo on this machine (all three measures)")
    ap.add_argument("--commits", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    # Shown on the monitor's "Now running" (app/live.JobProgress): a 15-minute
    # measurement used to leave the page reading "0 running" the whole time.
    job = JobProgress("calibrate", f"calibrate-{args.tenant}-{args.measure}",
                      f"Calibrate {args.measure} · {args.tenant}", tenant=args.tenant,
                      unit="reviews" if args.measure == "entailment" else "queries")
    measured = False
    try:
        job.phase("measuring")
        result = _measure(args, job.progress)
        measured = True
    finally:
        # The job's end is recorded on EVERY exit, the failure with its reason,
        # without catching anything: sys.exception() sees the exception in flight
        # inside `finally`, and it propagates unchanged afterwards. Only when the
        # body did NOT finish, though: on success sys.exception() can still return
        # an unrelated exception a CALLER is handling, which would end a finished
        # job as "error".
        _end_job(job, None if measured else sys.exception())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k:28} {v}")
        print("\nNothing was changed. To adopt a value, edit "
              f"tenants/{args.tenant}/policy.json and re-run the review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
