"""Measure the two gate thresholds instead of guessing them.

    python -m tools.calibrate --tenant masterclass --root ~/Dev/<repo> relevance
    python -m tools.calibrate --tenant masterclass entailment
    python -m tools.calibrate --tenant masterclass relevance --json
    python -m tools.calibrate --tenant masterclass --root ~/Dev/<repo> rerank --commits 15

`min_relevance_score` and `min_entailment_score` decide approve-vs-escalate, and
both shipped values were calibrated for an embedder this project no longer uses
(CLAUDE.md, "Thresholds are measured, not guessed"). This tool produces the
measurement. It never edits a policy: it prints the numbers and a suggestion, and
a human writes the value into `tenants/<name>/policy.json`.

## What each measurement is -- and what it is a PROXY for

**relevance** (needs the embedder, i.e. LM Studio running). For each of the last
N commits, every changed file the index knows becomes a query built exactly the
way a review builds one (path + that file's diff, `surfaces/code_review.py`).
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

import config
import registry
from adapters.audit_jsonl import iter_records
from adapters.rerank_cross_encoder import passage_text
from adapters.retriever_indexed import RERANK_CANDIDATES, IndexedRetriever
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


def relevance(tenant: str, root_override: str | None, commits: int) -> dict:
    policy = config.get_tenant_store().policy(tenant, "code_review")
    namespace = (policy.get("codebase") or {}).get("repo")
    root = codebase_root(policy.get("codebase") or {}, root_override)
    index, embedder = config.get_index(), config.get_embedder()
    metas = index.documents(namespace)
    paths = {m.get("path") for m in metas}
    pos, neg, queries = [], [], 0
    for sha in _git(root, "log", f"-{commits}", "--no-merges", "--format=%H").split():
        chunks = split_diff(_git(root, "show", "--format=", "--no-renames", sha))
        for path, diff in chunks.items():
            if path not in paths or file_rank(path) != CODE:
                continue
            query = f"{path}\n{diff[:PER_FILE_QUERY_CHARS]}"
            qvec = embedder.embed([query], kind="query")[0]
            hits = index.search(namespace, qvec, len(metas))
            same = [h["score"] for h in hits if h.get("path") == path]
            other = [h["score"] for h in hits if h.get("path") != path]
            if same and other:
                pos.append(max(same))
                neg.extend(other[:NEGATIVES_PER_QUERY])
                queries += 1
    out = suggest(pos, neg)
    out.update(measure="relevance (cosine)", tenant=tenant, queries=queries,
               current=policy.get("min_relevance_score"), embedder=embedder.model)
    return out


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


def rerank(tenant: str, root_override: str | None, commits: int) -> dict:
    """Rerank vs cosine on the retriever's real candidate pool. See the docstring."""
    policy = config.get_tenant_store().policy(tenant, "code_review")
    namespace = (policy.get("codebase") or {}).get("repo")
    root = codebase_root(policy.get("codebase") or {}, root_override)
    reranker = config.get_reranker() or registry.get_adapter("reranker", "cross_encoder")()
    pool_retriever = IndexedRetriever(reranker=None)
    paths = {m.get("path") for m in config.get_index().documents(namespace)}
    r_pos, r_neg, c_pos, c_neg = [], [], [], []
    top_fused = top_reranked = queries = 0
    started = time.monotonic()
    for path, query in _commit_queries(root, commits, paths):
        pool = pool_retriever.retrieve(query, k=RERANK_CANDIDATES, namespace=namespace)
        same = [h for h in pool if h.get("path") == path]
        other = [h for h in pool if h.get("path") != path]
        if not same or not other:
            continue
        scores = reranker.scores(query, [passage_text(h) for h in pool])
        for h, s in zip(pool, scores, strict=True):
            h["rerank"] = s
        r_pos.append(max(h["rerank"] for h in same))
        r_neg.extend(sorted((h["rerank"] for h in other), reverse=True)[:NEGATIVES_PER_QUERY])
        c_pos.append(max(h["cosine"] for h in same))
        c_neg.extend(sorted((h["cosine"] for h in other), reverse=True)[:NEGATIVES_PER_QUERY])
        top_fused += _top_has_same(pool, path, PER_QUERY_K)
        top_reranked += _top_has_same(sorted(pool, key=lambda h: h["rerank"], reverse=True),
                                      path, PER_QUERY_K)
        queries += 1
    out = {"rerank": suggest(r_pos, r_neg), "cosine_same_pairs": suggest(c_pos, c_neg)}
    out.update(measure="rerank vs cosine (same candidate pool)", tenant=tenant,
               queries=queries, pool=RERANK_CANDIDATES, reranker=reranker.model,
               current=policy.get("min_rerank_score", "default"),
               same_file_in_top_k={"k": PER_QUERY_K,
                                   "fusion_order": round(top_fused / max(queries, 1), 3),
                                   "reranked": round(top_reranked / max(queries, 1), 3)},
               seconds_per_query=round((time.monotonic() - started) / max(queries, 1), 2))
    return out


def _diff_for(root, target: str | None) -> str:
    """The change a recorded run reviewed, re-read from git. `staged` cannot be
    reproduced later, so it yields "" and those claims are scored on citations."""
    if not target or target == "staged":
        return ""
    if ".." in target:
        return _git(root, "diff", "--no-renames", target)
    return _git(root, "show", "--format=", "--no-renames", target)


def entailment(tenant: str, root_override: str | None = None) -> dict:
    """Per-claim entailment scores on REAL recorded reviews, computed by the gate's
    own `entailment_claims` -- so the numbers are what the gate would score now.

    There is no ground-truth label for "this finding is supported", so this does
    not suggest a cut. It reports the distribution, the lowest claim per review
    (what decides), and how many reviews the current bar would pass.
    """
    policy = config.get_tenant_store().policy(tenant, "code_review")
    namespace = (policy.get("codebase") or {}).get("repo")
    root = codebase_root(policy.get("codebase") or {}, root_override)
    text = {m["id"]: m.get("text", "") for m in config.get_index().documents(namespace)}
    entailer = config.get_entailer()
    bar = effective_gates(policy)["min_entailment_score"]
    claims, lowest = [], []
    for rec in (r for r in iter_records() if r.get("tenant") == tenant):
        d = rec.get("diagnosis") or {}
        if "findings" not in d:
            continue        # recorded before findings were (2026-11-16)
        diag = Diagnosis.model_validate({
            "summary": d.get("summary") or "", "severity": d.get("severity") or "high",
            "proposed_comment": d.get("proposed_comment") or "",
            "citations": d.get("citations") or [], "findings": d.get("findings") or []})
        retrieved = [{"id": r.get("id"), "text": text.get(r.get("id"), "")}
                     for r in rec.get("retrieved") or []]
        scored = [entailer.score(c, p) if c and p else 0.0 for _, c, p in
                  entailment_claims(diag, {"diff": _diff_for(root, rec.get("target"))},
                                    retrieved)]
        claims.extend(scored)
        lowest.append(min(scored, default=0.0))
    out = {"per_claim": suggest(claims, [])["positive"],
           "lowest_claim_per_review": suggest(lowest, [])["positive"],
           "reviews": len(lowest),
           "reviews_passing_current_bar": sum(s >= bar for s in lowest)}
    out.update(measure="entailment (per claim, the gate's own premises)", tenant=tenant,
               entailer=type(entailer).__name__, current=bar)
    return out


def main() -> int:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Measure the gate thresholds.")
    ap.add_argument("measure", choices=("relevance", "rerank", "entailment"))
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--root", help="the tenant's repo on this machine (all three measures)")
    ap.add_argument("--commits", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.measure == "relevance":
        result = relevance(args.tenant, args.root, args.commits)
    elif args.measure == "rerank":
        result = rerank(args.tenant, args.root, args.commits)
    else:
        result = entailment(args.tenant, args.root)
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
