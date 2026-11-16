"""Measure the two gate thresholds instead of guessing them.

    python -m tools.calibrate --tenant masterclass --root ~/Dev/<repo> relevance
    python -m tools.calibrate --tenant masterclass entailment
    python -m tools.calibrate --tenant masterclass relevance --json

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

**entailment** (no model needed). For every finding in the tenant's audit
records, the entailer's score of the finding against the passages it cites,
split by whether those passages come from the finding's OWN file. Real model
output, real citations; the label is again a proxy, and the sample is only as
big as the number of reviews run.

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

import config
from adapters.audit_jsonl import iter_records
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


def entailment(tenant: str) -> dict:
    policy = config.get_tenant_store().policy(tenant, "code_review")
    namespace = (policy.get("codebase") or {}).get("repo")
    text = {m["id"]: m.get("text", "") for m in config.get_index().documents(namespace)}
    path_of = {cid: cid.split(":", 1)[0] for cid in text}
    entailer = config.get_entailer()
    own, other = [], []
    for rec in (r for r in iter_records() if r.get("tenant") == tenant):
        for f in (rec.get("diagnosis") or {}).get("findings") or []:
            cited = [c for c in f.get("citation_ids") or [] if c in text]
            if not cited or not f.get("file"):
                continue
            claim = f"{f.get('issue', '')} {f.get('suggestion', '')}"
            score = entailer.score(claim, " ".join(text[c] for c in cited))
            (own if any(path_of[c] == f["file"] for c in cited) else other).append(score)
    out = suggest(own, other)
    out.update(measure="entailment", tenant=tenant, entailer=type(entailer).__name__,
               current=policy.get("min_entailment_score", "default"))
    return out


def main() -> int:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Measure the gate thresholds.")
    ap.add_argument("measure", choices=("relevance", "entailment"))
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--root", help="the tenant's repo on this machine (relevance only)")
    ap.add_argument("--commits", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = relevance(args.tenant, args.root, args.commits) if args.measure == "relevance" \
        else entailment(args.tenant)
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
