"""Build a namespaced vector index from a codebase. Idempotent.

    python -m tools.ingest --tenant milesone
    python -m tools.ingest --tenant milesone --include app/auth app/celery
    python -m tools.ingest --tenant milesone --dry-run

Replaces a skeleton that raised `NotImplementedError` and described a plan that
was wrong for this corpus in two ways: it chunked 800/40 (fixed windows, which
cut functions in half -- see tools/chunking.py) and embedded with
`text-embedding-3-small`, a model this project has not used for three embedder
changes.

## The namespace comes from the tenant, never from a flag

`policy.codebase.repo`, resolved through the same `TenantStore` the graph uses
and the same `namespace_for` contract. An index built under a namespace the
graph would not ask for is an index the graph cannot see, and the failure would
look like "retrieval returns nothing" rather than "you indexed the wrong key".

## What is deliberately NOT indexed

Tests, migrations-as-data, vendored code and virtualenvs. Tests would flood the
corpus with assertions that read like the code they test, so a query about
`callback` retrieves `test_callback` -- a citation that grounds a review in an
expectation rather than in the implementation.

## Secrets

Every chunk passes through the InputGuard before it is embedded. Real source is
the first thing that will genuinely test T12, and an index is a durable copy: a
key embedded once stays retrievable and quotable until the index is rebuilt.
"""
from __future__ import annotations

import argparse
import fnmatch
import pathlib
import sys
import time

import config
from app.activity import WARNING, ActivityLogger, capture_library_logs, log_event
from app.live import IngestProgress
from tools.bm25 import BM25
from tools.bm25 import FILE as BM25_FILE
from tools.chunk_text import embed_text
from tools.chunking import chunk_file
from tools.codebase import codebase_root

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages", "dist", "build", ".tox",
    "staticfiles", "static", "media", ".idea", ".vscode", "htmlcov",
}
# Matched against the FILE NAME. Incomplete on its own -- see SKIP_PATH_PARTS.
SKIP_NAME_PARTS = ("test_", "_test", "conftest", "tests.py", "tests_")
# Matched against any PATH SEGMENT. The name check alone missed a whole class of
# test file in the real corpus: `AILabs/test/tests.py` contains neither "test_"
# nor "_test", so it sailed through and brought its fixtures with it -- fake
# users complete with emails and mobile numbers, which then showed up as 900+
# redactions. Excluding the directory is what actually works, because test
# layouts vary and file naming does not have to.
SKIP_PATH_PARTS = {"test", "tests", "testing", "fixtures", "factories"}
INDEX_SUFFIXES = {".py", ".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg"}
BATCH = 32          # embedding batch; keeps one request well inside any limit


def excluded(rel: str, patterns: list[str]) -> bool:
    """True if `rel` (a POSIX path under the repo root) matches a tenant exclude.

    A pattern matches the file itself or ANY of its parent directories, so
    `"Miscellaneous/User Migration Dump*"` excludes everything beneath that
    directory without the tenant having to know `**` syntax.

    Why tenants need this at all: the built-in skips know what a test or a
    virtualenv looks like in any repo, but not that one repo keeps user-data
    exports beside its code. Real repos do -- MilesOne's first dry run found
    migration reports full of user records, which pattern-based redaction cannot
    fully clean (it catches an email, not a name). An index is a durable copy,
    so the only complete fix for data that should not be there is not reading it.
    """
    parts = rel.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    return any(fnmatch.fnmatchcase(p, pat) for pat in patterns for p in prefixes)


def iter_files(root: pathlib.Path, include: list[str] | None,
               exclude: list[str] | None = None) -> list[pathlib.Path]:
    """Indexable files under `root` (or its `include` subpaths), sorted.

    Sorted so two runs over the same tree chunk in the same order -- part of what
    makes the fingerprint, and so "idempotent", checkable.
    """
    roots = [root / i for i in include] if include else [root]
    exclude = exclude or []
    found: list[pathlib.Path] = []
    for base in roots:
        if not base.exists():
            print(f"  ! {base} does not exist, skipping", file=sys.stderr)
            log_event("job", "ingest.include_missing", severity=WARNING,
                      message=f"{base} does not exist, skipped",
                      metadata={"path": str(base)})
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in INDEX_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if any(s in p.name for s in SKIP_NAME_PARTS):
                continue
            if any(part in SKIP_PATH_PARTS for part in p.parts):
                continue
            if exclude and excluded(p.relative_to(root).as_posix(), exclude):
                continue
            found.append(p)
    return sorted(found)


def build(tenant: str, include: list[str] | None, dry_run: bool,
          limit: int | None, surface: str = "code_review",
          root_override: str | None = None) -> int:
    """Chunk, redact, embed and write one tenant's namespace. Returns an exit code.

    The namespace and the exclusions come from the tenant's policy, never from a
    flag; only WHERE the repo lives on this machine may be overridden.
    """
    policy = config.get_tenant_store().policy(tenant, surface)
    codebase = policy.get("codebase") or {}
    namespace = codebase.get("repo")
    root_setting = codebase.get("root")
    if not namespace:
        raise SystemExit(
            f"tenant '{tenant}' declares no codebase.repo, so it has no retrieval "
            f"namespace. Retrieval must never be global (G1)."
        )
    # `codebase.root` is COMMITTED and shared across machines, so it cannot be
    # right everywhere: milesone's "/repos/..." is a container path, and the same
    # repo lives under ~/Dev on the Mac. CRA_CODEBASE_ROOT (or --root) overrides
    # it per machine, the same split the platform layer already uses -- shared
    # config names WHAT, local config says WHERE.
    root = codebase_root(codebase, root_override)
    if not root.exists():
        raise SystemExit(
            f"tenant '{tenant}' points codebase.root at {root_setting!r}, which does "
            f"not exist on this machine. Pass --root <path>, or set "
            f"CRA_CODEBASE_ROOT. The namespace still comes from the tenant."
        )

    print(f"tenant    : {tenant}")
    print(f"namespace : {namespace}")
    print(f"root      : {root}")

    # Live progress for `python -m app.monitor` -- a 10-minute embed that only
    # prints "\r embedded 3424/17622" is invisible to anyone not tailing a log.
    live = IngestProgress(tenant, namespace)
    try:
        # One activity record for the whole job: its ~550 embedding batches roll
        # up here (adapters/embed_lmstudio.py logs query embeddings only).
        with ActivityLogger("job", "ingest", tenant=tenant,
                            metadata={"namespace": namespace, "dry_run": dry_run}) as alog:
            code = _build(live, tenant, namespace, root, codebase, include, dry_run, limit)
            for key in ("files", "chunks", "redacted", "elapsed_s"):
                if live.data.get(key) is not None:
                    alog.add_metadata(key, live.data[key])
            return code
    except KeyboardInterrupt:
        live.end("interrupted")
        raise
    except Exception as e:
        # Recorded so the monitor shows WHY, then re-raised unchanged.
        live.end("error", error=f"{type(e).__name__}: {e}"[:300])
        raise


def _save_bm25(index, namespace: str) -> None:
    """Write the BM25 statistics beside the vectors, keyed by the new fingerprint.
    An index without a `sidecar` location simply has none; the retriever builds
    them in memory instead."""
    sidecar = getattr(index, "sidecar", None)
    if callable(sidecar):
        BM25.for_chunks(index.documents(namespace)).save(
            sidecar(namespace, BM25_FILE), index.stats(namespace).get("fingerprint", ""))


def _build(live, tenant, namespace, root, codebase, include, dry_run, limit) -> int:
    """The body of `build`, run under a live-progress record that marks failures."""
    exclude = list(codebase.get("index_exclude") or [])
    if exclude:
        print(f"exclude   : {exclude}")
    live.phase("scanning")
    files = iter_files(root, include, exclude)
    if limit:
        files = files[:limit]
    print(f"files     : {len(files)}")
    live.phase("chunking", files=len(files))

    chunks: list[dict] = []
    for f in files:
        chunks.extend(chunk_file(f, root))
    print(f"chunks    : {len(chunks)}")
    if not chunks:
        raise SystemExit("nothing to index")

    # Redact BEFORE embedding. An index is durable: a secret embedded once stays
    # retrievable and quotable until the whole namespace is rebuilt.
    live.phase("redacting", chunks=len(chunks))
    guard = config.get_input_guard()
    redacted_total = 0
    for c in chunks:
        c["text"], report = guard.sanitize(c["text"], tenant=tenant)
        redacted_total += report.get("total_redactions", 0)
    print(f"redacted  : {redacted_total} value(s) removed before embedding")

    if dry_run:
        by_kind: dict[str, int] = {}
        for c in chunks:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        print(f"kinds     : {by_kind}")
        print("dry run - nothing embedded or written")
        live.end("finished", dry_run=True, redacted=redacted_total)
        return 0

    embedder = config.get_embedder()
    index = config.get_index()
    print(f"embedder  : {embedder}")
    live.phase("embedding", redacted=redacted_total, model=embedder.model)

    started = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vectors = embedder.embed([embed_text(c) for c in batch], kind="passage")
        for c, v in zip(batch, vectors, strict=True):
            c["vector"] = v
        done = min(i + BATCH, len(chunks))
        print(f"\r  embedded {done}/{len(chunks)}", end="", flush=True)
        live.progress(done, len(chunks))
    print()

    # The manifest is what lets the retriever refuse a query from a different
    # embedder (adapters/retriever_indexed.py). `root` is deliberately NOT
    # recorded: it is a per-machine path, and the index must stay meaningful
    # when the same corpus is built on the other machine.
    manifest = {"tenant": tenant, "model": embedder.model, "dims": embedder.dims,
                "include": include or [], "exclude": exclude}
    live.phase("writing")
    written = index.upsert(namespace, chunks, manifest=manifest)
    # The BM25 statistics, saved beside the vectors so a review loads them
    # instead of re-tokenising the whole corpus in every new process.
    live.phase("bm25")
    _save_bm25(index, namespace)
    print(f"indexed   : {written} chunks in {time.time() - started:.1f}s")
    print(f"stats     : {index.stats(namespace)}")
    live.end("finished", chunks=written)
    return 0


def main() -> None:
    """CLI entry point; see the module docstring for usage."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--surface", default="code_review")
    ap.add_argument("--include", nargs="*",
                    help="paths relative to the codebase root; omit for the whole repo")
    ap.add_argument("--root", help="override codebase.root for THIS machine")
    ap.add_argument("--limit", type=int, help="cap the file count (for a quick trial)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and redact, but do not embed or write")
    args = ap.parse_args()
    capture_library_logs()
    raise SystemExit(build(args.tenant, args.include, args.dry_run,
                           args.limit, args.surface, args.root))


if __name__ == "__main__":
    main()
