"""A namespaced, on-disk vector index. No server, no new service.

Chosen over pgvector or Pinecone deliberately. There is no Postgres on either
development machine (docs/DECISIONS.md section 1), the project is run locally
only, and a retrieval layer that needs a service to be up would have the same
problem `doctor` and the gate suite are written to avoid. numpy is the only
dependency, declared in `requirements.txt`.

## Namespacing is structural, not a parameter

Each codebase gets its own directory and its own matrix. G1 -- never retrieve
across codebases -- is then a property of the filesystem layout rather than of a
caller remembering to pass a filter. `search()` takes `namespace` first and has
no default; there is no way to spell "search everything".

    <CRA_DATA_DIR>/index/<namespace>/vectors.npy     float32 [n, dims]
    <CRA_DATA_DIR>/index/<namespace>/chunks.jsonl    one record per row, same order
    <CRA_DATA_DIR>/index/<namespace>/manifest.json   which embedder, how many, when

Vectors and metadata are separate files because the vectors are read as a block
for the dot product while the metadata is only touched for the rows that win.

## The manifest, and why an index must say what built it

A vector means nothing outside the space of the model that produced it. Without
a record of that model, changing `EMBEDDING_MODEL` would leave every existing
index answering queries with cosines that look entirely normal and are noise --
and every `min_relevance_score` would be thresholding that noise. The manifest
lets the retriever refuse instead (adapters/retriever_indexed.py).

`manifest.json` is also what `doctor` reads to report which namespaces are
built, because doctor is stdlib-only and cannot load a `.npy`.

## Cosine, and why the vectors are stored normalised

Rows are L2-normalised at upsert, so a similarity is a plain dot product and the
score is a real cosine in [-1, 1]. That matters beyond speed: `min_relevance_score`
is meant to be a calibrated similarity, and T21 is the record of what happens when
the gate is handed a number that only looks like one.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import time

import numpy as np

from registry import adapter

MANIFEST = "manifest.json"


def _root() -> pathlib.Path:
    """`<CRA_DATA_DIR>/index`, beside audit records -- and just as sensitive."""
    return pathlib.Path(os.getenv("CRA_DATA_DIR", "./var")).expanduser() / "index"


def namespace_dirname(namespace: str) -> str:
    """Map a logical namespace to one safe directory name.

    Module-level so `app/doctor.py` can locate an index with the stdlib alone,
    using exactly the mapping the index writes with. Two copies of this rule
    would eventually disagree, and doctor would then report a built index as
    missing (or worse, the reverse).

    The namespace is a repo path (`nodes/_surface.py::namespace_for` returns
    `policy.codebase.repo`, e.g. "Org/Some-Backend"), so it contains a
    separator. It is flattened rather than nested: a nested layout would make
    "Org" a directory shared by every repo under that org, and a traversal bug
    there would breach exactly the isolation this class provides.

    Encoded rather than stripped, so two different repos cannot collide into one
    directory and quietly share an index -- a G1 breach that looks like a working
    index.
    """
    if not namespace or not namespace.strip():
        raise ValueError("namespace is required: retrieval must never be global (G1)")
    safe = re.sub(r"[^A-Za-z0-9._-]", "__", namespace.strip())
    if safe.startswith(".") or safe in {"", ".", ".."}:
        raise ValueError(f"unsafe namespace {namespace!r}")
    return safe


def fingerprint(chunks: list[dict]) -> str:
    """Content identity of a namespace: the sorted chunk ids, hashed.

    Chunk ids already embed a content hash (tools/chunking.py), so re-ingesting
    unchanged code yields the same fingerprint -- which is what "idempotent"
    should be checkable against, and what an in-process cache keys on.
    """
    h = hashlib.sha256()
    for cid in sorted(c["id"] for c in chunks):
        h.update(cid.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


@adapter("index", "local")
class LocalVectorIndex:
    """Flat exact search. Exact rather than approximate, on purpose.

    A repo-sized corpus is tens of thousands of chunks; a brute-force dot product
    over that is milliseconds in numpy. ANN indexes trade recall for speed at a
    scale this does not reach, and recall is the thing grounding depends on.
    """

    def __init__(self, root: pathlib.Path | None = None):
        self.root = pathlib.Path(root) if root else _root()
        # namespace -> (vectors.npy mtime_ns, vectors, metas, id->row)
        self._cache: dict[str, tuple] = {}

    def _dir(self, namespace: str) -> pathlib.Path:
        return self.root / namespace_dirname(namespace)

    # ── write ────────────────────────────────────────────────────────────────
    def upsert(self, namespace: str, chunks: list[dict],
               manifest: dict | None = None) -> int:
        """Replace a namespace's contents. Whole-namespace, not incremental.

        Ingest is defined as idempotent, and a partial update is how an index
        drifts from the source it claims to describe: a deleted file's chunks
        would linger and stay citable forever. Rebuilding is cheap at this scale
        and leaves no way to be subtly stale.

        Written to a temporary directory and renamed, so a crash mid-write leaves
        the previous index intact rather than a half-written one that still loads.
        """
        ids = [c["id"] for c in chunks]
        if len(set(ids)) != len(ids):
            # Ids carry path, name, line and content hash, so a duplicate means
            # the chunker emitted the same text twice from one place. Refusing is
            # cheaper than discovering later that `similarities()` silently
            # resolved a citation to whichever copy came last.
            dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
            raise ValueError(f"duplicate chunk ids, e.g. {dupes}")

        target = self._dir(namespace)
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        vectors = np.asarray([c["vector"] for c in chunks], dtype="float32")
        if vectors.size:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0          # a zero vector stays zero, not NaN
            vectors = vectors / norms
        np.save(tmp / "vectors.npy", vectors)

        with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                meta = {k: v for k, v in c.items() if k != "vector"}
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

        record = {
            **(manifest or {}),
            "namespace": namespace,
            "chunks": len(chunks),
            "dims": int(vectors.shape[1]) if vectors.ndim == 2 and len(vectors) else 0,
            "files": len({c.get("path") for c in chunks if c.get("path")}),
            "fingerprint": fingerprint(chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (tmp / MANIFEST).write_text(json.dumps(record, indent=2), encoding="utf-8")

        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
        self._cache.pop(namespace, None)
        return len(chunks)

    # ── read ─────────────────────────────────────────────────────────────────
    def _load(self, namespace: str):
        """(vectors, metas, id->row), cached until the namespace is rebuilt.

        Re-reading a .npy and parsing tens of thousands of JSON lines on every
        query would put a disk read on the critical path of every diagnosis. The
        cache keys on the file's mtime, so a rebuild by another process (ingest
        runs as its own command) is picked up without a restart.
        """
        d = self._dir(namespace)
        vec_path, meta_path = d / "vectors.npy", d / "chunks.jsonl"
        if not vec_path.exists():
            self._cache.pop(namespace, None)
            return None, [], {}
        # inode + size + mtime: a rebuild renames a NEW file into place, so its
        # inode changes even if two writes land inside one timestamp tick.
        st = vec_path.stat()
        stamp = (st.st_ino, st.st_size, st.st_mtime_ns)
        hit = self._cache.get(namespace)
        if hit and hit[0] == stamp:
            return hit[1], hit[2], hit[3]
        vectors = np.load(vec_path)
        metas = [json.loads(line) for line in
                 meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(metas) != len(vectors):
            # The two files are written together and renamed together, so this
            # only happens if someone edited one by hand. A misaligned row would
            # attach one chunk's score to another chunk's text -- a citation
            # that points at code the score was never computed for.
            raise ValueError(
                f"index '{namespace}' is corrupt: {len(vectors)} vectors but "
                f"{len(metas)} chunk records. Re-run tools.ingest.")
        by_id = {m["id"]: i for i, m in enumerate(metas)}
        self._cache[namespace] = (stamp, vectors, metas, by_id)
        return vectors, metas, by_id

    @staticmethod
    def _unit(vector: list[float]):
        q = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(q)) or 1.0
        return q / norm

    def search(self, namespace: str, vector: list[float], k: int = 8) -> list[dict]:
        """Top-k by cosine. Returns [] for an unbuilt namespace rather than raising.

        An empty result is the honest answer to "what does this codebase contain"
        when nothing has been ingested, and the gate handles it correctly already:
        no citations means `groundedness_ok` is False and the run escalates.
        `doctor` reports unbuilt namespaces, so that state is not silent.
        """
        vectors, metas, _ = self._load(namespace)
        if vectors is None or not len(vectors):
            return []
        if vectors.shape[1] != len(vector):
            raise ValueError(
                f"query vector has {len(vector)} dims but index '{namespace}' "
                f"holds {vectors.shape[1]}: a different embedder built it")
        sims = vectors @ self._unit(vector)          # rows are already unit length
        top = np.argsort(-sims)[:k]
        out = []
        for rank, i in enumerate(top):
            meta = dict(metas[i])
            meta["score"] = round(float(sims[i]), 6)
            meta["rank"] = rank
            out.append(meta)
        return out

    def similarities(self, namespace: str, vector: list[float],
                     ids: list[str]) -> dict[str, float]:
        """Cosine for specific chunks. Unknown ids are omitted, not zeroed.

        Omitted rather than 0.0 so a caller cannot mistake "this id is not in
        the index" for "this id is in the index and irrelevant".
        """
        vectors, _, by_id = self._load(namespace)
        if vectors is None:
            return {}
        rows = [(cid, by_id[cid]) for cid in ids if cid in by_id]
        if not rows:
            return {}
        sims = vectors[[r for _, r in rows]] @ self._unit(vector)
        return {cid: round(float(s), 6) for (cid, _), s in zip(rows, sims, strict=True)}

    def documents(self, namespace: str) -> list[dict]:
        """Every chunk's metadata, in row order. For BM25; never mutated here."""
        _, metas, _ = self._load(namespace)
        return metas

    def namespaces(self) -> list[str]:
        """Built namespaces, as directory names. A half-written `.tmp` is never one."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and not p.name.endswith(".tmp"))

    def stats(self, namespace: str) -> dict:
        """The manifest, plus `built`. Read from disk only -- no numpy needed."""
        path = self._dir(namespace) / MANIFEST
        if not path.exists():
            return {"namespace": namespace, "chunks": 0, "dims": 0, "built": False}
        return {**json.loads(path.read_text(encoding="utf-8")), "built": True}
