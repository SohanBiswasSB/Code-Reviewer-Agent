"""Chunk source code by STRUCTURE, not by character count.

`tools/ingest.py`'s skeleton said "chunk 800/40" -- fixed windows with overlap.
For prose that is fine. For code it is close to the worst option available:
a 800-character window cuts a function in half, so the half holding the signature
is separated from the half holding the bug, and a citation can point at a
fragment that means nothing on its own. Grounding is the entire safety story
here, and a citation the reviewer cannot interpret is not grounding.

So a chunk is a FUNCTION or a CLASS, found with the stdlib `ast` module. No
tree-sitter dependency: the target codebase is Python (921 files), `ast` is
exact rather than heuristic, and it costs nothing to import.

Three properties worth stating, because each is a decision:

**Every chunk carries its qualified name and line span.** A citation then reads
`app/auth/oauth.py:callback:18-34` instead of an opaque id, so an approver can
open exactly what the model looked at. `Finding.line` finally has something real
to be compared against.

**Oversized functions are split, but only as a last resort**, and the split
inherits the parent's qualified name so the pieces stay attributable.

**Module-level code is kept as its own chunk** -- imports, constants, Django
settings and URL tables live there, and in this codebase that is where routing
and configuration actually are.

Non-Python files fall back to paragraph-ish splitting on blank lines, which is
right for Markdown runbooks and prior RCAs -- part of the corpus
docs/MILESONE_ADAPTATION.md describes.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib

# Large enough to hold a real function, small enough that an embedding stays
# specific. Measured in characters because that is what the embedder charges for.
MAX_CHARS = 2000
MIN_CHARS = 40          # below this a chunk is punctuation or a stub


def _chunk_id(path: str, name: str, start: int, text: str) -> str:
    """Stable across re-ingests of unchanged code, so citations stay valid.

    Includes a content hash as well as the location: if a function moves, its id
    changes and a stale citation fails loudly at the gate rather than silently
    pointing at whatever now occupies those lines.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{path}:{name}:{start}:{digest}"


def _blocks(text: str, limit: int) -> list[str]:
    """Blank-line blocks, with any single block over `limit` cut on line breaks.

    Without the second step a 6,000-character function with no blank line in it
    went out as one chunk regardless of MAX_CHARS -- and the embedding server
    truncates over-long input SILENTLY, so the tail of that function (often where
    the bug is) would be unretrievable while looking indexed.
    """
    out: list[str] = []
    for block in text.split("\n\n"):
        if len(block) <= limit:
            out.append(block)
            continue
        buf, size = [], 0
        for line in block.split("\n"):
            if size + len(line) + 1 > limit and buf:
                out.append("\n".join(buf))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            out.append("\n".join(buf))
    return out


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split on blank lines, packing up to `limit`. Never mid-line."""
    parts, buf, size = [], [], 0
    for block in _blocks(text, limit):
        block_len = len(block) + 2
        if size + block_len > limit and buf:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(block)
        size += block_len
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [text[:limit]]


def chunk_python(source: str, path: str) -> list[dict]:
    """One chunk per top-level function/class, plus one for module-level code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A file that does not parse is still worth indexing -- it may be a
        # template or a py2 leftover -- but it gets the generic treatment.
        return chunk_text(source, path)

    lines = source.splitlines()
    out: list[dict] = []
    covered: set[int] = set()

    def emit(name: str, start: int, end: int, kind: str) -> None:
        body = "\n".join(lines[start - 1:end])
        if len(body.strip()) < MIN_CHARS:
            return
        pieces = [body] if len(body) <= MAX_CHARS else _split_oversized(body, MAX_CHARS)
        for n, piece in enumerate(pieces):
            suffix = f"#{n + 1}" if len(pieces) > 1 else ""
            out.append({
                "id": _chunk_id(path, name + suffix, start, piece),
                "text": piece,
                "path": path,
                "name": name + suffix,
                "kind": kind,
                "start_line": start,
                "end_line": end,
            })

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
            # Decorators carry real meaning in this codebase -- @task, @receiver,
            # @login_required -- so include them in the chunk they belong to.
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            emit(node.name, start, end, kind)
            covered.update(range(start, end + 1))

    module_lines = [ln for i, ln in enumerate(lines, 1) if i not in covered]
    module_src = "\n".join(module_lines).strip()
    if len(module_src) >= MIN_CHARS:
        pieces = _split_oversized(module_src, MAX_CHARS)
        for n, piece in enumerate(pieces):
            suffix = f"#{n + 1}" if len(pieces) > 1 else ""
            out.append({
                "id": _chunk_id(path, "<module>" + suffix, 1, piece),
                "text": piece, "path": path, "name": "<module>" + suffix,
                "kind": "module", "start_line": 1, "end_line": len(lines),
            })
    return out


def chunk_text(source: str, path: str) -> list[dict]:
    """Blank-line blocks for Markdown, YAML, runbooks and unparseable files."""
    out = []
    for n, piece in enumerate(_split_oversized(source, MAX_CHARS), 1):
        if len(piece.strip()) < MIN_CHARS:
            continue
        out.append({
            "id": _chunk_id(path, f"part{n}", n, piece),
            "text": piece, "path": path, "name": f"part{n}",
            "kind": "text", "start_line": 1, "end_line": source.count("\n") + 1,
        })
    return out


def chunk_file(path: pathlib.Path, repo_root: pathlib.Path) -> list[dict]:
    """Chunks for one file, with `path` relative to the repo root.

    Relative, so a chunk id means the same thing on both machines -- the repo
    lives at different absolute paths on each.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rel = str(path.relative_to(repo_root))
    if path.suffix == ".py":
        return chunk_python(source, rel)
    return chunk_text(source, rel)
