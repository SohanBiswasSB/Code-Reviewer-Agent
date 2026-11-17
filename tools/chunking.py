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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _Span:
    """A named, line-ranged region of one file that becomes one or more chunks.

    Grouped so the chunk builder takes one value instead of five loose
    arguments, and every piece of a split region carries the same location.
    """

    path: str
    name: str
    kind: str
    start: int
    end: int


def _span_chunks(span: _Span, pieces: list[str]) -> list[dict]:
    """One chunk per piece of `span`; a split region's names get `#n` suffixes.

    Each piece keeps the WHOLE region's line span, so a citation to any piece
    still opens the function it came from.
    """
    out = []
    for n, piece in enumerate(pieces):
        suffix = f"#{n + 1}" if len(pieces) > 1 else ""
        out.append({
            "id": _chunk_id(span.path, span.name + suffix, span.start, piece),
            "text": piece,
            "path": span.path,
            "name": span.name + suffix,
            "kind": span.kind,
            "start_line": span.start,
            "end_line": span.end,
        })
    return out


def _def_span(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, path: str) -> _Span:
    """The lines a top-level def or class occupies, its decorators included.

    Decorators carry real meaning in this codebase -- @task, @receiver,
    @login_required -- so they belong in the chunk of what they decorate.
    """
    start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
    if node.decorator_list:
        start = min(start, min(d.lineno for d in node.decorator_list))
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    return _Span(path, node.name, kind, start, end)


def _def_chunks(span: _Span, lines: list[str]) -> list[dict]:
    """The chunk(s) for a def's lines, split into `#n` pieces only when over
    MAX_CHARS; a body under MIN_CHARS is dropped."""
    body = "\n".join(lines[span.start - 1:span.end])
    if len(body.strip()) < MIN_CHARS:
        return []
    pieces = [body] if len(body) <= MAX_CHARS else _split_oversized(body, MAX_CHARS)
    return _span_chunks(span, pieces)


def _module_chunks(lines: list[str], covered: set[int], path: str) -> list[dict]:
    """Module-level code -- every line no top-level def or class covers -- as chunks.

    Imports, constants, settings and URL tables live here, so it is indexed too.
    """
    module_src = "\n".join(ln for i, ln in enumerate(lines, 1) if i not in covered).strip()
    if len(module_src) < MIN_CHARS:
        return []
    span = _Span(path, "<module>", "module", 1, len(lines))
    return _span_chunks(span, _split_oversized(module_src, MAX_CHARS))


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
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            span = _def_span(node, path)
            out += _def_chunks(span, lines)
            covered.update(range(span.start, span.end + 1))
    return out + _module_chunks(lines, covered, path)


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
