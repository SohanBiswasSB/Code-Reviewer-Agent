"""T10: chunking, BM25, the namespaced index, the indexed retriever, and ingest.

Everything here runs the REAL index, BM25, fusion and gate-score code. Only the
embedder is faked, and the fake is a deterministic hashed bag-of-words rather
than random vectors: it has the one property these tests depend on, that text
sharing words with the query scores higher than text that does not. A random
fake would make the separation assertions pass or fail by luck -- exactly the
"convenient fixture" this project's recurring lesson is about.
"""
from __future__ import annotations

import hashlib
import json

import pytest

np = pytest.importorskip("numpy")

from adapters.platforms import base  # noqa: E402
from adapters.retriever_indexed import IndexedRetriever  # noqa: E402
from adapters.vector_index_local import (  # noqa: E402
    LocalVectorIndex,
    fingerprint,
    namespace_dirname,
)
from tools.bm25 import BM25, tokenize  # noqa: E402
from tools.chunking import MAX_CHARS, chunk_python, chunk_text  # noqa: E402

DIMS = 64


class FakeEmbedder:
    """Hashed bag-of-words. Deterministic, shared-words -> higher cosine."""

    def __init__(self, model: str = base.EMBEDDING_MODEL, dims: int = DIMS):
        self.model, self.dims = model, dims
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts, *, kind="passage"):
        self.calls.append((kind, len(texts)))
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
            out.append(v)
        return out


def _chunk(cid: str, text: str, path: str = "app/x.py", name: str = "f") -> dict:
    return {"id": cid, "text": text, "path": path, "name": name, "kind": "function",
            "start_line": 1, "end_line": 2}


def _build(index, namespace, chunks, embedder=None, model=None):
    emb = embedder or FakeEmbedder()
    for c in chunks:
        c["vector"] = emb.embed([c["text"]])[0]
    return index.upsert(namespace, chunks,
                        manifest={"model": model or emb.model, "dims": emb.dims})


CORPUS = [
    _chunk("oauth", "def oauth_callback(request): state missing returns 500 in auth middleware",
           "app/auth/oauth.py", "oauth_callback"),
    _chunk("refresh", "def refresh_token(user): race causes intermittent 401 after login",
           "app/auth/tokens.py", "refresh_token"),
    _chunk("invoice", "def render_invoice(order): pdf layout totals currency rounding",
           "app/billing/invoice.py", "render_invoice"),
    _chunk("webhook", "def verify_webhook(sig): hmac secret compare digest",
           "app/payments/webhooks.py", "verify_webhook"),
]


@pytest.fixture
def index(tmp_path):
    return LocalVectorIndex(root=tmp_path / "index")


@pytest.fixture
def built(index):
    _build(index, "org/backend", [dict(c) for c in CORPUS])
    return index


# ── chunking ─────────────────────────────────────────────────────────────────
SRC = '''
import os

TIMEOUT = 30
RETRY_POLICY = {"attempts": 3, "backoff_seconds": 2}


@login_required
def view(request):
    """A view."""
    return render(request, "x.html", {"timeout": TIMEOUT})


class Thing:
    def method(self):
        return os.getenv("THING_SETTING", "a fairly long default value")
'''


def test_python_is_chunked_by_definition_not_by_window():
    chunks = chunk_python(SRC, "app/views.py")
    by_name = {c["name"]: c for c in chunks}
    assert {"view", "Thing", "<module>"} <= set(by_name)
    # The decorator belongs to the function it decorates.
    assert by_name["view"]["text"].lstrip().startswith("@login_required")
    assert by_name["Thing"]["kind"] == "class"
    # Module-level code is kept, without the bodies already chunked.
    assert "TIMEOUT = 30" in by_name["<module>"]["text"]
    assert "def view" not in by_name["<module>"]["text"]


def test_chunk_ids_are_stable_and_change_with_content():
    a = chunk_python(SRC, "app/views.py")
    b = chunk_python(SRC, "app/views.py")
    assert [c["id"] for c in a] == [c["id"] for c in b]
    c = chunk_python(SRC.replace("x.html", "y.html"), "app/views.py")
    view_ids = {x["id"] for x in a if x["name"] == "view"}
    assert view_ids.isdisjoint({x["id"] for x in c if x["name"] == "view"})


def test_an_oversized_function_with_no_blank_lines_is_still_split():
    """The embedder truncates silently, so an over-long chunk loses its tail."""
    body = "\n".join(f"    x{i} = compute_something_long({i})" for i in range(200))
    chunks = chunk_python(f"def huge():\n{body}\n", "app/huge.py")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    assert all(c["name"].startswith("huge#") for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_unparseable_python_falls_back_to_text_chunks():
    chunks = chunk_python("def broken(:\n" + "this is not python at all " * 5, "app/bad.py")
    assert chunks and all(c["kind"] == "text" for c in chunks)


def test_text_chunks_skip_fragments():
    assert chunk_text("tiny", "README.md") == []


# ── BM25 ─────────────────────────────────────────────────────────────────────
def test_tokenize_splits_identifiers_so_prose_reaches_code():
    assert tokenize("getUserToken") == ["get", "user", "token"]
    assert tokenize("get_user_token") == ["get", "user", "token"]


def test_bm25_weights_rare_terms_above_common_ones():
    """No-IDF word overlap let `self` count as much as `oauth`. BM25 must not."""
    docs = [tokenize(t) for t in ["self return self value", "self oauth callback",
                                  "self return none", "self return list"]]
    bm = BM25(docs)
    top = bm.top("self oauth", 4)
    assert top[0][0] == 1


def test_bm25_returns_nothing_for_unrelated_queries():
    bm = BM25([tokenize(c["text"]) for c in CORPUS])
    assert bm.top("zzzz qqqq", 3) == []


# ── the index ────────────────────────────────────────────────────────────────
def test_search_returns_real_cosines(built):
    q = FakeEmbedder().embed(["oauth callback state missing"])[0]
    hits = built.search("org/backend", q, 4)
    assert hits[0]["id"] == "oauth"
    assert all(-1.0 <= h["score"] <= 1.0 for h in hits)
    assert hits[0]["score"] > hits[-1]["score"]


def test_namespaces_are_isolated(index):
    """G1 as a property of the layout: one namespace never returns another's rows."""
    _build(index, "org/a", [_chunk("a1", "oauth callback in project a")])
    _build(index, "org/b", [_chunk("b1", "oauth callback in project b")])
    q = FakeEmbedder().embed(["oauth callback"])[0]
    assert {h["id"] for h in index.search("org/a", q, 10)} == {"a1"}
    assert {h["id"] for h in index.search("org/b", q, 10)} == {"b1"}


def test_namespace_mapping_refuses_global_and_traversal():
    with pytest.raises(ValueError, match="G1"):
        namespace_dirname("")
    with pytest.raises(ValueError, match="unsafe"):
        namespace_dirname("..")
    # Distinct repos must never collapse into one directory.
    assert namespace_dirname("org/repo") != namespace_dirname("org-repo")
    assert "/" not in namespace_dirname("org/repo")


def test_unbuilt_namespace_searches_empty(index):
    assert index.search("org/never-built", [1.0] * DIMS, 5) == []
    assert index.stats("org/never-built")["built"] is False


def test_upsert_replaces_the_whole_namespace(built):
    """A deleted file's chunks must not linger and stay citable."""
    _build(built, "org/backend", [dict(CORPUS[0])])
    assert {d["id"] for d in built.documents("org/backend")} == {"oauth"}


def test_duplicate_chunk_ids_are_refused(index):
    with pytest.raises(ValueError, match="duplicate"):
        _build(index, "org/x", [_chunk("same", "one text"), _chunk("same", "two text")])


def test_manifest_records_the_embedder_and_a_stable_fingerprint(built):
    s = built.stats("org/backend")
    assert s["built"] and s["model"] == base.EMBEDDING_MODEL and s["dims"] == DIMS
    assert s["chunks"] == len(CORPUS) and s["files"] == len(CORPUS)
    assert s["fingerprint"] == fingerprint(CORPUS)


def test_similarities_omit_unknown_ids(built):
    q = FakeEmbedder().embed(["oauth"])[0]
    sims = built.similarities("org/backend", q, ["oauth", "no-such-id"])
    assert set(sims) == {"oauth"}


def test_a_misaligned_index_is_refused_not_misread(built, tmp_path):
    d = built.root / namespace_dirname("org/backend")
    lines = (d / "chunks.jsonl").read_text().splitlines()
    (d / "chunks.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        LocalVectorIndex(root=built.root).search("org/backend", [1.0] * DIMS, 3)


def test_wrong_width_query_is_refused(built):
    with pytest.raises(ValueError, match="dims"):
        built.search("org/backend", [1.0] * (DIMS + 1), 3)


# ── the indexed retriever ────────────────────────────────────────────────────
def test_retriever_refuses_without_a_namespace(built):
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    with pytest.raises(ValueError, match="G1"):
        r.retrieve("oauth", k=3, namespace=None)


def test_retriever_returns_nothing_for_an_unbuilt_namespace(index):
    emb = FakeEmbedder()
    r = IndexedRetriever(embedder=emb, index=index)
    assert r.retrieve("oauth", k=3, namespace="org/unbuilt") == []
    assert emb.calls == [], "no point embedding a query with nothing to search"


def test_retriever_refuses_an_index_from_a_different_embedder(index):
    """Cosines across two embedding spaces are noise the gate would threshold."""
    _build(index, "org/backend", [dict(c) for c in CORPUS], model="some-other-embedder")
    r = IndexedRetriever(embedder=FakeEmbedder(), index=index)
    with pytest.raises(ValueError, match="Cosines across"):
        r.retrieve("oauth", k=3, namespace="org/backend")


def test_retriever_embeds_the_query_as_a_query(built):
    emb = FakeEmbedder()
    IndexedRetriever(embedder=emb, index=built).retrieve("oauth", k=2, namespace="org/backend")
    assert emb.calls == [("query", 1)]


def test_score_is_the_cosine_not_the_fusion_value(built):
    """T21's contract, kept: the gate reads a magnitude, never a rank artefact."""
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    hits = r.retrieve("oauth callback state missing", k=3, namespace="org/backend")
    assert hits[0]["id"] == "oauth"
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert all(h["score"] != h["fusion"] for h in hits)
    q = FakeEmbedder().embed(["oauth callback state missing"])[0]
    expected = built.similarities("org/backend", q, [h["id"] for h in hits])
    assert {h["id"]: h["score"] for h in hits} == expected


def test_on_topic_outscores_off_topic(built):
    """The property a calibrated threshold needs. If these were equal, the
    relevance score would carry no information and the gate would be inert."""
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    on = r.retrieve("refresh token race 401 login", k=1, namespace="org/backend")
    off = r.retrieve("banana bread walnuts recipe", k=1, namespace="org/backend")
    assert on[0]["id"] == "refresh"
    assert on[0]["score"] > (off[0]["score"] if off else 0.0)


def test_a_keyword_only_hit_still_carries_a_real_cosine(index):
    """A chunk BM25 finds but dense does not rank must still reach the gate with
    its own cosine -- never a missing score that passes by default.

    Forced deterministically: the dense side is restricted to a window that
    excludes the target, so the target can only arrive through BM25.
    """
    filler = [_chunk(f"f{i}", f"generic helper number {i} value", f"app/f{i}.py", f"f{i}")
              for i in range(30)]
    target = _chunk("rare", "zygomorphic flux capacitor", "app/rare.py", "rare")
    _build(index, "org/x", filler + [target])

    class DenseBlind(LocalVectorIndex):
        def search(self, namespace, vector, k=8):
            return [h for h in super().search(namespace, vector, 1000) if h["id"] != "rare"][:k]

    blind = DenseBlind(root=index.root)
    r = IndexedRetriever(embedder=FakeEmbedder(), index=blind)
    hits = r.retrieve("zygomorphic", k=3, namespace="org/x")
    rare = next(h for h in hits if h["id"] == "rare")
    q = FakeEmbedder().embed(["zygomorphic"])[0]
    assert rare["score"] == blind.similarities("org/x", q, ["rare"])["rare"]
    assert rare["text"] == "zygomorphic flux capacitor"


def test_bm25_reaches_a_chunk_through_its_path(built):
    """The path is indexed for BM25, so a query naming a module finds it."""
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    hits = r.retrieve("payments webhooks", k=2, namespace="org/backend")
    assert "webhook" in {h["id"] for h in hits}


def test_hits_carry_location_for_the_approver(built):
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    hit = r.retrieve("oauth callback", k=1, namespace="org/backend")[0]
    assert hit["path"] == "app/auth/oauth.py" and hit["name"] == "oauth_callback"


def test_the_bm25_cache_follows_a_rebuild(built):
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built)
    r.retrieve("oauth", k=1, namespace="org/backend")
    _build(built, "org/backend", [_chunk("new", "brand new quokka module")])
    assert r.retrieve("quokka", k=1, namespace="org/backend")[0]["id"] == "new"


# ── wiring ───────────────────────────────────────────────────────────────────
def test_the_default_retriever_is_the_real_index(monkeypatch):
    """The demo used to be the default, which refused every codebase tenant."""
    import config
    monkeypatch.delenv("RETRIEVER", raising=False)
    assert isinstance(config.get_retriever(), IndexedRetriever)
    assert config.registered()["selected"]["retriever"] == "indexed"


# ── ingest ───────────────────────────────────────────────────────────────────
@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "auth.py").write_text(
        'import os\n\n'
        'AWS_KEY = "AKIAABCDEFGHIJKLMNOP"\n'
        'REGION = os.getenv("AWS_REGION", "ap-south-1")\n\n\n'
        "def oauth_callback(request):\n"
        "    if 'state' not in request.GET:\n"
        "        return HttpResponseBadRequest('missing state parameter')\n"
        "    return redirect('/home')\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_auth.py").write_text(
        "def test_callback():\n    assert oauth_callback(fake_request()) is not None\n")
    (root / "app" / "test").mkdir()
    (root / "app" / "test" / "tests.py").write_text(
        "def fixture_user():\n    return {'email': 'someone@example.com', 'name': 'x'}\n")
    return root


def _ingest(monkeypatch, repo, embedder):
    import config
    from tools import ingest
    monkeypatch.setattr(config, "get_embedder", lambda: embedder)
    return ingest.build("milesone", None, dry_run=False, limit=None, root_override=str(repo))


def test_ingest_builds_the_tenants_namespace(monkeypatch, repo, capsys):
    emb = FakeEmbedder()
    assert _ingest(monkeypatch, repo, emb) == 0
    idx = LocalVectorIndex()
    stats = idx.stats("MilesEducation-Tech/Miles-USP-Backend")
    assert stats["built"] and stats["tenant"] == "milesone"
    assert stats["model"] == emb.model
    paths = {d["path"] for d in idx.documents("MilesEducation-Tech/Miles-USP-Backend")}
    assert paths == {"app/auth.py"}, "tests and test/ directories must not be indexed"
    assert all(kind == "passage" for kind, _ in emb.calls)


def test_ingest_reports_live_progress(monkeypatch, repo, tmp_path):
    _ingest(monkeypatch, repo, FakeEmbedder())
    snap = json.loads((tmp_path / "var" / "live" / "ingest-milesone.json").read_text())
    assert snap["status"] == "finished" and snap["done"] == snap["total"] > 0
    assert snap["namespace"] == "MilesEducation-Tech/Miles-USP-Backend"


def test_a_failed_ingest_is_marked_failed(monkeypatch, repo, tmp_path):
    class Broken(FakeEmbedder):
        def embed(self, texts, *, kind="passage"):
            raise ConnectionError("embedder went away")
    with pytest.raises(ConnectionError):
        _ingest(monkeypatch, repo, Broken())
    snap = json.loads((tmp_path / "var" / "live" / "ingest-milesone.json").read_text())
    assert snap["status"] == "error" and "embedder went away" in snap["error"]


def test_ingest_redacts_before_anything_is_stored(monkeypatch, repo):
    """An index is durable: a secret embedded once stays citable until rebuilt."""
    _ingest(monkeypatch, repo, FakeEmbedder())
    stored = json.dumps(LocalVectorIndex().documents("MilesEducation-Tech/Miles-USP-Backend"))
    assert "AWS_REGION" in stored, "the secret's chunk must be indexed, or this proves nothing"
    assert "AKIAABCDEFGHIJKLMNOP" not in stored
    assert "REDACTED" in stored


def test_ingest_is_idempotent(monkeypatch, repo):
    _ingest(monkeypatch, repo, FakeEmbedder())
    first = LocalVectorIndex().stats("MilesEducation-Tech/Miles-USP-Backend")["fingerprint"]
    _ingest(monkeypatch, repo, FakeEmbedder())
    second = LocalVectorIndex().stats("MilesEducation-Tech/Miles-USP-Backend")["fingerprint"]
    assert first == second


def test_ingest_embeds_location_but_stores_the_source(monkeypatch, repo):
    from tools.ingest import embed_text
    chunk = {"path": "app/auth.py", "name": "oauth_callback", "text": "def oauth_callback(): ..."}
    assert embed_text(chunk).startswith("app/auth.py :: oauth_callback\n")
    _ingest(monkeypatch, repo, FakeEmbedder())
    docs = LocalVectorIndex().documents("MilesEducation-Tech/Miles-USP-Backend")
    assert docs and all(not d["text"].startswith("app/") for d in docs)


def test_tenant_excludes_match_a_file_or_any_parent_directory():
    from tools.ingest import excluded
    pats = ["Misc/User Migration Dump*", "*.secret.md"]
    assert excluded("Misc/User Migration Dump V3/_full_user_report.txt", pats)
    assert excluded("docs/keys.secret.md", pats)
    assert not excluded("Misc/Documentation/runbook.md", pats)
    assert not excluded("app/views.py", [])


def test_ingest_honours_the_tenants_index_exclude(monkeypatch, repo, tmp_path):
    from tools.ingest import iter_files
    (repo / "dumps").mkdir()
    (repo / "dumps" / "users.txt").write_text("user records " * 20)
    rels = {p.relative_to(repo).as_posix() for p in iter_files(repo, None, ["dumps"])}
    assert rels == {"app/auth.py"}


def test_ingest_refuses_a_tenant_without_a_codebase(monkeypatch, repo):
    from tools import ingest
    with pytest.raises(SystemExit, match="G1"):
        ingest.build("_default", None, dry_run=True, limit=None, root_override=str(repo))


# ── doctor ───────────────────────────────────────────────────────────────────
def _doctor_index_lines(monkeypatch):
    from app import doctor
    monkeypatch.setenv("RETRIEVER", "indexed")
    r = doctor.Report()
    doctor._check_indexes(r)
    return {check: (status, detail) for status, check, detail in r.lines}


def test_doctor_warns_about_an_unbuilt_index(monkeypatch):
    lines = _doctor_index_lines(monkeypatch)
    assert lines["index: milesone"][0] == "WARN"
    assert "tools.ingest --tenant milesone" in lines["index: milesone"][1]


def test_doctor_fails_an_index_built_by_another_embedder(monkeypatch):
    _build(LocalVectorIndex(), "MilesEducation-Tech/Miles-USP-Backend",
           [dict(c) for c in CORPUS], model="retired-embedder")
    assert _doctor_index_lines(monkeypatch)["index: milesone"][0] == "FAIL"


def test_doctor_reports_a_good_index(monkeypatch):
    _build(LocalVectorIndex(), "MilesEducation-Tech/Miles-USP-Backend", [dict(c) for c in CORPUS])
    status, detail = _doctor_index_lines(monkeypatch)["index: milesone"]
    assert status == "OK  " and f"{len(CORPUS)} chunks" in detail


# ── saved BM25 statistics ─────────────────────────────────────────────────────
def test_saved_bm25_round_trips_and_refuses_another_index(tmp_path):
    docs = [tokenize(t) for t in ["self oauth callback", "self return none", "getUserToken"]]
    bm = BM25(docs)
    path = tmp_path / "bm25.npz"
    assert bm.save(path, "fp-1")
    loaded = BM25.load(path, "fp-1")
    assert loaded.top("oauth user token", 3) == bm.top("oauth user token", 3)
    assert BM25.load(path, "fp-2") is None, "statistics of a different index"
    assert BM25.load(tmp_path / "missing.npz", "fp-1") is None


def test_a_new_process_loads_bm25_instead_of_rebuilding_it(built, monkeypatch):
    """Every CLI review is a new process; the first one saves, the next loads."""
    IndexedRetriever(embedder=FakeEmbedder(), index=built).retrieve(
        "oauth", k=1, namespace="org/backend")
    assert built.sidecar("org/backend", "bm25.npz").exists()

    def no_rebuild(*a, **k):
        raise AssertionError("rebuilt BM25 although a saved copy matched")
    monkeypatch.setattr(BM25, "for_chunks", classmethod(no_rebuild))
    hit = IndexedRetriever(embedder=FakeEmbedder(), index=built).retrieve(
        "oauth", k=1, namespace="org/backend")[0]
    assert hit["id"] == "oauth"
