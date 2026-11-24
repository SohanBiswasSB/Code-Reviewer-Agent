"""Local review: the git_local source, the review_file sink, and `app.review`.

The source runs against a REAL temporary git repository rather than a mocked
subprocess: what is under test is precisely how git output is parsed (paths with
spaces, ranges, truncation), and a mock would only test our idea of git.
"""
from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import config
from adapters.action_sinks import ReviewFileSink
from adapters.audit_jsonl import iter_records
from adapters.entail_lexical import LexicalEntailer
from adapters.llm_stub import StubLLM
from adapters.source_git_local import MAX_DIFF_CHARS, GitLocalSource, GitTargetError, pack_diff
from app import live, monitor, review
from nodes._render import render_checklist, render_review
from nodes.independent_review import _entailment, entailment_claims, independent_review
from nodes.prepush_checks import prepush_checks
from registry import adapter
from schemas.diagnosis import Citation, Diagnosis, Finding
from schemas.evidence import CodeChange
from surfaces.code_review import CodeReviewSurface


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture(name="repo")
def fixture_repo(tmp_path):
    """A real repository: one commit, then a staged change that includes a path with a space."""
    root = tmp_path / "repo"
    (root / "app" / "auth").mkdir(parents=True)
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "app" / "auth" / "oauth.py").write_text("def callback(request):\n    return 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first", "-m", "the body")
    (root / "app" / "auth" / "oauth.py").write_text(
        "def callback(request):\n    if 'state' not in request.GET:\n"
        "        return 400\n    return 1\n")
    (root / "Code Documentation").mkdir()
    (root / "Code Documentation" / "notes.md").write_text("spaces in the path\n")
    _git(root, "add", ".")
    return root


def gather(repo, target):
    """The git_local source's raw evidence for `target` in `repo`."""
    return GitLocalSource().gather(target, codebase={"root": str(repo), "name": "Demo"})


def test_staged_target_reads_the_index_not_the_worktree(repo):
    """`staged` reviews the index: an untracked worktree file never appears."""
    (repo / "unstaged.py").write_text("x = 1\n")          # untracked, must not appear
    raw = gather(repo, "staged")
    names = [f["filename"] for f in raw["files"]]
    assert names == ["Code Documentation/notes.md", "app/auth/oauth.py"]
    assert "'state' not in request.GET" in raw["diff"]
    assert "unstaged.py" not in raw["diff"]
    assert raw["title"] == "Staged changes in Demo"


def test_paths_with_spaces_survive(repo):
    """The Masterclass repo has `Catalogue/Code Documentation/`; whitespace
    splitting would turn one path into two non-existent ones."""
    assert "Code Documentation/notes.md" in [f["filename"] for f in gather(repo, "staged")["files"]]


def test_a_commit_target_uses_its_message(repo):
    """A commit target takes its title and body from the commit message, and only its files."""
    raw = gather(repo, "HEAD")
    assert raw["title"] == "first" and raw["body"] == "the body"
    assert [f["filename"] for f in raw["files"]] == ["app/auth/oauth.py"]


def test_range_targets(repo):
    """Both `a..b` and `a...b` ranges are read and titled with their commit count."""
    _git(repo, "commit", "-q", "-m", "second")
    for target in ("HEAD~1..HEAD", "HEAD~1...HEAD"):
        raw = gather(repo, target)
        assert "1 commit(s)" in raw["title"] and "second" in raw["body"]
        assert "app/auth/oauth.py" in [f["filename"] for f in raw["files"]]


@pytest.mark.parametrize("target", ["--output=/tmp/pwned", "-p", "HEAD..--all"])
def test_option_shaped_targets_are_refused_before_git_sees_them(repo, target):
    """Two layers stop these: this guard, and the `^{commit}` suffix plus
    SHA-only follow-up calls. The match pins the FIRST layer specifically, so
    removing it fails here instead of being silently covered by the second."""
    with pytest.raises(GitTargetError, match="option-shaped"):
        gather(repo, target)


def test_unknown_revisions_are_refused(repo):
    """An unknown revision fails at rev-parse, before any diff is attempted."""
    with pytest.raises(GitTargetError, match="rev-parse"):
        gather(repo, "no-such-branch")


def test_an_empty_change_is_refused_rather_than_reviewed(repo):
    """Nothing staged is an error: an empty diff must not be reviewed as if it were a change."""
    _git(repo, "commit", "-q", "-m", "second")
    with pytest.raises(GitTargetError, match="no changes"):
        gather(repo, "staged")


def test_no_checkout_is_a_clear_error(tmp_path):
    """A root that is not a git checkout names the setting to fix, CRA_CODEBASE_ROOT."""
    with pytest.raises(GitTargetError, match="CRA_CODEBASE_ROOT"):
        GitLocalSource().gather("staged", codebase={"root": str(tmp_path)})


def test_a_huge_diff_is_truncated_visibly(repo):
    """A diff over the budget is cut with a visible marker in diff and body, never silently."""
    (repo / "big.py").write_text("x = 'y'\n" * 20_000)
    _git(repo, "add", ".")
    raw = gather(repo, "staged")
    assert "[DIFF TRUNCATED" in raw["diff"] and "[DIFF TRUNCATED" in raw["body"]
    assert len(raw["diff"]) < MAX_DIFF_CHARS + 500


def test_the_surface_normalizes_it_unchanged(repo):
    """Same shape as the GitHub source, so routing works with no surface change."""
    ev = CodeReviewSurface().normalize(gather(repo, "staged"))
    assert ev["touched_areas"] == ["auth"]
    assert ev["has_failing_ci"] is False
    assert CodeReviewSurface().classify(ev) == "ai_investigation"


def test_the_typed_code_change_accepts_what_the_surface_really_produces(repo):
    """CodeChange forbids extra keys, so any key the surface adds without declaring
    it here would make adopting the model reject every review. It already drifted
    once (`revision`, which the pre-push checks read)."""
    ev = CodeReviewSurface().normalize(gather(repo, "staged"))
    typed = CodeChange.model_validate(ev)
    assert typed.revision["diff_args"] and typed.touched_areas == ["auth"]
    # The GitHub-shaped input (no revision) still fits too.
    assert CodeChange.model_validate(CodeReviewSurface().normalize({"files": []})).revision == {}


# ── the sink ─────────────────────────────────────────────────────────────────
def test_review_file_writes_the_action_verbatim(tmp_path):
    """The review file holds the approved comment and label verbatim, with each citation's score."""
    sink = ReviewFileSink(root=str(tmp_path))
    out = sink.emit({"comment": "LGTM, one nit.", "label": "needs-nit"}, target="staged",
                    context={"tenant": "demo", "run_id": "r1", "severity": "low",
                             "citations": [{"snippet_id": "x", "score": 0.5}]})
    text = Path(out["review"]["markdown"]).read_text(encoding="utf-8")
    assert "LGTM, one nit." in text and "needs-nit" in text and "`x` (relevance 0.5)" in text


def test_review_file_cannot_be_steered_out_of_its_directory(tmp_path):
    """Path-shaped target and tenant values cannot move the review file out of var/reviews."""
    out = ReviewFileSink(root=str(tmp_path)).emit(
        {"comment": "c"}, target="../../etc/passwd", context={"tenant": "../../x"})
    assert out["review"]["markdown"].startswith(str(tmp_path / "reviews"))


# ── overrides ────────────────────────────────────────────────────────────────
def test_source_and_sink_overrides_are_per_surface(monkeypatch):
    """code_review defaults to GitHub in and a PR comment out; CRA_SOURCE_/CRA_SINK_ switch it."""
    s = CodeReviewSurface()
    assert (config.source_name_for(s), config.sink_name_for(s)) == ("github", "pr_comment")
    monkeypatch.setenv("CRA_SOURCE_CODE_REVIEW", "git_local")
    monkeypatch.setenv("CRA_SINK_CODE_REVIEW", "review_file")
    assert (config.source_name_for(s), config.sink_name_for(s)) == ("git_local", "review_file")


# ── the CLI, end to end on the real graph ────────────────────────────────────
@adapter("retriever", "test_grounded")
class GroundedRetriever:
    """One strongly relevant hit, so the run is approve-ELIGIBLE on purpose.

    The demo corpus scores this fixture's diff at ~0.02, which escalates every
    run before the gate -- and a test of the approval prompt that never reaches
    the prompt would pass while testing nothing. The score here is the
    retriever's own, exactly as the gate reads it in production.
    """

    def retrieve(self, _query, **_scope):
        """The one hit, whatever the query and namespace."""
        return [{"id": "iss-101", "score": 0.9, "text":
                 "OAuth callback returns 500 when state param missing; fix in auth middleware"}]


@pytest.fixture(name="cli")
def fixture_cli(repo, monkeypatch):
    """The real graph on `_default`, with the grounded retriever above."""
    monkeypatch.setenv("RETRIEVER", "test_grounded")
    # `apply_defaults` writes os.environ directly, so monkeypatch must own each key
    # BEFORE it runs. `setenv` records the prior state even when that state is
    # "absent"; `delenv(raising=False)` on an absent key records nothing, and so
    # restores nothing -- which leaked CRA_CODEBASE_ROOT into every later test,
    # whose git source then read THIS test's repo. `apply_defaults` uses setdefault for
    # the source and sink, so they are set to their real values here.
    monkeypatch.setenv("CRA_SOURCE_CODE_REVIEW", "git_local")
    monkeypatch.setenv("CRA_SINK_CODE_REVIEW", "review_file")
    monkeypatch.setenv("CRA_CODEBASE_ROOT", "")
    review.apply_defaults(str(repo))
    return repo


def _reviews(tmp_path_factory_root):
    return list(tmp_path_factory_root.glob("var/reviews/**/*.md"))


@pytest.mark.usefixtures("cli")
def test_cli_approval_writes_exactly_the_proposed_review(tmp_path, capsys):
    """Approving writes the very text shown as the proposed action, once (invariant 4)."""
    code = review.run("_default", "staged", ask=lambda _: "y", interactive=True)
    out = capsys.readouterr().out
    assert "DECISION : approve" in out, "the fixture must reach the approval gate"
    assert code == 0 and "approved and written" in out
    written = _reviews(tmp_path)
    assert len(written) == 1
    proposed = out.split("PROPOSED ACTION (exactly what will be written if you approve):\n\n")[1]
    proposed = proposed.split("\n" + review.RULE)[0].split("\nlabel:")[0]
    assert proposed.strip() in written[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("answer", ["n", "", "yes please", "Y "])
@pytest.mark.usefixtures("cli")
def test_cli_anything_but_y_denies_and_writes_nothing(tmp_path, capsys, answer):
    """Only `y` (trimmed, any case) approves; anything else is a recorded denial that writes
    nothing."""
    code = review.run("_default", "staged", ask=lambda _: answer, interactive=True)
    out = capsys.readouterr().out
    if answer.strip().lower() == "y":
        assert "approved" in out
        return
    assert code == 0 and "denied" in out
    assert not _reviews(tmp_path)
    audit = list(tmp_path.glob("var/audit/**/*.jsonl"))
    records = [json.loads(line) for p in audit for line in p.read_text().splitlines()]
    assert any(r.get("approved") is False for r in records), "a denial must be recorded"


@pytest.mark.usefixtures("cli")
def test_cli_without_a_terminal_refuses_to_decide(tmp_path, capsys):
    """Without a terminal the CLI never asks, never approves and writes no review."""
    asked = []
    code = review.run("_default", "staged", ask=asked.append, interactive=False)
    assert code == 2 and not asked
    assert "refusing to decide" in capsys.readouterr().out
    assert not _reviews(tmp_path)
    # ...but the run is RECORDED as abandoned, findings and booleans included:
    # it is real model output over a real change, and calibration needs it.
    abandoned = [r for r in iter_records() if r.get("outcome") == "abandoned"]
    assert len(abandoned) == 1 and abandoned[0].get("decision") == "approve"
    assert abandoned[0].get("approved") is None, "abandoned is not a denial"


# ── the diff budget ──────────────────────────────────────────────────────────


def _diffs(sizes: dict[str, int]):
    return lambda p: f"<{p}>" + "x" * sizes[p]


def test_budget_is_spent_on_code_before_docs_and_never_on_data():
    """The first real run showed ~40k chars of CSV while the riskiest script fell
    past the cut. Path order is alphabetical, and alphabetical is not priority."""
    files = ["a_data.csv", "b_notes.md", "c_settings.yml", "z_split_banks.py"]
    sizes = {"a_data.csv": 10, "b_notes.md": 40, "c_settings.yml": 40, "z_split_banks.py": 50}
    diff, note = pack_diff(files, _diffs(sizes), budget=150)   # .py 68 + .yml 56 fit; .md does not
    assert diff.index("<z_split_banks.py>") < diff.index("<c_settings.yml>")
    assert "<a_data.csv>" not in diff, "data content must never be shown"
    assert "<b_notes.md>" not in diff
    assert "NOT SHOWN: b_notes.md" in note
    assert "DATA FILES, content never shown: a_data.csv" in note


def test_a_lone_oversized_file_is_cut_not_dropped():
    """A lone file over the budget is cut to fit and named, not dropped to an empty diff."""
    diff, note = pack_diff(["huge.py"], _diffs({"huge.py": 500}), budget=100)
    assert len(diff) == 100 and "CUT PART-WAY: huge.py" in note


def test_everything_fitting_means_no_note():
    """When everything fits the note is empty: a note means something was left out."""
    diff, note = pack_diff(["a.py", "b.md"], _diffs({"a.py": 5, "b.md": 5}), budget=100)
    assert note == "" and "<a.py>" in diff and "<b.md>" in diff


def test_a_real_staged_csv_is_named_but_its_rows_never_reach_the_evidence(repo):
    """A staged CSV is named and routed on, but its rows (PII here) never reach diff or body."""
    (repo / "export.csv").write_text("email,name\nperson@example.com,Some Person\n")
    _git(repo, "add", ".")
    raw = gather(repo, "staged")
    assert "person@example.com" not in raw["diff"] and "person@example.com" not in raw["body"]
    assert "export.csv" in raw["diff"]                       # named, so a human knows
    assert "export.csv" in [f["filename"] for f in raw["files"]]   # and still routed on


# ── the review text: comment AND findings, judged and posted as one ─────────


def _diag(findings):
    return Diagnosis(summary="s", severity="low",
                     proposed_comment="Some areas need attention:",
                     citations=[Citation(snippet_id="c1", score=0.9)], findings=findings)


FINDING = Finding(severity="medium", file="Masterclass/models.py", line=42,
                  issue="unique constraint fails on existing duplicate banks",
                  suggestion="dedupe before adding the constraint",
                  grounded=True, citation_ids=["c1"])


def test_findings_are_rendered_into_the_review():
    """Each finding is rendered under the comment with its file:line, issue and suggestion."""
    text = render_review(_diag([FINDING]))
    assert text.startswith("Some areas need attention:")
    assert "`Masterclass/models.py`:42" in text
    assert "unique constraint fails on existing duplicate banks" in text
    assert "Suggestion: dedupe before adding the constraint" in text


def test_no_findings_leaves_the_comment_byte_identical():
    """Keeps every gate task's expected text unchanged."""
    assert render_review(_diag([])) == "Some areas need attention:"


def test_an_ungrounded_finding_is_labelled_not_hidden():
    """An ungrounded finding is posted with a label saying so, not silently dropped."""
    ungrounded = FINDING.model_copy(update={"grounded": False})
    assert "not grounded in retrieved code" in render_review(_diag([ungrounded]))


@dataclass
class _RecordingEntailer:
    claims: list[str] = field(default_factory=list)

    def score(self, claim, _passage):
        """Record the claim and fully entail it, so the test sees every claim the gate judges."""
        self.claims.append(claim)
        return 1.0


def test_the_gate_judges_and_the_action_posts_the_same_text():
    """A finding that reached the page without being judged would be unchecked
    model output -- exactly what the four booleans exist to prevent."""
    ent = _RecordingEntailer()
    state = {
        "diagnosis": _diag([FINDING]), "tenant": "_default",
        "_retrieved": [{"id": "c1", "score": 0.9, "text": "class QuestionBank: ..."}],
        "_policy": {"min_relevance_score": 0.35, "severity_allowlist": ["low", "medium"],
                    "allowed_actions": ["comment", "comment+label"]},
        "_prompts": {"judge": "explain only"},
        "_surface_name": "code_review",
    }
    out = independent_review(state, entailer=ent)
    assert out["review"].decision == "approve"
    posted = out["proposed_action"]["comment"]
    assert posted == render_review(state["diagnosis"])
    assert "unique constraint fails" in posted
    # Every piece of MODEL text on the page was judged -- now claim by claim
    # (2026-11-17), so each finding is judged on its own rather than as part of
    # one blended review.
    assert ent.claims[0] == "Some areas need attention:"
    assert ent.claims[1] == f"{FINDING.issue} {FINDING.suggestion}"
    assert len(ent.claims) == 1 + len(state["diagnosis"].findings)


# ── entailment, claim by claim ───────────────────────────────────────────────

DIFF = ("diff --git a/Masterclass/models.py b/Masterclass/models.py\n"
        "--- a/Masterclass/models.py\n+++ b/Masterclass/models.py\n"
        "@@ -1 +1 @@\n+    unique_together = ('course', 'slot')  # one question bank per slot\n"
        "diff --git a/billing/views.py b/billing/views.py\n"
        "--- a/billing/views.py\n+++ b/billing/views.py\n"
        "@@ -1 +1 @@\n+    refund = invoice.total * rate\n")
RETRIEVED = [{"id": "c1", "text": "class QuestionBank(models.Model): course slot"},
             {"id": "c2", "text": "def refund(invoice): currency rounding totals"}]


def _finding(file, issue, cites):
    return Finding(severity="medium", file=file, issue=issue, suggestion="",
                   citation_ids=cites)


def test_a_finding_is_judged_against_its_own_file_and_citations_only():
    """A finding's premise is its own file's hunk and its own citations, never another file's."""
    diag = Diagnosis(summary="s", severity="low", proposed_comment="",
                     citations=[Citation(snippet_id="c1", score=0.9)],
                     findings=[_finding("Masterclass/models.py", "x", ["c1"])])
    claims = entailment_claims(diag, {"diff": DIFF}, RETRIEVED)
    assert len(claims) == 1
    premise = claims[0][2]
    assert "unique_together" in premise and "QuestionBank" in premise
    assert "refund" not in premise, "another file's diff and passages are never borrowed"


def test_one_unsupported_finding_fails_the_whole_review():
    """One finding its premise does not support fails entailment for the whole review, by name."""
    good = _finding("Masterclass/models.py",
                    "unique_together course slot enforces one question bank per slot", ["c1"])
    bad = _finding("Masterclass/models.py",
                   "deleting a student wipes stripe payouts and kafka offsets", ["c1"])
    diag = Diagnosis(summary="s", severity="low", proposed_comment="",
                     citations=[Citation(snippet_id="c1", score=0.9)], findings=[good, bad])
    ok, lowest, failures = _entailment(diag, {"diff": DIFF}, RETRIEVED, LexicalEntailer(), 0.25)
    assert not ok and lowest < 0.25
    assert len(failures) == 1 and failures[0].startswith("finding 2")


def test_a_finding_about_new_code_is_entailed_by_the_diff():
    """The new line is not in any retrieved passage -- in normal use it is not in
    the index yet -- and the finding is still checkable against the change."""
    f = _finding("Masterclass/models.py", "unique_together now enforces one question bank "
                 "per slot", [])
    diag = Diagnosis(summary="s", severity="low", proposed_comment="", findings=[f])
    ok, _, _ = _entailment(diag, {"diff": DIFF}, [], LexicalEntailer(), 0.25)
    assert ok


def test_an_empty_review_still_fails_closed():
    """A review with no comment and no findings scores 0.0 and fails closed (invariant 12)."""
    diag = Diagnosis(summary="s", severity="low", proposed_comment="")
    ok, lowest, _ = _entailment(diag, {"diff": DIFF}, RETRIEVED, LexicalEntailer(), 0.25)
    assert not ok and lowest == 0.0


# ── live status and the monitor ──────────────────────────────────────────────



def _live_runs(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / "var" / "live").glob("run-*.json")]


@pytest.mark.usefixtures("cli")
def test_a_run_records_every_stage_in_order_and_finishes(tmp_path):
    """The live snapshot lists every node in order, the approval pause included, with the gate's
    checks."""
    review.run("_default", "staged", ask=lambda _: "y", interactive=True)
    (snap,) = _live_runs(tmp_path)
    names = [s["name"] for s in snap["stages"]]
    assert names[:5] == ["resolve_tenant", "fetch_evidence", "normalize_evidence",
                         "prepush_checks", "classify"]
    assert names[-1] == "audit" and snap["status"] == "finished"
    assert "await_approval" in names, "the pause must be visible, not just the result"
    assert all(s["status"] in ("ok", "waiting") for s in snap["stages"])
    gate = next(s for s in snap["stages"] if s["name"] == "independent_review")
    assert gate["detail"]["decision"] == "approve" and gate["detail"]["checks"]


@pytest.mark.usefixtures("cli")
def test_the_snapshot_never_carries_evidence_text(tmp_path):
    """Live snapshots hold ids and counts only: no diff text and no retrieved passage."""
    review.run("_default", "staged", ask=lambda _: "n", interactive=True)
    raw = (next((tmp_path / "var" / "live").glob("run-*.json"))).read_text()
    assert "'state' not in request.GET" not in raw        # diff text
    assert "OAuth callback returns 500" not in raw         # snippet text


@pytest.mark.parametrize("answer,interactive,status", [("n", True, "denied"),
                                                        ("y", False, "abandoned")])
@pytest.mark.usefixtures("cli")
def test_a_run_that_ends_at_the_prompt_says_how(tmp_path, answer, interactive, status):
    """A run that ends at the prompt finishes as `denied` or `abandoned`, saying which."""
    review.run("_default", "staged", ask=lambda _: answer, interactive=interactive)
    assert _live_runs(tmp_path)[0]["status"] == status


@pytest.mark.usefixtures("cli")
def test_live_status_can_never_fail_a_run(monkeypatch):
    """A live-status write that raises OSError must not fail the review it describes."""
    def broken(*a, **k):
        raise OSError("disk full")              # the failure the contract is about
    monkeypatch.setattr(live, "stage", broken)
    assert review.run("_default", "staged", ask=lambda _: "y", interactive=True) == 0


@pytest.mark.usefixtures("cli")
def test_a_failing_stage_is_recorded_and_still_raises(tmp_path, monkeypatch):
    """A node that raises marks its stage and the run as error, and still propagates."""
    monkeypatch.setattr(GitLocalSource, "gather",
                        lambda self, target, **o: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        review.run("_default", "staged", ask=lambda _: "y", interactive=True)
    (snap,) = _live_runs(tmp_path)
    fetch = next(s for s in snap["stages"] if s["name"] == "fetch_evidence")
    assert fetch["status"] == "error" and "boom" in fetch["error"] and snap["status"] == "error"


@pytest.mark.usefixtures("cli")
def test_monitor_status_combines_live_traces_and_audit(monkeypatch):
    """monitor.status() joins model reachability, live jobs, trace estimates and audit records."""
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1")      # unreachable
    review.run("_default", "staged", ask=lambda _: "y", interactive=True)
    s = monitor.status()
    assert s["models"]["reachable"] is False
    assert s["jobs"] and s["jobs"][0]["kind"] == "run"
    assert "generate_diagnosis" in s["estimates"]
    assert s["recent"] and s["recent"][0]["decision"] == "approve"


def test_monitor_serves_only_its_two_routes_on_localhost(monkeypatch):
    """The monitor binds 127.0.0.1, serves / and /api/status, and refuses a traversal."""
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1")
    assert monitor.HOST == "127.0.0.1"
    server = monitor.ThreadingHTTPServer((monitor.HOST, 0), monitor.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://{monitor.HOST}:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as page:
            assert b"Code Reviewer Agent" in page.read()
        with urllib.request.urlopen(base + "/api/status") as api:
            assert "jobs" in json.loads(api.read())
        with pytest.raises(urllib.error.HTTPError), urllib.request.urlopen(base + "/../var/audit"):
            pass
    finally:
        server.shutdown()
        server.server_close()


def test_the_page_never_injects_data_as_html():
    """Targets, errors and ids come from callers and models; the page must render
    them as text. One `innerHTML` would make a crafted branch name an XSS."""
    assert "innerHTML" not in monitor.PAGE and "insertAdjacentHTML" not in monitor.PAGE


# ── pre-push checks, end to end on the real graph ────────────────────────────
def test_checks_run_and_reach_the_review_the_prompt_and_the_audit(cli, tmp_path, monkeypatch):
    """A flagged check reaches the written review, the model's prompt and the audit record."""
    (cli / "app" / "loop.py").write_text(
        "def f(ids):\n    for i in ids:\n        User.objects.get(pk=i)\n")
    _git(cli, "add", "app/loop.py")
    prompts = []
    real = StubLLM.complete

    def spy(self, system, user, **kw):
        prompts.append(user)
        return real(self, system, user, **kw)
    monkeypatch.setattr(StubLLM, "complete", spy)

    review.run("_default", "staged", ask=lambda _: "y", interactive=True)

    written = _reviews(tmp_path)[0].read_text(encoding="utf-8")
    assert "Pre-push checklist" in written and "n_plus_one" in written
    assert any("CHECKS" in p and "n_plus_one" in p for p in prompts), "facts reach the model"
    audit = [json.loads(line) for p in tmp_path.glob("var/audit/**/*.jsonl")
             for line in p.read_text().splitlines()]
    assert any(c["check"] == "n_plus_one" and c["status"] == "flag"
               for r in audit for c in r.get("checks", []))


def test_entailment_never_judges_the_code_generated_checklist():
    """Entailment asks whether a MODEL claim follows from retrieved code. The
    checklist is computed facts; judging it would only add noise to the gate."""
    ent = _RecordingEntailer()
    state = {
        "diagnosis": _diag([FINDING]), "tenant": "_default",
        "_retrieved": [{"id": "c1", "score": 0.9, "text": "class QuestionBank: ..."}],
        "_policy": {"min_relevance_score": 0.35, "severity_allowlist": ["low", "medium"],
                    "allowed_actions": ["comment", "comment+label"]},
        "_prompts": {"judge": "explain only"}, "_surface_name": "code_review",
        "_checks": [{"check": "n_plus_one", "step": 0, "status": "flag",
                     "message": "HIGH: db round trip", "file": "a.py", "line": 3}],
    }
    out = independent_review(state, entailer=ent)
    posted = out["proposed_action"]["comment"]
    assert "Pre-push checklist" in posted and "n_plus_one" in posted
    assert all("Pre-push checklist" not in c for c in ent.claims)
    assert posted.startswith(render_review(state["diagnosis"]))


def test_a_surface_without_a_diff_gets_no_checks():
    """A surface whose evidence has no diff (log_review) gets an empty checklist, not a crash."""
    assert prepush_checks({"_surface_name": "log_review", "evidence": {}}) == {"_checks": []}


@pytest.mark.usefixtures("cli")
def test_the_judges_guard_report_reaches_the_audit_record(tmp_path):
    """The judge reads the DIAGNOSIS, which is model output built from untrusted
    evidence. Its guard report used to be discarded (`judge_turn, _ = ...`), so
    injection phrasing carried forward by the model was counted and then lost."""
    review.run("_default", "staged", ask=lambda _: "y", interactive=True)
    (rec,) = [json.loads(line) for p in tmp_path.glob("var/audit/**/*.jsonl")
              for line in p.read_text().splitlines()]
    judge = rec["guard"]["judge"]
    assert {"redacted", "injection_suspected", "fence_forgery"} <= set(judge)


def test_a_flag_without_a_file_renders_instead_of_crashing():
    """Masterclass c91d369 (2026-11-17): the tracker check flags the commit's
    AUTHOR, not a path, and render_checklist crashed with KeyError: 'file'."""
    text = render_checklist([
        {"check": "tracker", "step": 4, "status": "flag",
         "message": "git author 'me-Usaid' has no tracker in the policy"},
        {"check": "docs_updated", "step": 1, "status": "flag", "message": "not updated",
         "file": "Masterclass/Code Documentation/ADMIN_DOCS.md", "line": 3}])
    assert "**tracker** — git author 'me-Usaid'" in text
    assert "`Masterclass/Code Documentation/ADMIN_DOCS.md`:3" in text
