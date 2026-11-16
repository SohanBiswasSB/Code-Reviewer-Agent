"""The pre-push change checks, each against a REAL temporary git repository.

Every check is tested both ways -- it flags what the ritual forbids AND passes
what the ritual allows -- because a check that only ever passes is decoration,
and one that only ever flags is noise nobody reads.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from tools import change_checks as cc


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Sohan Biswas")
    (root / "README.md").write_text("base\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def write(root, files: dict[str, str], commit: str | None = None):
    for path, text in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        _git(root, "add", "--", path)          # ONLY what was written: tests rely on it
    if commit:
        _git(root, "commit", "-q", "-m", commit)


STAGED = {"diff_args": ["diff", "--cached"], "new_ref": ":", "base_ref": "HEAD:"}


def run(root, check, config=None):
    files = [f for f in subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only"],
                                       capture_output=True, text=True).stdout.split("\n") if f]
    return check(cc.Change(root=root, files=files, revision=dict(STAGED)), config or {})


def flags(results, check=None):
    return [r for r in results if r["status"] == "flag" and (check is None or r["check"] == check)]


# ── step 0: N+1 ──────────────────────────────────────────────────────────────
LOOP_DB = """
def view(rows):
    out = []
    for row in rows:
        out.append(User.objects.get(pk=row.user_id))
    return out
"""


def test_a_query_per_row_is_flagged_high(repo):
    write(repo, {"app/views.py": LOOP_DB})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert f["line"] == 5 and "HIGH [request]" in f["message"] and "loop at line 4" in f["message"]


@pytest.mark.parametrize("loop", ["for row in (1, 2, 3):", "for row in STATUSES:",
                                  "for row in range(3):", "for batch in batches:",
                                  "for attempt in range(retries):"])
def test_bounded_and_chunked_loops_are_not_n_plus_one(repo, loop):
    write(repo, {"app/views.py": f"def v(batches, retries):\n    {loop}\n"
                                 f"        User.objects.get(pk=1)\n"})
    assert flags(run(repo, cc.check_n_plus_one)) == []


@pytest.mark.parametrize("call,kind", [("cache.get(k)", "MEDIUM"), ("requests.get(u)", "MEDIUM"),
                                       ("self.session.post(u)", "MEDIUM"),
                                       ("row.items_set.filter(a=1).count()", "HIGH"),
                                       ("Order.objects.filter(a=1).exists()", "HIGH")])
def test_round_trip_kinds(repo, call, kind):
    write(repo, {"app/x.py": f"def f(rows, k, u):\n    for row in rows:\n        {call}\n"})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert f["message"].startswith(kind)


@pytest.mark.parametrize("call", ["get(k)", "form.save()", "serializer.save()", "d.get(k)"])
def test_things_that_are_not_round_trips(repo, call):
    write(repo, {"app/x.py": f"def f(rows, k, form, serializer, d):\n    for r in rows:\n"
                             f"        {call}\n"})
    assert flags(run(repo, cc.check_n_plus_one)) == []


def test_a_comprehension_is_a_loop(repo):
    write(repo, {"app/x.py": "def f(ids):\n    return [User.objects.get(pk=i) for i in ids]\n"})
    assert len(flags(run(repo, cc.check_n_plus_one))) == 1


def test_a_helper_that_queries_is_followed(repo):
    write(repo, {"app/x.py": "def load(i):\n    return User.objects.get(pk=i)\n\n\n"
                             "def f(ids):\n    for i in ids:\n        load(i)\n"})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert "via load()" in f["message"] and f["line"] == 7


def test_a_def_inside_a_loop_is_not_in_the_loop(repo):
    write(repo, {"app/x.py": "def f(ids):\n    for i in ids:\n        def g():\n"
                             "            return User.objects.get(pk=1)\n"})
    assert flags(run(repo, cc.check_n_plus_one)) == []


def test_exemption_needs_a_reason(repo):
    write(repo, {"app/x.py": "def f(ids):\n    for i in ids:  # O(1)-exempt: ids is the 3 regions\n"
                             "        User.objects.get(pk=i)\n"})
    assert flags(run(repo, cc.check_n_plus_one)) == []
    _git(repo, "reset", "-q", "--hard")
    write(repo, {"app/y.py": "def f(ids):\n    for i in ids:  # O(1)-exempt:\n"
                             "        User.objects.get(pk=i)\n"})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert "no reason" in f["message"]


def test_only_new_findings_count(repo):
    write(repo, {"app/x.py": LOOP_DB}, commit="old n+1")
    write(repo, {"app/x.py": LOOP_DB + "\n\ndef other():\n    return 1\n"})
    assert flags(run(repo, cc.check_n_plus_one)) == []


def test_serializer_method_field_is_a_per_row_query(repo):
    write(repo, {"app/serializers.py":
                 "class S(serializers.ModelSerializer):\n    def get_n(self, obj):\n"
                 "        return Order.objects.filter(user=obj).count()\n"})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert "once per serializer row" in f["message"]


def test_offline_scripts_are_labelled_offline(repo):
    write(repo, {"Miscellaneous/Scripts/fix.py": LOOP_DB})
    assert "[offline]" in flags(run(repo, cc.check_n_plus_one))[0]["message"]


# ── step 1: hygiene ──────────────────────────────────────────────────────────
def test_a_new_pragma_is_flagged_but_a_docstring_quoting_one_is_not(repo):
    write(repo, {"app/x.py": 'def f():\n    """Never write # noqa here."""\n    return 1\n'})
    assert flags(run(repo, cc.check_suppressions)) == []
    write(repo, {"app/x.py": "import os  # noqa: F401\n"})
    (f,) = flags(run(repo, cc.check_suppressions))
    assert "type-ignore/noqa/isort-skip" in f["message"] and f["line"] == 1


def test_an_existing_pragma_is_not_new(repo):
    write(repo, {"app/x.py": "x = 1  # pylint: disable=invalid-name\n"}, commit="old")
    write(repo, {"app/x.py": "x = 1  # pylint: disable=invalid-name\ny = 2\n"})
    assert flags(run(repo, cc.check_suppressions)) == []
    write(repo, {"app/x.py": "x = 1  # pylint: disable=invalid-name\n"
                             "y = 2  # pylint: disable=invalid-name\n"})
    assert "pylint:invalid-name" in flags(run(repo, cc.check_suppressions))[0]["message"]


def test_long_and_non_ascii_added_lines(repo):
    write(repo, {"app/x.py": "x = '" + "a" * 130 + "'\ny = 'café'\nz = 1\n"})
    res = run(repo, cc.check_line_hygiene, {"max_line_length": 120})
    assert [f["line"] for f in flags(res, "line_length")] == [1]
    assert [f["line"] for f in flags(res, "ascii")] == [2]


def test_only_added_lines_are_held_to_the_rules(repo):
    write(repo, {"app/x.py": "x = '" + "a" * 130 + "'\n"}, commit="old long line")
    write(repo, {"app/x.py": "x = '" + "a" * 130 + "'\ny = 1\n"})
    assert flags(run(repo, cc.check_line_hygiene)) == []


def test_a_new_handler_without_a_docstring(repo):
    write(repo, {"app/views.py": "class V(APIView):\n    def get(self, r):\n        return 1\n"})
    (f,) = flags(run(repo, cc.check_handler_docstrings))
    assert f["message"] == "V.get has no docstring"


def test_an_old_handler_gap_is_not_new(repo):
    write(repo, {"app/views.py": "class V(APIView):\n    def get(self, r):\n        return 1\n"},
          commit="old")
    write(repo, {"app/views.py": "class V(APIView):\n    def get(self, r):\n        return 2\n"})
    assert flags(run(repo, cc.check_handler_docstrings)) == []


def test_an_import_of_an_untracked_file(repo):
    (repo / "app").mkdir()
    (repo / "app" / "helpers.py").write_text("X = 1\n")             # present, NOT added
    write(repo, {"app/views.py": "from app.helpers import X\nimport os\nimport json\n"})
    (f,) = flags(run(repo, cc.check_untracked_imports))
    assert "app/helpers.py" in f["message"] and f["line"] == 1
    _git(repo, "add", "app/helpers.py")
    assert flags(run(repo, cc.check_untracked_imports)) == []


def test_a_relative_import_of_an_untracked_file(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "util.py").write_text("Y = 1\n")
    write(repo, {"pkg/views.py": "from .util import Y\n"})
    assert flags(run(repo, cc.check_untracked_imports))


# ── conventions ──────────────────────────────────────────────────────────────
DOCS = {"docs": {"folder": "Code Documentation",
                 "files": ["DATA_MODELS.md", "ADMIN_DOCS.md", "CODEBASE_OVERVIEW.md", "README.md"],
                 "allowed_dirs": ["Legacy Source Records"],
                 "rules": [{"when": ["models.py"], "doc": "DATA_MODELS.md"},
                           {"when": ["views*.py", "utils/"], "doc": "CODEBASE_OVERVIEW.md"}],
                 "changelog_heading": r"^###\s*\[\d{4}-\d{2}-\d{2}\]\s*-\s*\S"}}


@pytest.fixture
def app_repo(repo):
    write(repo, {f"Shop/Code Documentation/{n}": "# doc\n## Logic & Schema Change Log\n"
                 for n in DOCS["docs"]["files"]} | {"Shop/models.py": "x = 1\n"}, commit="app")
    return repo


def test_a_model_change_without_its_doc_is_flagged(app_repo):
    write(app_repo, {"Shop/models.py": "x = 2\n", "Shop/utils/money.py": "y = 1\n"})
    msgs = {f["file"]: f["message"] for f in flags(run(app_repo, cc.check_docs, DOCS))}
    assert "Shop/Code Documentation/DATA_MODELS.md" in msgs
    assert "Shop/Code Documentation/CODEBASE_OVERVIEW.md" in msgs


def test_a_model_change_with_a_dated_doc_entry_passes(app_repo):
    write(app_repo, {"Shop/models.py": "x = 2\n", "Shop/Code Documentation/DATA_MODELS.md":
                     "# doc\n## Logic & Schema Change Log\n### [2026-11-16] - x is 2\n"})
    assert flags(run(app_repo, cc.check_docs, DOCS)) == []


def test_a_doc_change_without_a_changelog_entry(app_repo):
    write(app_repo, {"Shop/Code Documentation/README.md": "# doc\nmore words\n"})
    assert flags(run(app_repo, cc.check_docs, DOCS), "docs_changelog")


def test_a_fifth_doc_file_is_flagged(app_repo):
    write(app_repo, {"Shop/Code Documentation/NOTES.md": "### [2026-11-16] - n\n"})
    (f,) = flags(run(app_repo, cc.check_docs, DOCS), "docs_four_files")
    assert "NOTES.md" in f["message"]


def test_a_repo_without_the_convention_is_not_flagged_for_it(repo):
    write(repo, {"Shop/models.py": "x = 2\n"})
    assert run(repo, cc.check_docs, {}) == []
    assert run(repo, cc.check_postman, {}) == []
    assert run(repo, cc.check_tracker, {}) == []
    assert run(repo, cc.check_commit_message, {}) == []


def test_never_commit_and_read_only_paths(repo):
    cfg = {"never_commit": ["Misc/API Captures/"], "read_only": ["*/Legacy Source Records/*"]}
    write(repo, {"Misc/API Captures/a.json": "{}", "Shop/Docs/Legacy Source Records/x.md": "x"})
    assert {f["check"] for f in flags(run(repo, cc.check_paths, cfg))} == {"never_commit",
                                                                           "read_only"}


PM = {"postman": {"collection": "pm/all.postman_collection.json",
                  "environments": "pm/*environment*.json", "when": ["views*.py", "urls.py"],
                  "generator_inputs": ["scripts/gen.py"], "no_failure_example_ok": ["faqs"]}}


def test_an_api_change_without_the_collection(repo):
    write(repo, {"Shop/views.py": "x = 1\n"})
    assert flags(run(repo, cc.check_postman, PM), "postman_collection")


def test_a_hand_edited_or_thin_collection(repo):
    col = {"item": [{"name": "list", "request": {"description": ""}, "response": [{"code": 200}]},
                    {"name": "faqs", "request": {"description": "d"}, "response": [{"code": 200}]}]}
    write(repo, {"pm/all.postman_collection.json": json.dumps(col)})
    res = run(repo, cc.check_postman, PM)
    assert flags(res, "postman_hand_edit")
    msgs = [f["message"] for f in flags(res, "postman_structure")]
    assert "list: no description" in msgs and "list: no failure example" in msgs
    assert not any(m.startswith("faqs") for m in msgs), "allow-listed request"


def test_a_secret_left_in_an_environment(repo):
    env = {"values": [{"key": "token", "type": "secret", "value": "abc"},
                      {"key": "host", "type": "default", "value": "x"}]}
    write(repo, {"pm/LOCAL.postman_environment.json": json.dumps(env)})
    (f,) = flags(run(repo, cc.check_postman, PM), "postman_env_secret")
    assert "token" in f["message"] and "host" not in f["message"]


HEADER = ["Issue Key", "Issue Type", "Summary", "Description", "Original Estimate", "Components",
          "Labels", "Sprint", "Reporter", "Assignee", "Epic Link", "Date"]
TR = {"tracker": {"dir": "T", "owners": {"Sohan Biswas": {"base": "SB", "prefix": "MO-"}},
                  "header": HEADER, "required_blocks": ["**Estimate**", "**VERIFIED.**"]}}


def _row(key, desc="- **Estimate**: 2h **VERIFIED.** ok"):
    vals = [key, "Task", "s", desc, "2h", "c", "l", "Sprint-2026-W46", "Sohan Biswas",
            "Sohan Biswas", "", "2026-11-16"]
    return ",".join(f'"{v}"' for v in vals)


@pytest.fixture
def tracked(repo):
    write(repo, {"T/SB.csv": ",".join(f'"{h}"' for h in HEADER) + "\n" + _row("MO-7") + "\n",
                 "T/SB.md": "## MO-7 - s\n"}, commit="tracker")
    return repo


def test_a_change_without_a_tracker_row(tracked):
    write(tracked, {"app/x.py": "x = 1\n"})
    assert len(flags(run(tracked, cc.check_tracker, TR), "tracker_row")) == 2


def test_a_valid_new_tracker_row_passes(tracked):
    csv_text = (tracked / "T/SB.csv").read_text() + _row("MO-8") + "\n"
    write(tracked, {"T/SB.csv": csv_text, "T/SB.md": "## MO-7 - s\n## MO-8 - t\n"})
    assert flags(run(tracked, cc.check_tracker, TR)) == []


@pytest.mark.parametrize("key,desc,check", [("MO-9", None, "tracker_key"),
                                            ("MO-8", "no blocks here", "tracker_blocks")])
def test_bad_tracker_rows(tracked, key, desc, check):
    csv_text = (tracked / "T/SB.csv").read_text() + (_row(key, desc) if desc else _row(key)) + "\n"
    write(tracked, {"T/SB.csv": csv_text, "T/SB.md": f"## {key} - t\n"})
    assert flags(run(tracked, cc.check_tracker, TR), check)


def test_an_author_with_no_tracker(tracked):
    _git(tracked, "config", "user.name", "Someone Else")
    write(tracked, {"app/x.py": "x = 1\n"})
    (f,) = flags(run(tracked, cc.check_tracker, TR))
    assert "Someone Else" in f["message"]


CM = {"commit_message": {"file": "commit_message.txt", "required_blocks": ["VERIFIED."]}}
GOOD = "feat(shop): orders load in O(1) queries (MO-8)\n\nWhy.\n\nVERIFIED. tests pass\n"


@pytest.mark.parametrize("text,check", [(None, "commit_message"),
                                        ("fixed stuff\n", "commit_subject"),
                                        (GOOD + "Co-Authored-By: x\n", "commit_attribution"),
                                        (GOOD.replace("VERIFIED.", "Checked."), "commit_blocks")])
def test_bad_commit_messages(repo, text, check):
    write(repo, {"app/x.py": "x = 1\n"})
    if text is not None:
        (repo / "commit_message.txt").write_text(text)
    assert flags(run(repo, cc.check_commit_message, CM), check)


def test_a_good_commit_message_passes(repo):
    write(repo, {"app/x.py": "x = 1\n"})
    (repo / "commit_message.txt").write_text(GOOD)
    assert flags(run(repo, cc.check_commit_message, CM)) == []


def test_a_reviewed_commit_is_held_to_its_own_message(repo):
    write(repo, {"app/x.py": "x = 1\n"}, commit="wip")
    rev = {"diff_args": ["show", "--format=", "HEAD"], "new_ref": "HEAD:", "base_ref": "HEAD^:",
           "commit": "HEAD"}
    res = cc.check_commit_message(cc.Change(root=repo, files=["app/x.py"], revision=rev), CM)
    assert flags(res, "commit_subject")


# ── the runner ───────────────────────────────────────────────────────────────
def test_what_cannot_be_checked_is_said_out_loud(repo):
    write(repo, {"app/x.py": "x = 1\n"})
    res = run(repo, cc.check_not_checkable, {"not_checkable": [["isort", 1, "needs the binary"]]})
    assert res == [{"check": "isort", "step": 1, "status": "skip", "message": "needs the binary"}]


def test_an_unreadable_repo_skips_that_check_and_the_rest_still_run(repo, monkeypatch):
    def unreadable(change, config):
        raise OSError("permission denied reading app/x.py")
    monkeypatch.setattr(cc, "CHECKS", [unreadable, cc.check_line_hygiene])
    write(repo, {"app/x.py": "x = 1\n"})
    res = cc.run_all({"files": ["app/x.py"], "revision": STAGED}, repo, {})
    assert res[0]["status"] == "skip" and "permission denied" in res[0]["message"]
    assert any(r["check"] == "line_length" for r in res)


def test_a_bug_in_a_check_is_not_hidden_as_a_skip(repo, monkeypatch):
    """A TypeError is a bug, not an unreadable repo. Turning it into a permanent
    "skip" is how a check silently stops checking anything."""
    def buggy(change, config):
        return [None.missing]                                  # AttributeError
    monkeypatch.setattr(cc, "CHECKS", [buggy])
    write(repo, {"app/x.py": "x = 1\n"})
    with pytest.raises(AttributeError):
        cc.run_all({"files": ["app/x.py"], "revision": STAGED}, repo, {})


def test_an_unreadable_change_skips_rather_than_passes(tmp_path):
    res = cc.run_all({"files": ["a.py"], "revision": {}}, tmp_path, {})
    assert any(r["check"] == "n_plus_one" and r["status"] == "skip" for r in res)


# ── the detector's known misreads (2026-11-16 audit), and batched git reads ─
@pytest.mark.parametrize("src,expect", [
    # false positives that used to be flagged
    ("def v(request, rows):\n    for r in rows:\n        request.session.get(r)\n", None),
    ("def v(rows):\n    cache = {}\n    for r in rows:\n        cache.get(r)\n", None),
    # false negatives that used to pass
    ("def v(rows):\n    for r in rows:\n        for k in ('a', 'b'):\n"
     "            User.objects.get(pk=r)\n", "db"),
    ("def v(entries):\n    for i in range(len(entries)):\n"
     "        User.objects.get(pk=entries[i])\n", "db"),
    ("def v(page_rows):\n    for row in page_rows:\n        User.objects.get(pk=row)\n", "db"),
])
def test_the_detector_reads_these_loops_correctly(src, expect):
    tree = cc.ast.parse(src)
    visitor = cc._LoopVisitor(cc._Module(tree))
    visitor.visit(tree)
    assert [k for _, _, k, _ in visitor.found][:1] == ([expect] if expect else [])


def test_a_query_added_inside_an_existing_serializer_hook_is_new(repo):
    """The hook's `def` line is old, but the round trip in it is not."""
    base = ("class S(serializers.ModelSerializer):\n    def get_n(self, obj):\n"
            "        return obj.n\n")
    write(repo, {"app/serializers.py": base}, commit="hook")
    write(repo, {"app/serializers.py": base.replace(
        "return obj.n", "return Order.objects.filter(user=obj).count()")})
    (f,) = flags(run(repo, cc.check_n_plus_one))
    assert "once per serializer row" in f["message"]


def test_batched_reads_match_per_file_git_and_use_a_handful_of_calls(repo, monkeypatch):
    files = {f"app/m{i}.py": f"import os\nX = {i}\n" for i in range(12)}
    write(repo, dict.fromkeys(list(files)[:6], "OLD = 1\n"), commit="base files")
    write(repo, files)
    calls = []
    real = cc.subprocess.run

    def counting(cmd, *a, **k):
        calls.append(cmd[3] if len(cmd) > 3 else cmd)
        return real(cmd, *a, **k)
    monkeypatch.setattr(cc.subprocess, "run", counting)
    change = cc.Change(root=repo, files=sorted(files), revision=dict(STAGED))
    for path in sorted(files):
        alone = real(["git", "-C", str(repo), "diff", "--cached", "--", path],
                     capture_output=True, text=True).stdout
        assert change.diff_for(path) == alone
        assert change.content_for(path) == files[path]
        assert change.base_for(path) == ("OLD = 1\n" if path in list(files)[:6] else "")
        assert change.tracked(path)
    assert not change.tracked("app/nope.py")
    assert len(calls) == 4, calls          # diff + new blobs + base blobs + ls-files


def test_a_range_is_checked_against_its_head_commit_not_the_reviewer(repo):
    _git(repo, "-c", "user.name=Someone Else", "commit", "-q", "--allow-empty",
         "-m", "feat(x): thing\n\nVERIFIED. yes")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    change = cc.Change(root=repo, files=[], revision={
        "diff_args": ["diff", f"{head}~1..{head}"], "new_ref": f"{head}:",
        "base_ref": f"{head}~1:", "head": head, "commit": None})
    assert cc._author(change) == "Someone Else"
    assert flags(cc.check_commit_message(change, CM)) == [], "read from the commit, not a file"


def test_a_query_several_helpers_deep_is_still_followed():
    """Propagation ran three rounds and stopped; it now runs to a fixed point."""
    src = ("def d(x):\n    return User.objects.get(pk=x)\n"
           "def c(x):\n    return d(x)\n"
           "def b(x):\n    return c(x)\n"
           "def a(x):\n    return b(x)\n"
           "def view(rows):\n    for r in rows:\n        a(r)\n")
    tree = cc.ast.parse(src)
    visitor = cc._LoopVisitor(cc._Module(tree))
    visitor.visit(tree)
    assert [(k, how) for _, _, k, how in visitor.found] == [("db", "via a()")]
