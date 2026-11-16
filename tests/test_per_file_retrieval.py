"""Per-file retrieval (2026-11-16): each changed file gets its own evidence.

One query for a 15-file change returned 5 passages -- two chunks of one file and
three docs -- so 4 of 5 findings cited a DIFFERENT file than the one they were
about, and entailment rightly failed every real run.
"""
from __future__ import annotations

from nodes._render import PROMPT_SNIPPET_CHARS, render_snippets
from nodes.generate_diagnosis import MAX_HITS, merge_hits
from surfaces.code_review import MAX_QUERY_FILES, CodeReviewSurface


def _diff(path: str, line: str) -> str:
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1 +1 @@\n-old\n+{line}\n")


def test_one_query_per_file_code_first_and_data_never():
    files = ["docs/notes.md", "exports/users.csv", "app/views.py", "config/app.yaml",
             "app/models.py"]
    ev = {"title": "t", "files": files,
          "diff": _diff("app/views.py", "def view(): ...") + _diff("docs/notes.md", "x")}
    queries = CodeReviewSurface().build_queries(ev)
    heads = [q.split("\n")[0] for q in queries[:-1]]
    assert heads == ["app/views.py", "app/models.py", "config/app.yaml", "docs/notes.md"]
    assert "def view()" in queries[0], "a file's query carries its own diff"
    assert queries[1] == "app/models.py", "past the budget: path only, BM25 still matches"
    assert not any("users.csv" in q.split("\n")[0] for q in queries), "data is never queried"
    assert queries[-1] == CodeReviewSurface().build_query(ev), "whole-change query last"


def test_the_number_of_file_queries_is_capped():
    files = [f"app/m{i}.py" for i in range(MAX_QUERY_FILES + 4)]
    queries = CodeReviewSurface().build_queries({"files": files, "diff": ""})
    assert len(queries) == MAX_QUERY_FILES + 1


def test_merge_is_round_robin_dedups_and_keeps_the_best_score():
    a = [{"id": "a1", "score": 0.8}, {"id": "shared", "score": 0.4}, {"id": "a3", "score": 0.3}]
    b = [{"id": "shared", "score": 0.7}, {"id": "b2", "score": 0.6}]
    merged = merge_hits([a, b])
    assert [h["id"] for h in merged] == ["a1", "shared", "b2", "a3"]  # by rank, then query
    assert next(h for h in merged if h["id"] == "shared")["score"] == 0.7
    many = [[{"id": f"q{q}-{r}", "score": 0.5} for r in range(3)] for q in range(5)]
    assert len(merge_hits(many)) == MAX_HITS


def test_the_prompt_clips_a_long_passage_but_the_hit_keeps_its_text():
    hit = {"id": "big", "score": 0.9, "text": "x" * (PROMPT_SNIPPET_CHARS * 3)}
    rendered = render_snippets([hit])
    assert len(rendered) < PROMPT_SNIPPET_CHARS + 200 and "cut for the prompt" in rendered
    assert len(hit["text"]) == PROMPT_SNIPPET_CHARS * 3, "entailment still reads it all"
