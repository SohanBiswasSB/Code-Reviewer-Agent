"""Code review: a PR diff in, a grounded review comment out.

This surface owns its own `normalize` because its evidence SHAPE differs from
issue triage. Without it the graph handed this surface an `IssueEvidence` — which
carries no `touched_areas` — so `classify` returned `human_review` for every
change. That failed safe, but it made the surface useless, and the failure was
invisible because `human_review` is a legitimate route.
"""
from __future__ import annotations

from registry import surface
from tools.filekinds import DATA, file_rank
from tools.prepush.change import split_diff

# Paths whose changes get the expensive, LLM-backed investigation. Deliberately
# path-based and blunt: the whole value of a rule-based classify is that it is
# cheap, deterministic and testable, so it routes on what is visible in the diff
# rather than trying to infer risk. A change whose risk is NOT visible in its
# paths will be mis-routed -- to `human_review`, which is the safe direction.
RISKY_AREAS = ("migrations", "auth", "permission", "credential", "credly",
               "celery", "routing", "webhook", "billing", "payment")

# Changes that cannot break behaviour on their own.
TRIVIAL_SUFFIXES = (".md", ".rst", ".txt")
TRIVIAL_DIRS = ("docs/", "doc/", ".github/")


def risky_areas_for(policy: dict | None) -> tuple[str, ...]:
    """The surface's defaults UNION the tenant's `risky_areas` (T27).

    A UNION, never a replacement: a tenant ADDS what is risky in its codebase
    (`subscription` in Masterclass, `zoom` in MilesOne) but cannot make
    `migrations` or `auth` routine. Tenants tune values, never remove a boundary.

    Until 2026-11-17 the tenant list was decorative -- every policy.json declared
    it and no code read it, so a Masterclass change touching only Subscriptions/
    went to human_review. That failed safe, which is why nobody saw it (the T5
    shape).
    """
    extra = [str(a).lower() for a in ((policy or {}).get("risky_areas") or [])]
    return tuple(dict.fromkeys([*RISKY_AREAS, *extra]))


def _touched_areas(files: list[str], areas: tuple[str, ...] = RISKY_AREAS) -> list[str]:
    """Which risky areas this change touches, by path substring."""
    joined = " ".join(files).lower()
    return [area for area in areas if area in joined]


def _is_trivial(files: list[str]) -> bool:
    if not files:
        return False
    return all(
        f.lower().endswith(TRIVIAL_SUFFIXES) or f.lower().startswith(TRIVIAL_DIRS)
        for f in files
    )


# Per-file retrieval (see CodeReviewSurface.build_queries). At most this many files
# get their own query, and each query carries this much of the file's diff -- the
# embedder's context is 2,048 tokens, and the hunk's first lines say most.
MAX_QUERY_FILES = 6
PER_FILE_QUERY_CHARS = 1500

@surface("code_review")
class CodeReviewSurface:
    name = "code_review"
    source_name, sink_name = "github", "pr_comment"
    # Node KEYS, resolved per tenant by the TenantStore -- not flat paths under
    # prompts/. The flat files duplicate tenants/_default/prompts/** and are not
    # the canonical copy (T19).
    # nodes/prepush_checks.py runs the deterministic change checks only for
    # surfaces that say so; a log window has no diff to check.
    runs_change_checks = True
    diagnosis_prompt = "diagnosis"
    judge_prompt = "judge"

    def normalize(self, raw: dict, policy: dict | None = None) -> dict:
        """Provider payload -> code-review evidence.

        `policy` is the tenant's, pinned in state by resolve_tenant; its
        `risky_areas` widen what counts as risky (T27, `risky_areas_for`).

        A dict rather than a dataclass, for now, precisely because it must survive
        the checkpoint without adding another unregistered msgpack type (T15).
        Becomes a Pydantic model under T11.
        """
        files = [f.get("filename", "") for f in raw.get("files", []) if isinstance(f, dict)]
        checks = raw.get("check_runs", []) or []
        return {
            "url": raw.get("html_url", ""),
            "title": raw.get("title", ""),
            "body": raw.get("body", "") or "",
            "diff": raw.get("diff", ""),
            "files": files,
            "touched_areas": _touched_areas(files, risky_areas_for(policy)),
            "has_failing_ci": any(c.get("conclusion") == "failure"
                                  for c in checks if isinstance(c, dict)),
            "is_trivial": _is_trivial(files),
            "revision": raw.get("revision") or {},
        }

    def action_type(self, diagnosis) -> str:
        """What the gate checks against this tenant's `allowed_actions`.

        The label variant is a distinct permission on purpose: on GitHub,
        commenting is open to any authenticated user while labelling needs
        triage/write access, so a tenant may legitimately allow one and not
        the other.
        """
        return "comment+label" if diagnosis.proposed_label else "comment"

    def build_queries(self, evidence: dict) -> list[str]:
        """One retrieval query PER CHANGED FILE: its path, then its own diff.

        A single query for the whole change found evidence for whichever file
        dominated the embedding. On the first real Masterclass run it returned 5
        passages for a 15-file change -- two chunks of admin.py and three docs --
        so 4 of the 5 findings cited passages from a DIFFERENT file than the one
        they were about, and the entailment check correctly failed every run.

        Code first, then config, then docs, by the same rank the git source packs
        the diff with; data files are never queried (their content is records,
        not code). A file whose diff fell past the budget still gets a path-only
        query, which BM25 matches on. The whole-change query goes last, as
        context, so it never crowds out a file.
        """
        chunks = split_diff(evidence.get("diff", ""))
        files = sorted((f for f in evidence.get("files", []) if file_rank(f) != DATA),
                       key=lambda f: (file_rank(f), evidence["files"].index(f)))
        queries = [f"{path}\n{chunks.get(path, '')[:PER_FILE_QUERY_CHARS]}".strip()
                   for path in files[:MAX_QUERY_FILES]]
        whole = self.build_query(evidence)
        return [*queries, whole] if whole else queries

    def build_query(self, evidence: dict) -> str:
        """Retrieval query. Paths matter as much as prose for finding prior work
        on the same modules, so they are part of the query rather than dropped."""
        parts = [evidence.get("title", ""), " ".join(evidence.get("files", [])),
                 evidence.get("diff", "")[:2000]]
        return "\n".join(p for p in parts if p).strip()

    def classify(self, evidence: dict) -> str:
        """Rule-based route. Precedence: failing CI, then risk, then triviality.

        Failing CI wins because a red build has a cheap deterministic response and
        does not need an LLM to reach it -- the same precedence the issue-triage
        path locks in its eval tasks.
        """
        if evidence.get("has_failing_ci"):
            return "deterministic"
        if evidence.get("touched_areas"):
            return "ai_investigation"
        # Trivial or not, anything else is a human's call (invariant 12: an
        # ambiguous route goes to human_review). There used to be a separate
        # `is_trivial` branch returning the same thing.
        return "human_review"
