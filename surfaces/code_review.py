"""Code review: a PR diff in, a grounded review comment out.

This surface owns its own `normalize` because its evidence SHAPE differs from
issue triage. Without it the graph handed this surface an `IssueEvidence` — which
carries no `touched_areas` — so `classify` returned `human_review` for every
change. That failed safe, but it made the surface useless, and the failure was
invisible because `human_review` is a legitimate route.
"""
from __future__ import annotations

from registry import surface

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


def _touched_areas(files: list[str]) -> list[str]:
    """Which risky areas this change touches, by path substring."""
    joined = " ".join(files).lower()
    return [area for area in RISKY_AREAS if area in joined]


def _is_trivial(files: list[str]) -> bool:
    if not files:
        return False
    return all(
        f.lower().endswith(TRIVIAL_SUFFIXES) or f.lower().startswith(TRIVIAL_DIRS)
        for f in files
    )


@surface("code_review")
class CodeReviewSurface:
    name = "code_review"
    source_name, sink_name = "github", "pr_comment"
    # Node KEYS, resolved per tenant by the TenantStore -- not flat paths under
    # prompts/. The flat files duplicate tenants/_default/prompts/** and are not
    # the canonical copy (T19).
    diagnosis_prompt = "diagnosis"
    judge_prompt = "judge"

    def normalize(self, raw: dict) -> dict:
        """Provider payload -> code-review evidence.

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
            "touched_areas": _touched_areas(files),
            "has_failing_ci": any(c.get("conclusion") == "failure"
                                  for c in checks if isinstance(c, dict)),
            "is_trivial": _is_trivial(files),
        }

    def action_type(self, diagnosis) -> str:
        """What the gate checks against this tenant's `allowed_actions`.

        The label variant is a distinct permission on purpose: on GitHub,
        commenting is open to any authenticated user while labelling needs
        triage/write access, so a tenant may legitimately allow one and not
        the other.
        """
        return "comment+label" if diagnosis.proposed_label else "comment"

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
        if evidence.get("is_trivial"):
            return "human_review"
        return "human_review"
