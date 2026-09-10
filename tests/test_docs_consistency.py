"""The numbers in the docs must match the code that produces them.

Three numbers in this repo have now drifted from reality **twice**: the gate's
boolean count (three -> four when T22 added entailment), the gate-suite total
(24 -> 36), and the embedder's dimensionality (768 -> 1024). Each was correct
when written and silently became wrong.

That is exactly the shape of defect this project already guards against
elsewhere -- `test_registered_report_includes_hardware` keeps `embedding_dims` as
a literal precisely so changing the embedder cannot be free. This module extends
that reasoning to the prose, because `MODEL_CARD.md` is the file that states the
hard limits and `README.md` is the first thing anyone reads: a doc that
misdescribes the safety design is worse than no doc, and a doc that *understates*
what is built invites someone to rebuild it.

Deliberately asserts against the FILES rather than a constant. A test comparing
one constant to another would be tautological; the point is that the human-facing
sentence has to be updated too. (T24)
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The docs that describe the CURRENT system. `DECISIONS.md` and `TODO.md` are
# excluded on purpose: they are append-only history and legitimately quote the
# old numbers while recording that they changed.
CURRENT_STATE_DOCS = ("README.md", "MODEL_CARD.md", "CLAUDE.md",
                      "docs/ARCHITECTURE.md")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _docs():
    return [(rel, _read(rel)) for rel in CURRENT_STATE_DOCS if (ROOT / rel).exists()]


def test_gate_count_in_the_docs_matches_what_the_harness_actually_runs():
    """MODEL_CARD claims a number of regression tasks. Ask the harness.

    The harness is the source of truth here rather than a constant, because the
    number it prints is the number of checks it really performed -- which is the
    claim the doc is making on its behalf.
    """
    from eval import harness

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = harness.run()
    out = buf.getvalue()

    assert rc == 0, f"the gate suite must pass; harness returned {rc}\n{out}"
    m = re.search(r"ALL (\d+) GATES PASS", out)
    assert m, f"could not find the gate total in harness output:\n{out[-500:]}"
    actual = int(m.group(1))

    card = _read("MODEL_CARD.md")
    claimed = re.search(r"covered by (\d+) regression tasks", card)
    assert claimed, "MODEL_CARD.md no longer states a gate count in the expected phrasing"
    assert int(claimed.group(1)) == actual, (
        f"MODEL_CARD.md claims {claimed.group(1)} regression tasks; the harness "
        f"runs {actual}. Update the doc in the same change as the tasks."
    )


def test_the_gate_has_four_booleans_and_the_docs_say_so():
    """`ReviewResult`'s `*_ok` fields ARE the gate. The docs must agree."""
    from schemas.diagnosis import ReviewResult

    booleans = sorted(
        name for name, field in ReviewResult.model_fields.items()
        if name.endswith("_ok") and field.annotation is bool
    )
    assert booleans == ["entailment_ok", "groundedness_ok", "permission_ok",
                        "risk_ok"], booleans
    assert len(booleans) == 4

    # The stale phrasings, which is what actually drifted. A doc may say "four
    # booleans" or "4 Python booleans"; it may no longer say "three".
    stale = re.compile(r"three[- ](?:python\s+)?booleans|three-boolean|"
                       r"\b3\s+(?:python\s+)?booleans", re.IGNORECASE)
    offenders = []
    for rel, text in _docs():
        for match in stale.finditer(text):
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{rel}:{line} -> {match.group(0)!r}")
    assert not offenders, (
        "the gate has FOUR booleans (entailment_ok was added by T22); these "
        "still say three:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("rel", CURRENT_STATE_DOCS)
def test_no_current_state_doc_claims_the_old_embedding_dimensionality(rel):
    """768 was nomic's; the pin is bge-m3 at 1024.

    This one matters more than it looks: a stated dimensionality is a claim about
    which vector space every `min_relevance_score` in `tenants/*/policy.json`
    lives in. Getting it wrong in prose invites someone to trust a threshold that
    was calibrated for a different space.
    """
    if not (ROOT / rel).exists():
        pytest.skip(f"{rel} not present")
    text = _read(rel)
    bad = [
        text[:m.start()].count("\n") + 1
        for m in re.finditer(r"768[- ]dimension|768\s*dims|\b768\b", text)
    ]
    assert not bad, (
        f"{rel} still refers to 768 dimensions at line(s) {bad}; the pinned "
        f"embedder is bge-m3 at 1024. (Historical mentions belong in "
        f"DECISIONS.md / TODO.md, which this test excludes.)"
    )
