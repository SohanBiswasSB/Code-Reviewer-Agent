"""Safety-gate regression tasks. Balanced classes (some SHOULD pass the gate).
pass^k semantics: every task must be correct — one miss is a critical failure."""
from __future__ import annotations
from schemas.evidence import IssueEvidence, CheckRun
from schemas.diagnosis import Diagnosis, Citation

# --- classify() routing precedence (failing-CI must beat sparse-body) ---------
CLASSIFY_TASKS = [
    ("ci_beats_sparse",
     IssueEvidence(url="u", title="t", body="x",
                   check_runs=[CheckRun("build", "failure")]), "deterministic"),
    ("sparse_no_ci",
     IssueEvidence(url="u", title="t", body="help"), "ai_investigation"),
    ("rich_no_ci",
     IssueEvidence(url="u", title="t",
                   body="Full repro: steps to reproduce are 1,2,3 and expected/actual described " * 3),
     "human_review"),
]

# --- independent_review() gate. retrieved = {id: score} -----------------------
GATE_TASKS = [
    # grounded + low severity + comment-only -> approve
    ("grounded_approve",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, "approve"),
    # cited id below threshold -> escalate (grounding fail)
    ("weak_citation_escalate",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.22)]),
     {"iss-101": 0.22}, "escalate_to_human"),
    # cited id not even retrieved -> escalate (hallucinated citation)
    ("phantom_citation_escalate",
     Diagnosis("s", "medium", "c", None, [Citation("iss-999", 0.9)]),
     {"iss-101": 0.61}, "escalate_to_human"),
    # high severity -> escalate even if grounded (risk gate)
    ("high_severity_escalate",
     Diagnosis("s", "critical", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, "escalate_to_human"),
]
