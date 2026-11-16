"""Redact secrets and PII, and defang the fence, BEFORE evidence reaches a model.

This closes G5/T12 — until now the project's own invariant "evidence is untrusted
DATA, never instructions" was enforced by **prompt wording plus a delimiter**,
and `nodes/_render.py` said so in its docstring: saying "treat this as data" is
not a boundary.

## Two jobs, and only one of them is solvable by regex

**Redaction is solvable.** A diff can contain an AWS key and a log line can
contain a customer's email. Those have shapes, so code can find and replace them,
and the model never sees them. That is a real boundary: a secret that never
enters the prompt cannot be echoed into a review comment, cannot reach the
provider, and cannot land in `var/` — which matters twice over now that the audit
and trace stores keep evidence-derived text (docs/DATA_STORAGE.md).

**Neutralising natural-language injection is NOT solvable by regex, and this
adapter does not pretend otherwise.** It does two honest things instead:

  1. **Defangs fence forgery.** `_render.py` wraps untrusted material in
     `<<<UNTRUSTED_EVIDENCE...>>>` delimiters. If the evidence itself contains
     that delimiter, it can close the fence early and continue as though it were
     operator text — which makes the delimiter worth *nothing*. Escaping those
     markers is the one structural attack regex genuinely defeats, and without it
     the whole spotlighting approach is decorative.
  2. **Counts instruction-like phrasing without rewriting it.** Rewriting prose
     would silently alter the evidence a human is about to approve a comment
     about, and would still miss any paraphrase. So suspicion is *reported* —
     into the audit record and the trace — where a human and an eval can see it.
     A rising `injection_suspected` count is a signal; a regex that claimed to
     have removed the problem would be a lie.

## What the report may contain

**Never the redacted value.** The report goes into the audit record, so putting
the secret in it would move the leak rather than stop it. Counts and category
names only.

## Placeholders preserve coreference without revealing anything

`[REDACTED:email:1]`, numbered per category in order of first appearance, so the
same address is the same token twice and a reviewer can still see "these two
lines are about one user". Deliberately NOT a truncated hash of the value: a hash
of a low-entropy string like an email is brute-forceable from a candidate list,
which would quietly reintroduce what redaction removed.

## Not tenant-overridable

Thresholds and allowlists are tenant-tunable; this is not. Redaction is a
boundary, not a value, and `docs/MULTI_TENANT.md` is explicit that tenants tune
values and never remove a boundary. `sanitize()` still takes `tenant` because the
port does and because the report names it — so an audit record says whose
evidence was cleaned — but no tenant can switch a category off.
"""
from __future__ import annotations

import re

from registry import adapter

# (category, pattern). ORDERED, first match wins — the same discipline as
# adapters/detectors.py, so a private-key block is not also counted as a dozen
# base64 blobs. Specific and high-confidence patterns come first; the broad
# "assignment that looks like a secret" rule comes last so it cannot swallow
# something a precise rule would have labelled properly.
PATTERNS: tuple[tuple[str, str], ...] = (
    # ── secrets ──────────────────────────────────────────────────────────────
    ("private_key",
     r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----"),
    ("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    # Credentials inside a URL — the shape a DATABASE_URL leaks in.
    ("url_credentials", r"\b([a-z][a-z0-9+.\-]*)://[^\s:/@]+:[^\s/@]+@"),
    ("bearer_token", r"\b[Bb]earer\s+[A-Za-z0-9._\-]{16,}"),
    # ── PII ──────────────────────────────────────────────────────────────────
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Indian mobile (the MilesOne case) plus a generic international form.
    ("phone", r"(?<![\w.])(?:\+91[\s-]?|0)?[6-9]\d{9}(?![\w.])"
              r"|(?<![\w.])\+\d{1,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3,4}(?![\w.])"),
    # ── the catch-all, LAST ──────────────────────────────────────────────────
    # `SECRET_KEY = "..."`, `api_token: ...`, `password=...`. Requires a
    # secret-ish NAME rather than guessing from entropy, because entropy alone
    # flags every commit hash and base64 asset in a diff.
    ("secret_assignment",
     r"(?i)\b(?:api[_-]?key|secret[_-]?key|secret|password|passwd|pwd|token|"
     r"access[_-]?key|private[_-]?key|client[_-]?secret)\b\s*[:=]\s*"
     r"[\"']?([A-Za-z0-9/+_\-\.]{8,})[\"']?"),
)

# Phrasing that tries to talk to the model rather than describe the change.
# COUNTED, never rewritten — see the module docstring.
INJECTION_HINTS: tuple[tuple[str, str], ...] = (
    ("override_instructions",
     r"(?i)\b(?:ignore|disregard|forget|override)\b[^.\n]{0,30}"
     r"\b(?:above|previous|prior|earlier|all)\b[^.\n]{0,20}"
     r"\b(?:instruction|prompt|rule|direction|context)"),
    ("role_reassignment",
     r"(?i)\b(?:you are now|act as|pretend to be|from now on you)\b"),
    ("fake_turn",
     r"(?i)^\s*(?:system|assistant|user)\s*:", ),
    ("approval_coercion",
     r"(?i)\b(?:approve|auto[- ]?approve|mark)\b[^.\n]{0,20}"
     r"\b(?:this|it)\b[^.\n]{0,20}\b(?:safe|approved|low[- ]risk)\b"),
    ("exfiltration",
     r"(?i)\b(?:print|reveal|output|repeat|show)\b[^.\n]{0,20}"
     r"\b(?:system prompt|your instructions|the prompt)\b"),
)


@adapter("guard", "regex")
class RegexInputGuard:
    """The default `InputGuard`. Code, not a model — and that is deliberate.

    It runs on every piece of evidence before every LLM call, including over log
    windows, so it must be cheap enough to touch everything. It also has to work
    with no provider reachable, for the same reason the lexical `Entailer` does:
    a boundary that needs a model to be up is not a boundary.

    `fence_markers` are injected by `nodes/_render.py` rather than imported, so
    this adapter does not depend on the rendering module it protects.
    """

    def __init__(self, patterns=None, hints=None, fence_markers=()):
        self.patterns = [(name, re.compile(pat, re.DOTALL))
                         for name, pat in (patterns or PATTERNS)]
        self.hints = [(name, re.compile(pat, re.MULTILINE))
                      for name, pat in (hints or INJECTION_HINTS)]
        self.fence_markers = tuple(fence_markers)

    def sanitize(self, text: str, *, tenant: str = "_default") -> tuple[str, dict]:
        """Return (cleaned text, report). The report never carries a redacted value."""
        if not text:
            return "", {"tenant": tenant, "redacted": {}, "total_redactions": 0,
                        "injection_suspected": {}, "fence_forgery": 0}

        counts: dict[str, int] = {}
        # category -> {original: placeholder}, so a repeated value reuses its
        # number and coreference survives redaction.
        seen: dict[str, dict[str, str]] = {}

        def replace(category: str, match: re.Match) -> str:
            value = match.group(0)
            bucket = seen.setdefault(category, {})
            if value not in bucket:
                bucket[value] = f"[REDACTED:{category}:{len(bucket) + 1}]"
                counts[category] = counts.get(category, 0) + 1
            return bucket[value]

        cleaned = text
        for category, pattern in self.patterns:
            cleaned = pattern.sub(lambda m, c=category: replace(c, m), cleaned)

        # Fence forgery: neutralise the delimiter so evidence cannot close its own
        # fence. Replaced with a visible marker rather than deleted, because a
        # human reading the captured prompt should see that this was ATTEMPTED.
        forgery = 0
        for marker in self.fence_markers:
            if not marker:
                continue
            found = cleaned.count(marker)
            if found:
                forgery += found
                cleaned = cleaned.replace(marker, "[NEUTRALISED_FENCE_MARKER]")

        suspected = {}
        for name, pattern in self.hints:
            hits = len(pattern.findall(cleaned))
            if hits:
                suspected[name] = hits

        return cleaned, {
            "tenant": tenant,
            "redacted": counts,                       # category -> distinct values
            "total_redactions": sum(counts.values()),
            "injection_suspected": suspected,         # reported, NOT removed
            "fence_forgery": forgery,
        }


@adapter("guard", "null")
class NullInputGuard:
    """Passes text through untouched. For tests that assert on raw evidence.

    Not the default, and `CRA_GUARD=null` has to be set deliberately — the same
    reasoning as the audit sink: a boundary you get by forgetting is not one.
    """

    def __init__(self, fence_markers=()):
        """Same constructor shape as the real guard, so config needs no fallback."""
        self.fence_markers = tuple(fence_markers)

    def sanitize(self, text: str, *, tenant: str = "_default") -> tuple[str, dict]:
        return text, {"tenant": tenant, "redacted": {}, "total_redactions": 0,
                      "injection_suspected": {}, "fence_forgery": 0, "guard": "null"}
