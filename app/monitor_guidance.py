"""What the monitor says under a run: why it ended that way, and what to do next.

Split out of `app/monitor.py` (2026-11-24, which had grown past pylint's 1000-line
module limit); `app.monitor.status()` attaches `guidance()`, `retry_command()` and
`crash_hint()` to every recent run.

Read-only like the rest of the monitor: every next step is TEXT, and a step that
needs a process says the command to run rather than running it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import config
from adapters.llm_lmstudio import base_url
from adapters.vector_index_local import MANIFEST, namespace_dirname
from adapters.vector_index_local import _root as index_root
from errors import CONFIG_ERRORS


def retry_command(run: dict, runtime: dict, override: dict | None = None) -> str:
    """The exact command that re-runs this review, for the page's Retry button.

    The monitor stays read-only: it hands you the command rather than launching
    it. A page with no authentication that can start processes is a page any
    other local site can start processes through.
    """
    rt = (run.get("audit") or {}).get("runtime") or runtime or {}
    env = {"CRA_PROFILE": rt["profile"]} if rt.get("profile") else {}
    if rt.get("entailer") and rt["entailer"] != "lexical":
        env["ENTAILER"] = rt["entailer"]
    if rt.get("reranker") and rt["reranker"] != "none":
        env["RERANKER"] = rt["reranker"]
    env.update(override or {})
    target = run.get("target") or "staged"
    args = f"--tenant {run.get('tenant') or '<tenant>'}"
    if target != "staged":
        args += f" --target {target}"
    return " ".join([*(f"{k}={v}" for k, v in env.items()), "python -m app.review", args])


# ── what the page says under a run ───────────────────────────────────────────
# The gate's own failure strings (nodes/independent_review.py) are the source of
# truth for WHY a run escalated. The page used to infer the reason from the
# findings' citations instead, and on 2026-11-17 it blamed a finding that had
# PASSED while the one that failed sat just under the bar.
_ENTAIL_FAIL = re.compile(r"^(?P<label>.+?) not entailed by its premise "
                          r"\(score (?P<score>[0-9.]+) < (?P<threshold>[0-9.]+)\)$")
_FINDING_LABEL = re.compile(r"^finding (?P<n>\d+) \((?P<file>.*)\)$")
_SEVERITY_FAIL = re.compile(r"^severity '(?P<sev>[^']*)' not in (?P<allowed>.*)$")
_ACTION_FAIL = re.compile(r"^action '(?P<action>[^']*)' not in (?P<allowed>.*)$")


def _entailment_text(m: re.Match) -> tuple[str, int | None]:
    """An Entailed failure in words, and the finding number it names, if any."""
    score, threshold = float(m["score"]), float(m["threshold"])
    f = _FINDING_LABEL.match(m["label"])
    what = f"Finding {f['n']} ({f['file']})" if f else f"The {m['label']}"
    return (f"{what} scored {score:.3f} on Entailed against a bar of {threshold:.2f} "
            f"({threshold - score:.3f} short): its claim did not clearly follow from the "
            f"code it cites."), (int(f["n"]) if f else None)


# (does it match?, what to say) for the gate's other failure strings, in order.
_FAILURE_TEXT = (
    (lambda t: t == "no citations",
     lambda t, _m: ("The diagnosis cited no retrieved code, so nothing it says could be "
                    "checked against the codebase.")),
    (lambda t: t.startswith("citations not retrieved:"),
     lambda t, _m: ("The model cited passages that retrieval never returned"
                    f"{t[len('citations not retrieved'):]}. A citation that was not "
                    "retrieved is usually invented.")),
    (lambda t: t.startswith("citations below relevance:"),
     lambda t, _m: ("Cited passages scored under the relevance bar"
                    f"{t[len('citations below relevance'):]}.")),
    (_SEVERITY_FAIL.match,
     lambda t, m: (f"The change was rated {m['sev']} severity. This codebase's policy "
                   f"lets only {m['allowed']} through without a human.")),
    (_ACTION_FAIL.match,
     lambda t, m: (f"The proposed action '{m['action']}' is not on this codebase's "
                   f"allowed list {m['allowed']}.")),
)


def _explain_failure(text: str) -> tuple[str, int | None]:
    """One gate failure in plain words, plus the finding number it names, if any."""
    m = _ENTAIL_FAIL.match(text)
    if m:
        return _entailment_text(m)
    for matches, say in _FAILURE_TEXT:
        hit = matches(text)
        if hit:
            return say(text, hit), None
    return text, None


def _cited_files(finding: dict) -> list[str]:
    """The files a finding's citations come from, in order, without duplicates."""
    return list(dict.fromkeys(str(c).split(":", 1)[0] for c in finding.get("citation_ids") or []))


def _unreachable_hint(server: str, local: bool) -> str:
    """What to do when the model server did not answer: local and remote differ."""
    return ("Could not reach the model server. Start it with: lms server start" if local
            else f"Could not reach the model server at {server}. Check that it is up and "
            "that LMSTUDIO_BASE_URL is right.")


def _model_missing_hint(server: str, _local: bool) -> str:
    """What to do when the server does not serve a role's model."""
    return (f"The model this role needs is not served at {server}. "
            "Run `python -m app.doctor` to see which.")


# (substrings that identify the failure, the hint or a function of (server, local)).
# First match wins, so the specific key errors come before the generic connection one.
_CRASH_HINTS = (
    (("LMSTUDIO_API_KEY is not set",),
     "The model server needs an API key and none was loaded. Load your key file in the "
     "shell that runs the review, then retry."),
    (("refused the API key",),
     "The model server refused the API key. Update it in your key file, check it with "
     "doctor, then retry."),
    (("prediction stream", "generation thread", "mlx_engine"),
     "LM Studio's model backend crashed while generating, usually from memory pressure or "
     "a prompt longer than the loaded context. Reload the model, then retry."),
    (("Connection refused", "ConnectionError", "URLError", "APIConnectionError",
      "not reachable"), _unreachable_hint),
    (("context length", "context window", "n_ctx", "too many tokens"),
     "The prompt is longer than the model's context window."),
    (("timed out", "Timeout"),
     "The model call timed out. Retry; if it repeats, the server is overloaded."),
    (("Invalid model identifier", "no model", "No model"), _model_missing_hint),
)


def crash_hint(message: str, server: str) -> str:
    """What to do about a crashed step, for the model server this run used."""
    m = message or ""
    local = any(h in server for h in ("localhost", "127.0.0.1"))
    for needles, hint in _CRASH_HINTS:
        if any(n in m for n in needles):
            return hint(server, local) if callable(hint) else hint
    return "Read the full error on the failed step (Steps tab), fix it, then retry."


def index_built_at(tenant: str) -> str | None:
    """When this tenant's code index was built, or None if unknown or unbuilt."""
    try:
        policy = config.get_tenant_store().policy(tenant, "code_review")
    except CONFIG_ERRORS:
        return None
    repo = (policy.get("codebase") or {}).get("repo")
    if not repo:
        return None
    path = index_root() / namespace_dirname(repo) / MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("built_at")
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class _GateMisses:
    """Which findings the gate named, and by how much Entailed missed.

    `failed` maps a finding number the gate named to its file; `shortfall` is the
    smallest gap under the entailment bar (None if Entailed did not fail). Both
    come from the same pass over the gate's failure strings, so they travel
    together -- and keep `_RunView` within pylint's seven-attribute default.
    """

    failed: dict[int, str]
    shortfall: float | None


@dataclass(frozen=True)
class _RunView:
    """What the guidance reads from one run, parsed once.

    `gate` holds what the gate's failure strings named (see `_GateMisses`).
    """

    run: dict
    runtime: dict
    audit: dict
    review: dict
    tenant: str
    findings: list
    gate: _GateMisses

    @property
    def entailer(self) -> str:
        """The entailer this run used ('lexical' unless the run recorded otherwise)."""
        return ((self.audit.get("runtime") or self.runtime or {}).get("entailer")
                or "lexical")

    def failures_matching(self, pattern: re.Pattern) -> re.Match | None:
        """The first gate failure `pattern` matches, or None."""
        return next((pattern.match(t) for t in self.review.get("failures") or []
                     if pattern.match(t)), None)


def _read_run(run: dict, runtime: dict) -> tuple[_RunView, list[str]]:
    """The run's view, plus one plain-language 'why' line per gate failure."""
    a = run.get("audit") or {}
    review = a.get("review") or {}
    findings = (a.get("diagnosis") or {}).get("findings") or []
    why, failed, gaps = [], {}, []
    for text in review.get("failures") or []:
        line, n = _explain_failure(text)
        why.append(line)
        if n is not None:
            failed[n] = (findings[n - 1].get("file") if 0 < n <= len(findings) else None) or "?"
        m = _ENTAIL_FAIL.match(text)
        if m:
            gaps.append(float(m["threshold"]) - float(m["score"]))
    view = _RunView(run, runtime, a, review, run.get("tenant") or "<tenant>", findings,
                    _GateMisses(failed, min(gaps) if gaps else None))
    return view, why


def _step(who: str, text: str, **extra) -> dict:
    """One next step: who acts, what to do, and any command, tab or list to show."""
    return {"who": who, "text": text, **{k: v for k, v in extra.items() if v}}


def _reviewer_escalated(v: _RunView) -> list[dict]:
    """What the person reading an escalated change does next."""
    failed = v.gate.failed
    others = len(v.findings) - len(failed)
    rest = f"; the other {others} passed." if others else "."
    if len(failed) == 1:
        ((n, file),) = failed.items()
        steps = [_step("reviewer", f"Read finding {n} ({file}) yourself. It is the one the "
                       "gate could not verify" + rest, tab="findings")]
    else:
        # One step listing them: seven near-identical "read finding N" steps were
        # noise an admin had to read seven times.
        steps = [_step("reviewer", f"Read these {len(failed)} findings yourself. The gate "
                       "could not verify them" + rest, tab="findings",
                       items=[f"Finding {n} ({file})" for n, file in failed.items()])] \
            if failed else []
    if v.review.get("groundedness_ok") is False:
        steps.append(_step("reviewer", "Open Evidence to see which citations were missing or "
                           "scored under the bar.", tab="evidence"))
    if v.review.get("risk_ok") is False:
        steps.append(_step("reviewer", "Escalating this severity is the policy working as "
                           "intended: a human reads it before anything is posted."))
    steps.append(_step("reviewer", "Nothing was posted. If the findings hold up, post them "
                       "yourself; the drafted comment is at the bottom of the Findings tab.",
                       tab="findings"))
    return steps


def _admin_entailment(v: _RunView) -> list[dict]:
    """Admin steps for an Entailed failure: a semantic retry, then a measured bar."""
    steps = []
    if v.entailer == "lexical":
        near = f" The closest miss was {v.gate.shortfall:.3f} under the bar." \
            if v.gate.shortfall is not None else ""
        steps.append(_step("admin", "Re-run with the meaning-based entailer. The word-overlap "
                           "entailer this run used also fails correct paraphrases." + near,
                           command=retry_command(v.run, v.runtime,
                                                 {"ENTAILER": "cross_encoder"})))
    threshold = (v.audit.get("gates") or {}).get("min_entailment_score")
    shown = threshold if threshold is not None else "default"
    steps.append(_step("admin", f"If good findings keep failing Entailed, measure {v.tenant}'s "
                       f"entailment bar ({shown}) instead of "
                       "guessing it.",
                       command=f"python -m tools.calibrate --tenant {v.tenant} entailment"))
    return steps


def _admin_policy(v: _RunView) -> list[dict]:
    """Admin steps for Grounded, Risk and Permission failures."""
    steps = []
    if v.review.get("groundedness_ok") is False:
        steps.append(_step("admin", "If the cited passages look relevant but scored under the "
                           f"bar, measure {v.tenant}'s relevance bar.",
                           command=f"python -m tools.calibrate --tenant {v.tenant} relevance"))
    m = v.failures_matching(_SEVERITY_FAIL)
    if m:
        steps.append(_step("admin", f"To let {m['sev']} severity through without a human, add "
                           f"it to severity_allowlist in tenants/{v.tenant}/policy.json. That "
                           "is a policy decision, not a fix."))
    m = v.failures_matching(_ACTION_FAIL)
    if m:
        steps.append(_step("admin", f"If '{m['action']}' should be allowed here, add it to "
                           f"allowed_actions in tenants/{v.tenant}/policy.json."))
    return steps


def _guide_escalated(v: _RunView, _why: list[str]) -> list[dict]:
    """Escalated: reviewer steps first, then the admin's. `why` is already the gate's."""
    admin = _admin_entailment(v) if v.review.get("entailment_ok") is False else []
    return _reviewer_escalated(v) + admin + _admin_policy(v)


def _guide_crashed(v: _RunView, why: list[str]) -> list[dict]:
    """Crashed: say where, then the server-aware hint and the retry."""
    err = v.run.get("error") or {}
    why.append(f"Crashed at {err.get('step')}: {_first_line(err.get('message'))}")
    server = (v.audit.get("runtime") or v.runtime or {}).get("model_server") or base_url()
    return [_step("admin", crash_hint(err.get("message") or "", server), tab="steps"),
            _step("admin", "Then re-run it.", command=retry_command(v.run, v.runtime))]


def _guide_incomplete(v: _RunView, why: list[str]) -> list[dict]:
    """Stopped without a record: name the last step, offer the retry."""
    last = (v.run.get("steps") or [{}])[-1].get("name") or "start"
    why.append(f"Stopped after {last} without writing a record. The process was "
               "probably killed (Ctrl-C, a closed terminal or a crash outside Python).")
    return [_step("admin", "Re-run it.", command=retry_command(v.run, v.runtime))]


def _guide_awaiting(_v: _RunView, why: list[str]) -> list[dict]:
    """Paused at the approval prompt."""
    why.append("Passed the checks and is paused at the approval prompt.")
    return [_step("reviewer", "Answer the y/n prompt in the terminal that started this run. "
                  "Nothing is written until you do.")]


def _guide_abandoned(v: _RunView, why: list[str]) -> list[dict]:
    """Nobody answered the prompt."""
    why.append("Stopped at the approval prompt with nobody to answer it. Nothing was written.")
    return [_step("reviewer", "Re-run it and answer the prompt.",
                  command=retry_command(v.run, v.runtime))]


def _guide_executed(v: _RunView, why: list[str]) -> list[dict]:
    """Approved and written: say where."""
    why.append("Passed all four checks and was approved.")
    path = ((v.audit.get("result") or {}).get("review") or {}).get("markdown")
    return [_step("reviewer", "Read the written review" + (f": {path}" if path else "."))]


def _guide_denied(_v: _RunView, why: list[str]) -> list[dict]:
    """Declined at the prompt: nothing to do."""
    why.append("Passed the checks, but was declined at the approval prompt. "
               "Nothing was written.")
    return []


def _guide_human_review(_v: _RunView, why: list[str]) -> list[dict]:
    """Routed to a human by rule."""
    why.append("Routed straight to a human by rule. No model looked at it.")
    return [_step("reviewer", "Review the change yourself; the agent deliberately skips "
                  "this kind of change.")]


_STATUS_GUIDES = {"escalated": _guide_escalated, "crashed": _guide_crashed,
                  "incomplete": _guide_incomplete, "awaiting_approval": _guide_awaiting,
                  "abandoned": _guide_abandoned, "executed": _guide_executed,
                  "denied": _guide_denied, "human_review": _guide_human_review}


def _off_target_steps(v: _RunView, index_built: str | None) -> list[dict]:
    """Findings citing only other files: a re-ingest hint, with which passed or failed."""
    off = [(n, f) for n, f in enumerate(v.findings, start=1)
           if f.get("file") and _cited_files(f) and f["file"] not in _cited_files(f)]
    if not off:
        return []
    built = f" ({index_built})" if index_built else ""
    return [_step("admin", "Some findings cite only other files: retrieval returned no code "
                  "from the file they are about. If those files are new or changed since the "
                  f"index was built{built}, re-ingest it.",
                  items=[f"Finding {n} ({f['file']}) cites {', '.join(_cited_files(f))}"
                         + (" — failed the gate" if n in v.gate.failed else " — passed the gate")
                         for n, f in off],
                  command=f"python -m tools.ingest --tenant {v.tenant}")]


def _prepush_steps(v: _RunView) -> list[dict]:
    """The pre-push flags, grouped by file and check, as one reviewer step."""
    flags = [c for c in v.audit.get("checks") or [] if c.get("status") == "flag"]
    if not flags:
        return []
    by_file: dict[str, Counter] = {}
    for c in flags:
        where = c.get("file") or "(whole change)"
        by_file.setdefault(where, Counter())[c.get("check") or "?"] += 1
    plural = "s" if len(flags) > 1 else ""
    return [_step("reviewer", f"Fix the {len(flags)} pre-push flag{plural}. Code computed "
                  "these, not a model, so they are facts about the change.", tab="prepush",
                  items=[f"{file}: " + ", ".join(f"{k.replace('_', ' ')} ×{n}"
                                                 for k, n in counts.items())
                         for file, counts in by_file.items()])]


def guidance(run: dict, runtime: dict, index_built: str | None = None) -> dict:
    """Why a run ended the way it did, and the next steps, split by who acts.

    `why` holds plain-language facts read from the audit record. `steps` holds
    actions, each with `who` ("reviewer": whoever reads this change; "admin":
    whoever runs the agent and owns its config) and optionally a `command` to
    copy, a `tab` of the run panel to open, and `items` to list.
    """
    view, why = _read_run(run, runtime)
    guide = _STATUS_GUIDES.get(run.get("status"))
    steps = guide(view, why) if guide else []
    return {"why": why,
            "steps": steps + _off_target_steps(view, index_built) + _prepush_steps(view)}


def _first_line(message) -> str:
    """The first line of an error, without its traceback, capped for the page."""
    text = str(message or "")
    return text.split("\n", 1)[0].split("Traceback", 1)[0].strip()[:240]
