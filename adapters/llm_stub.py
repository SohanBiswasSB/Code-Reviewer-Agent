"""A deterministic, offline LLM. The default ONLY where a model must not be needed.

Exists because `config.get_llm(role)` had no caller outside tests: both LLM nodes
defaulted to `llm=None` and took a hardcoded stub branch, while `retriever`,
`entailer`, `source` and `sink` all resolved from config. That asymmetry meant the
graph could never reach a real model -- the T5 shape, where a correct component is
simply never called.

Removing the node stubs needs somewhere for the offline path to go, because
`eval/harness.py` must stay **dependency-free and model-free**: it is a safety
regression suite at a permanent 100% bar, and it has to give byte-identical
results on the Mac and the CUDA box. A gate suite that needs a GPU to run is not a
gate suite. Same argument that makes the lexical `Entailer` the default.

## This goes THROUGH `_parse_diagnosis`, and that is the point

The old in-node stub constructed a `Diagnosis` object directly, so the offline
path never exercised the parser -- the exact place the first live run failed
(a model that emits unparseable JSON). A stub that returns TEXT, and is then
parsed by the same code a real reply goes through, tests strictly more.

## Determinism is a hard requirement, not a nicety

`eval/harness.py` compares exact decisions. Everything here is derived from the
prompt by rule -- no randomness, no clock, no id generation -- so the same input
always yields the same bytes.
"""
from __future__ import annotations

import json
import re

from registry import adapter

# `nodes/_render.py::render_snippets` emits "- id=<id> relevance=<score> :: <text>".
# Parsed rather than passed in because the LLM port only ever sees the rendered
# prompt -- which is also what a real model sees, so the stub is grounded in the
# same information a real one has.
_SNIPPET = re.compile(r"^- id=(?P<id>\S+) relevance=(?P<score>\S+) :: (?P<text>.*)$",
                      re.MULTILINE)


@adapter("llm", "stub")
class StubLLM:
    """Offline stand-in. `LLM_PROVIDER=stub`, or injected directly in a test."""

    def __init__(self, model: str = "stub", runtime: dict | None = None,
                 role: str | None = None):
        self.model = model
        self.runtime = runtime or {}
        self.role = role
        # Resolved the same way the real adapter does, so the stub is a faithful
        # stand-in rather than a thing with its own settings. A test asserting
        # per-role temperature should not have to care which provider is active.
        from adapters.llm_lmstudio import _temperature_for
        self.temperature = _temperature_for(role)

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """A schema request means diagnosis; free text means judge.

        Keyed on `json_schema` rather than on `self.role`, so a directly-injected
        stub behaves correctly without anyone remembering to name its role.
        """
        if json_schema is None:
            return self._judge(user)
        return self._diagnosis(user)

    # ── the two roles ────────────────────────────────────────────────────────
    def _diagnosis(self, user: str) -> str:
        """Cite the top retrieved snippet and make a claim that FOLLOWS from it.

        Deriving the claim from the cited passage is load-bearing, not decorative.
        An earlier stub asserted "possible duplicate of a known auth regression",
        which entails nothing about whatever was actually retrieved -- so once the
        entailment check existed (T22) every offline run escalated, and the stub
        stopped modelling a *grounded* diagnosis at all.

        No score is emitted. The gate compares against the retriever's own number
        and `_parse_diagnosis` looks it up; a model-supplied score would be
        gameable, which is why the request schema omits the field (T17).
        """
        top = _SNIPPET.search(user)
        if top is None:
            return json.dumps({
                "summary": "[stub] nothing was retrieved",
                "severity": "high",          # nothing to ground on -> escalate
                "proposed_comment": "No related prior issue was retrieved.",
                "proposed_label": None,
                "citations": [],
            })
        return json.dumps({
            "summary": "[stub] likely regression in the area described by the cited issue",
            "severity": "medium",
            "proposed_comment": f"Related to a known issue: {top['text']}.",
            "proposed_label": "needs-triage",
            "citations": [{"snippet_id": top["id"]}],
        })

    def _judge(self, user: str) -> str:
        """Free text only. It cannot affect the decision -- four booleans do that.

        Says what it actually checked rather than offering a verdict, matching the
        judge prompt's own instruction that a verdict "will be ignored".
        """
        ids = [m["id"] for m in _SNIPPET.finditer(user)]
        seen = f" against retrieved ids {ids}" if ids else " with nothing retrieved"
        return (f"[stub] citations checked{seen}, and relevance compared to the "
                f"configured threshold. No verdict is offered here; the gate decides.")

    def __repr__(self) -> str:
        return f"StubLLM(role={self.role!r})"
