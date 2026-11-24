"""Web standards, and the standards no reviewer can measure from a diff.

  * css_empty_rules -- an empty ruleset (`.x {}`) in a stylesheet or a <style>
    block. It does nothing except suggest a style that is not there.
  * standards_unmeasured -- SKIP lines for the standards a code reader cannot
    decide: the rest of SOLID, duplicate code, the control-flow choice,
    serializer-per-URL, O(1) queries per request. Listed, so "not checked" is
    never read as "fine".
"""
from __future__ import annotations

import re
from collections import Counter

from tools.prepush.change import FLAG, SKIP, Change, _skip_path, report, result
from tools.prepush.standards.engine import enabled

RE_EMPTY_RULE = re.compile(r"([^{};]+)\{\s*\}")
RE_STYLE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)

UNMEASURED = (
    ("solid", "SOLID beyond size (open/closed, substitution, interfaces, dependency "
              "inversion) needs a person reading the design"),
    ("duplicate_code", "duplicate code (pylint R0801) needs the whole repository, not a diff"),
    ("control_flow", "whether match, dispatch, guard clauses or generators fit best is a "
                     "judgement call (CONTROL_FLOW_STANDARD.md)"),
    ("serializer_per_url", "one serializer per URL, separate request and response "
                           "serializers, and exactly the agreed response fields need the API "
                           "contract"),
    ("query_count", "O(1) queries per request needs a live database (db_operations)"),
    ("pagination_contract", "paginated endpoints declaring page/page_size through the "
                            "pagination mixin on their query serializer needs the view and "
                            "its serializer together, across files"),
)


def _css_blocks(path: str, text: str) -> list[tuple[int, str]]:
    """(offset in the file, css) for each block of CSS: a stylesheet is one block,
    a page contributes only its <style> blocks (never its scripts)."""
    if path.endswith(".css"):
        return [(0, text)]
    return [(m.start(1), m.group(1)) for m in RE_STYLE.finditer(text)]


def empty_rules(path: str, text: str) -> list[tuple[str, int]]:
    """(selector, line) for each empty ruleset, the line being the selector's own."""
    out = []
    for start, css in _css_blocks(path, text):
        for m in RE_EMPTY_RULE.finditer(css):
            raw = m.group(1)
            at = start + m.start(1) + (len(raw) - len(raw.lstrip()))
            out.append((" ".join(raw.split()), text.count("\n", 0, at) + 1))
    return out


def check_css_empty_rules(change: Change, config: dict) -> list[dict]:
    """No NEW empty CSS ruleset in a stylesheet or a page's <style> blocks."""
    if not enabled(config, "css_empty_rules"):
        return [result("css_empty_rules", 1, SKIP, "disabled in this codebase's standards config")]
    flags = []
    for path in (f for f in change.files if f.endswith((".css", ".html"))):
        text = change.content_for(path)
        if text is None or _skip_path(path, config):
            continue
        budget = Counter(sel for sel, _ in empty_rules(path, change.base_for(path)))
        for selector, line in empty_rules(path, text):
            if budget[selector]:
                budget[selector] -= 1
                continue
            flags.append(result("css_empty_rules", 1, FLAG,
                                f"empty ruleset `{selector} {{}}`: delete it", path, line))
    return report("css_empty_rules", 1, flags, "empty rulesets", "no new empty CSS rulesets")


def check_standards_unmeasured(_change: Change, config: dict) -> list[dict]:
    """One SKIP per standard that needs a person, a database or the whole repo."""
    return [result(name, 1, SKIP, f"not checked: {why}") for name, why in UNMEASURED
            if enabled(config, name)]
