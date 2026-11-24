"""Coding standards the reviewer holds every change to, framework-neutral at the core.

  * general.py -- any Python: no broad except, size limits (SRP), match capture.
  * api.py     -- any web API: {success, message, data} envelope, snake_case
    keys, versioned routes, strict declared input, V2 never imports V1.
  * web.py     -- stylesheets: no empty rulesets; plus the standards no diff can
    measure, reported as SKIP so "not checked" is never read as "fine".
  * packs/     -- per-framework rules (django, fastapi, ...), run only for a
    codebase that uses that framework.

Every check flags only what the change ADDED (engine.py). A codebase tunes them
under `prepush.standards` in its policy: `disable`, `limits`, `frameworks`, and
the per-rule settings each module documents.
"""
from __future__ import annotations

from tools.prepush.standards.api import (
    check_api_envelope,
    check_api_versioning,
    check_response_keys,
    check_strict_input,
    check_v2_imports_v1,
)
from tools.prepush.standards.general import (
    check_broad_except,
    check_match_capture,
    check_size_limits,
)
from tools.prepush.standards.packs import check_framework_packs
from tools.prepush.standards.web import check_css_empty_rules, check_standards_unmeasured

STANDARDS_CHECKS = [check_broad_except, check_size_limits, check_match_capture,
                    check_api_envelope, check_response_keys, check_api_versioning,
                    check_strict_input, check_v2_imports_v1, check_framework_packs,
                    check_css_empty_rules, check_standards_unmeasured]

__all__ = ["STANDARDS_CHECKS"]
