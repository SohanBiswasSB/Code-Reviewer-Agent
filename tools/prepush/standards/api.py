"""API standards, for Django REST and FastAPI alike.

  * api_envelope   -- every response body is {success, message, data}. Checked
    where the body is a dict literal: `Response({...})`, `JsonResponse({...})`,
    `JSONResponse(content={...})`, or a dict a FastAPI route returns. Vendor
    webhooks are excused: by path (`standards.envelope_exempt`) or, since they
    usually share views.py with ordinary views, by name
    (`standards.envelope_exempt_scopes`, default *Webhook*).
  * response_keys  -- response keys are snake_case. A legacy key is renamed, not
    aliased.
  * api_versioning -- the route, its name and its view class carry v1/V1 (Django
    urls.py); a FastAPI route path, or its router's prefix, carries /v1. Health
    probes listed in `standards.unversioned_ok` are excused.
  * strict_input   -- input is declared, and anything undeclared is a 400: no
    FastAPI body typed `dict`/`Any`, no `await request.json()`, a request model
    with extra="forbid", and no `request.data.get(...)` in a Django view.
  * v2_imports_v1  -- V2 owns everything and V1 is frozen, so V2 code never
    imports a V1 module.
"""
from __future__ import annotations

import ast
import fnmatch
import re

from tools.prepush.change import Change
from tools.prepush.standards.engine import Finding, dotted, keyword, run_rules, scoped

ENVELOPE = {"success", "message", "data"}
RESPONSE_CALLS = {"Response", "JsonResponse", "JSONResponse"}
ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "api_route"}
RE_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
RE_VERSION = re.compile(r"(^|[/_\-.])v\d+", re.I)
RE_V2_PATH = re.compile(r"(^|[/_])v2([/_.]|$)", re.I)
RE_V1_MODULE = re.compile(r"(^|[._])v1([._]|$)", re.I)
UNVERSIONED_OK = ["/", "/health", "/healthz", "/ready", "/readyz", "/metrics"]
RAW_DJANGO_INPUT = ("request.data", "request.POST", "request.GET", "request.query_params")


def _route_path(decorator: ast.AST) -> str | None:
    """The path of a `@app.get("/x")`-style route decorator, or None if it is not one."""
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in ROUTE_METHODS and decorator.args):
        return None
    first = decorator.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) \
        else None


def _routes(tree: ast.Module):
    """(function, decorator call, its path) for every FastAPI-style route."""
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,
                                                             ast.AsyncFunctionDef))):
        for d in fn.decorator_list:
            path = _route_path(d)
            if path is not None:
                yield fn, d, path


def _exempt(scope: str, cfg: dict) -> bool:
    """True inside a vendor webhook: it answers in the vendor's format, not ours.

    Matched by NAME (`standards.envelope_exempt_scopes`, default *Webhook*),
    because webhook views share views.py with ordinary ones, so a path cannot
    single them out.
    """
    patterns = cfg.get("envelope_exempt_scopes") or ["*Webhook*", "*webhook*"]
    return any(fnmatch.fnmatch(part, p) for part in scope.split(".") for p in patterns)


def _response_bodies(tree: ast.Module, cfg: dict):
    """(dict literal, where) for each response body this module builds as a literal,
    outside vendor webhooks."""
    for node, scope in scoped(tree):
        if isinstance(node, ast.Call) and dotted(node).rsplit(".", 1)[-1] in RESPONSE_CALLS \
                and not _exempt(scope, cfg):
            body = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg in ("content", "data")), None)
            if isinstance(body, ast.Dict):
                yield body, dotted(node)
    for fn, _, path in _routes(tree):
        if _exempt(fn.name, cfg):
            continue
        for ret in (n for n in ast.walk(fn) if isinstance(n, ast.Return)):
            if isinstance(ret.value, ast.Dict):
                yield ret.value, f"route {path}"


def _const_keys(d: ast.Dict) -> list[str]:
    """The string-constant keys of a dict literal (computed keys are skipped)."""
    return [k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def envelope(tree: ast.Module, path: str, cfg: dict) -> list[Finding]:
    """Response bodies that are not exactly the {success, message, data} envelope."""
    if any(fnmatch.fnmatch(path, p) for p in cfg.get("envelope_exempt") or []):
        return []
    out = []
    for body, where in _response_bodies(tree, cfg):
        keys = set(_const_keys(body))
        if keys != ENVELOPE and len(keys) == len(body.keys):
            got = ", ".join(sorted(keys)) or "nothing"
            out.append(Finding(f"{where}:{sorted(keys)}", body.lineno,
                               f"{where} returns {{{got}}}: every response is exactly "
                               "{success, message, data}"))
    return out


def response_keys(tree: ast.Module, _path: str, cfg: dict) -> list[Finding]:
    """Non-snake_case keys anywhere inside a response body literal."""
    out = []
    for body, where in _response_bodies(tree, cfg):
        for d in (n for n in ast.walk(body) if isinstance(n, ast.Dict)):
            out += [Finding(f"{where}:{k}", d.lineno, f"{where} sends key '{k}': response "
                            "keys are snake_case (rename a legacy key, never alias it)")
                    for k in _const_keys(d) if not RE_SNAKE.match(k)]
    return out


def _module_prefix_versioned(tree: ast.Module) -> bool:
    """True when the module's router or include_router prefix carries /vN."""
    return any(isinstance(n, ast.Call) and RE_VERSION.search(str(keyword(n, "prefix") or ""))
               for n in ast.walk(tree))


def _django_route_gaps(call: ast.Call) -> list[str]:
    """What one `path(...)` entry is missing: version in route, name or view class."""
    route = call.args[0].value
    view = call.args[1] if len(call.args) > 1 else None
    if not route or dotted(view).endswith("include"):
        return []
    gaps = []
    if not RE_VERSION.search(route):
        gaps.append(f"route '{route}' has no v1")
    name = keyword(call, "name")
    if name is not None and not RE_VERSION.search(str(name)):
        gaps.append(f"name '{name}' has no -v1")
    view_cls = dotted(view).removesuffix(".as_view")
    if view_cls and not re.search(r"V\d+", view_cls.rsplit(".", 1)[-1]):
        gaps.append(f"view {view_cls} has no V1")
    return gaps


def versioning(tree: ast.Module, path: str, cfg: dict) -> list[Finding]:
    """Unversioned Django routes (urls.py) and FastAPI route paths."""
    out = []
    if path.rsplit("/", 1)[-1] == "urls.py":
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and dotted(n) in ("path", "re_path") and n.args
                     and isinstance(n.args[0], ast.Constant)):
            out += [Finding(f"url:{call.args[0].value}:{g.split(' ', 1)[0]}", call.lineno,
                            f"{g}: the path, route name and view class all carry v1/V1")
                    for g in _django_route_gaps(call)]
    if _module_prefix_versioned(tree):
        return out
    ok = set(cfg.get("unversioned_ok") or UNVERSIONED_OK)
    out += [Finding(f"route:{route}", d.lineno,
                    f"route {route} is unversioned: serve it under /v1")
            for _, d, route in _routes(tree)
            if route not in ok and not RE_VERSION.search(route)]
    return out


def _untyped_params(fn: ast.AST) -> list[tuple[str, int]]:
    """FastAPI parameters typed dict/Any: an undeclared body that accepts anything."""
    bad = {"dict", "Dict", "Any", "typing.Any", "typing.Dict"}
    return [(a.arg, a.lineno) for a in (*fn.args.args, *fn.args.kwonlyargs)
            if a.annotation is not None
            and dotted(a.annotation.value if isinstance(a.annotation, ast.Subscript)
                       else a.annotation) in bad]


def _lax_models(tree: ast.Module, used: set[str]) -> list[ast.ClassDef]:
    """Request models defined here and used by a route, without extra='forbid'."""
    lax = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name in used):
        src = ast.unparse(cls)
        if "BaseModel" in {dotted(b).rsplit(".", 1)[-1] for b in cls.bases} and \
                not re.search(r"extra\s*=\s*['\"]forbid['\"]|Extra\.forbid", src):
            lax.append(cls)
    return lax


def strict_input(tree: ast.Module, path: str, _cfg: dict) -> list[Finding]:
    """Input that is not declared, or declared without rejecting the undeclared."""
    out = []
    annotations: set[str] = set()
    for fn, _, route in _routes(tree):
        out += [Finding(f"{route}:{name}", line, f"route {route} takes `{name}` as an "
                        "untyped dict: declare a Pydantic model with extra='forbid'")
                for name, line in _untyped_params(fn)]
        annotations |= {dotted(a.annotation) for a in fn.args.args if a.annotation}
        if any(dotted(n).endswith("request.json") for n in ast.walk(fn)):
            out.append(Finding(f"{route}:json", fn.lineno, f"route {route} reads "
                               "request.json() by hand: declare the body as a model"))
    out += [Finding(f"model:{c.name}", c.lineno, f"request model {c.name} accepts undeclared "
                    "fields: set model_config = ConfigDict(extra='forbid')")
            for c in _lax_models(tree, annotations)]
    if path.rsplit("/", 1)[-1] == "views.py":
        out += [Finding(f"raw:{dotted(n)}", n.lineno, f"{dotted(n)}(...) reads input by hand: "
                        "validate through a request serializer that rejects undeclared params")
                for n in ast.walk(tree) if isinstance(n, ast.Call)
                and dotted(n).endswith(".get") and dotted(n)[:-4] in RAW_DJANGO_INPUT]
    return out


def v2_imports_v1(tree: ast.Module, path: str, _cfg: dict) -> list[Finding]:
    """A V2 module importing a V1 one."""
    if not RE_V2_PATH.search(path):
        return []
    out = []
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        out += [Finding(f"import:{m}", node.lineno, f"V2 code imports {m}: V2 owns "
                        "everything and never imports or edits V1")
                for m in mods if RE_V1_MODULE.search(m)]
    return out


def check_api_envelope(change: Change, config: dict) -> list[dict]:
    """No NEW response body outside the {success, message, data} envelope."""
    return run_rules(change, config, ("api_envelope", 1, (envelope,)),
                     ("unenveloped responses", "new responses use {success, message, data}"))


def check_response_keys(change: Change, config: dict) -> list[dict]:
    """No NEW non-snake_case response key."""
    return run_rules(change, config, ("response_keys", 1, (response_keys,)),
                     ("non-snake_case keys", "new response keys are snake_case"))


def check_api_versioning(change: Change, config: dict) -> list[dict]:
    """No NEW unversioned route."""
    return run_rules(change, config, ("api_versioning", 1, (versioning,)),
                     ("unversioned routes", "new routes carry v1/V1"))


def check_strict_input(change: Change, config: dict) -> list[dict]:
    """No NEW undeclared or loosely declared input."""
    return run_rules(change, config, ("strict_input", 1, (strict_input,)),
                     ("loose inputs", "new inputs are declared and strict"))


def check_v2_imports_v1(change: Change, config: dict) -> list[dict]:
    """No NEW V2-to-V1 import."""
    return run_rules(change, config, ("v2_imports_v1", 1, (v2_imports_v1,)),
                     ("V2-to-V1 imports", "V2 code imports nothing from V1"))


API_RULES = (envelope, response_keys, versioning, strict_input, v2_imports_v1)
