"""The `fastapi` pack: what async web code gets wrong.

Strict typed input, the response envelope and /v1 paths are framework-neutral
and live in api.py. This pack adds the async-specific rules:

  * async_blocking -- a blocking call inside `async def`: `requests`, `time.sleep`,
    `urllib.request.urlopen`, `subprocess.run`, or a synchronous Django ORM query.
    One such call stalls every request the event loop is serving. Use an async
    client, `asyncio.sleep`, or move the work to a thread (run_in_threadpool,
    asyncio.to_thread).
  * auth_dependency -- a route that reads the Authorization header by hand.
    Authentication is a dependency (`Depends(...)`), so every route gets the same
    check and a route cannot quietly skip it.
"""
from __future__ import annotations

import ast

from tools.prepush.standards.engine import Concern, Finding, Pack, dotted, own_nodes, scoped

BLOCKING_ROOTS = {"requests"}
BLOCKING_CALLS = {"time.sleep", "urllib.request.urlopen", "subprocess.run",
                  "subprocess.check_output", "subprocess.call"}
SYNC_ORM = (".objects.",)


def _blocking(call: ast.Call) -> str | None:
    """Why this call blocks the event loop, or None."""
    name = dotted(call)
    if name.split(".", 1)[0] in BLOCKING_ROOTS or name in BLOCKING_CALLS:
        return name
    return name if any(s in f".{name}." for s in SYNC_ORM) else None


def async_blocking(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """Blocking calls made directly inside an async def (not in a nested sync def)."""
    out = []
    for node, scope in scoped(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        name = f"{scope}.{node.name}".lstrip(".")
        for call in (n for n in own_nodes(node) if isinstance(n, ast.Call)):
            what = _blocking(call)
            if what:
                out.append(Finding(f"{name}:{what}", call.lineno,
                                   f"async {name} calls {what}, which blocks the event loop: "
                                   "use an async client or asyncio.to_thread"))
    return out


ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "api_route"}


def _is_route(fn: ast.AST) -> bool:
    """A function registered as a route by a `@app.get(...)`-style decorator."""
    return any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
               and d.func.attr in ROUTE_METHODS for d in getattr(fn, "decorator_list", []))


def _reads_auth_header(node: ast.AST) -> bool:
    """`request.headers.get("Authorization")` or `request.headers["authorization"]`."""
    target = node.value if isinstance(node, ast.Subscript) else \
        node.func if isinstance(node, ast.Call) else None
    return target is not None and "headers" in dotted(target) and \
        "authorization" in ast.unparse(node).lower()


def auth_by_hand(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """Routes reading the Authorization header themselves instead of via Depends()."""
    out = []
    for node, scope in scoped(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _is_route(node):
            continue
        name = f"{scope}.{node.name}".lstrip(".")
        out += [Finding(f"{name}:authorization", n.lineno,
                        f"route {name} reads the Authorization header by hand: make "
                        "authentication a dependency, Depends(...)")
                for n in own_nodes(node) if _reads_auth_header(n)]
    return out


PACK = Pack(
    id="fastapi", title="FastAPI",
    markers=("fastapi", "starlette"),
    concerns=(
        Concern("async_blocking", (async_blocking,),
                "blocking calls in async code", "no new blocking call inside async def"),
        Concern("auth_dependency", (auth_by_hand,),
                "hand-rolled auth", "authentication goes through Depends()"),
    ),
)
