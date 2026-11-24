"""The `django` pack: models, serializers, transactions, signals, query tooling.

Each rule fires only on Django shapes (a `models.XField(...)` call, a `Meta`
class, `transaction.atomic`), so a codebase without Django is never flagged.

  * django_models   -- a string field with null=True (two empty values: '' and
    NULL) unless it is unique; a boolean with a default that is also nullable;
    null=True without blank=True. Optional, per codebase: UUID primary keys and
    a soft-delete field on every concrete model.
  * serializers     -- `fields = "__all__"` in a Meta: every column a model ever
    gains then ships without anyone deciding it should.
  * transactions    -- `transaction.atomic` inside `async def` (it is sync-only;
    wrap it with sync_to_async); a counter changed by read-modify-write and then
    saved (use F() or select_for_update); a task, mail or HTTP call fired inside
    an atomic block (fire it from transaction.on_commit, or it runs for a
    transaction that may still roll back).
  * signal_handlers -- a @receiver that names parameters it never reads. Keep the
    used ones and take the rest with **kwargs.
  * query_tooling   -- CaptureQueriesContext, debug toolbar, silk or an APM agent
    to count queries. The house measures with its activity logger.
  * api_views       -- an API view class defined outside views.py.
"""
from __future__ import annotations

import ast

from tools.prepush.standards.engine import (
    Concern,
    Finding,
    Pack,
    defs,
    dotted,
    keyword,
    own_nodes,
    scoped,
)

STRING_FIELDS = {"CharField", "TextField", "SlugField", "EmailField", "URLField",
                 "FilePathField"}
SIDE_EFFECT_CALLS = {"delay", "apply_async", "send_mail", "send_mass_mail"}
SIDE_EFFECT_ROOTS = {"requests", "httpx"}
QUERY_TOOLING = {"CaptureQueriesContext": "django.test.utils.CaptureQueriesContext",
                 "debug_toolbar": "django-debug-toolbar", "silk": "django-silk",
                 "ddtrace": "Datadog APM", "newrelic": "New Relic APM",
                 "elasticapm": "Elastic APM", "scout_apm": "Scout APM"}
API_VIEW_BASES = ("APIView", "ViewSet", "GenericAPIView")


# -- models ----------------------------------------------------------------------
def _field_calls(tree: ast.Module):
    """(Class.field, the field's Call node) for every `x = ...Field(...)` in a class."""
    for node, scope in scoped(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and scope and \
                dotted(node.value).endswith("Field") and isinstance(node.targets[0], ast.Name):
            yield f"{scope}.{node.targets[0].id}", node.value


def _null_problems(kind: str, call: ast.Call) -> list[str]:
    """What is wrong with one field's null/blank settings, in words."""
    if keyword(call, "null") is not True:
        return ["NullBooleanField is deprecated: BooleanField with a default"] \
            if kind == "NullBooleanField" else []
    problems = []
    if kind in STRING_FIELDS and keyword(call, "unique") is not True:
        problems.append(f"{kind} with null=True has two empty values ('' and NULL); "
                        "use blank=True, default='' (only a unique optional field may be null)")
    if kind == "BooleanField" and any(k.arg == "default" for k in call.keywords):
        problems.append("a BooleanField with a default is never nullable")
    if keyword(call, "blank") is not True:
        problems.append("null=True without blank=True: forms and serializers still "
                        "demand a value the database allows to be missing")
    return problems


def model_fields(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """The null/blank conventions, one finding per field and problem."""
    out = []
    for name, call in _field_calls(tree):
        kind = dotted(call).rsplit(".", 1)[-1]
        out += [Finding(f"{name}:{p[:24]}", call.lineno, f"{name}: {p}")
                for p in _null_problems(kind, call)]
    return out


def _is_concrete_model(cls: ast.ClassDef, cfg: dict) -> bool:
    """A class on models.Model directly (house base classes already carry the
    house fields), and not abstract."""
    bases = {dotted(b) for b in cls.bases}
    if "models.Model" not in bases or bases & set(cfg.get("model_bases") or []):
        return False
    meta = next((n for n in cls.body if isinstance(n, ast.ClassDef) and n.name == "Meta"), None)
    return not (meta and any(isinstance(s, ast.Assign) and dotted(s.targets[0]) == "abstract"
                             and getattr(s.value, "value", None) is True for s in meta.body))


def model_house_fields(tree: ast.Module, _path: str, cfg: dict) -> list[Finding]:
    """Per codebase, opt-in: UUID primary keys and a soft-delete field on every model."""
    soft = set(cfg.get("soft_delete_fields") or [])
    out = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if not _is_concrete_model(cls, cfg):
            continue
        fields = {n.targets[0].id: n.value for n in cls.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)}
        if cfg.get("require_uuid_pk") and not any(
                dotted(c).endswith("UUIDField") and keyword(c, "primary_key") is True
                for c in fields.values()):
            out.append(Finding(f"{cls.name}:uuid", cls.lineno,
                               f"model {cls.name} has no UUIDField(primary_key=True)"))
        if soft and not soft & set(fields):
            out.append(Finding(f"{cls.name}:soft", cls.lineno,
                               f"model {cls.name} has no soft-delete field "
                               f"({' or '.join(sorted(soft))}): rows are never hard-deleted"))
    return out


# -- serializers -----------------------------------------------------------------
def serializer_all(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """`fields = "__all__"` inside any Meta class."""
    return [Finding(f"{scope}:all", node.lineno,
                    f"{scope.rsplit('.', 1)[0]} exposes fields = '__all__': list the agreed "
                    "fields, so a new column never ships by accident")
            for node, scope in scoped(tree)
            if scope.endswith("Meta") and isinstance(node, ast.Assign)
            and dotted(node.targets[0]) == "fields"
            and getattr(node.value, "value", None) == "__all__"]


# -- transactions ----------------------------------------------------------------
def _atomic_withs(fn: ast.AST):
    """The `with transaction.atomic()` blocks a def opens itself."""
    return [n for n in own_nodes(fn) if isinstance(n, (ast.With, ast.AsyncWith)) and any(
        dotted(i.context_expr).endswith("atomic") for i in n.items)]


def atomic_in_async(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """transaction.atomic used inside an async def, as a block or a decorator."""
    out = []
    for node, name in defs(tree, (ast.AsyncFunctionDef,)):
        hits = [d.lineno for d in node.decorator_list if dotted(d).endswith("atomic")]
        hits += [w.lineno for w in _atomic_withs(node)]
        out += [Finding(f"{name}:atomic", ln, f"{name} is async but uses transaction.atomic, "
                        "which is sync-only: run the block via sync_to_async") for ln in hits]
    return out


def _side_effects(block: ast.AST):
    """Calls in an atomic block that act on the outside world; on_commit's own
    arguments are skipped, since deferring them is the fix."""
    stack = list(ast.iter_child_nodes(block))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and dotted(node).endswith("on_commit"):
            continue
        name = dotted(node) if isinstance(node, ast.Call) else ""
        if name and (name.rsplit(".", 1)[-1] in SIDE_EFFECT_CALLS
                     or name.split(".", 1)[0] in SIDE_EFFECT_ROOTS):
            yield node, name
        stack.extend(ast.iter_child_nodes(node))


def transaction_rules(tree: ast.Module, path: str, cfg: dict) -> list[Finding]:
    """Side effects inside atomic, and read-modify-write counters that get saved."""
    out = atomic_in_async(tree, path, cfg)
    for node, name in defs(tree):
        for block in _atomic_withs(node):
            out += [Finding(f"{name}:effect:{call}", c.lineno,
                            f"{name} calls {call} inside transaction.atomic: fire it from "
                            "transaction.on_commit, or it runs even if the write rolls back")
                    for c, call in _side_effects(block)]
        out += _counter_writes(node, name)
    return out


def _counter_writes(fn: ast.AST, name: str) -> list[Finding]:
    """`obj.n += 1` followed by `obj.save()` in the same def: a lost-update race."""
    nodes = list(own_nodes(fn))
    saved = {dotted(n.func.value) for n in nodes if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "save"}
    return [Finding(f"{name}:rmw:{dotted(n.target)}", n.lineno,
                    f"{name} increments {dotted(n.target)} in Python and saves it: two "
                    "requests lose an update. Use F() or select_for_update.")
            for n in nodes if isinstance(n, ast.AugAssign)
            and isinstance(n.target, ast.Attribute) and dotted(n.target.value) in saved]


# -- signals ---------------------------------------------------------------------
def signal_params(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """@receiver handlers that name parameters they never read."""
    out = []
    for node, name in defs(tree):
        if not any(dotted(d).endswith("receiver") for d in node.decorator_list):
            continue
        read = {n.id for n in own_nodes(node) if isinstance(n, ast.Name)}
        named = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        out += [Finding(f"{name}:{p}", node.lineno,
                        f"signal handler {name} names `{p}` but never reads it: "
                        "take it through **kwargs") for p in named if p not in read]
        if node.args.kwarg is None:
            out.append(Finding(f"{name}:kwargs", node.lineno,
                               f"signal handler {name} has no **kwargs: Django passes "
                               "more arguments than it names, and later versions add more"))
    return out


# -- query tooling ---------------------------------------------------------------
def query_tooling(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """Imports of the query-counting and APM tools the house does not use."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(a.name for a in node.names)]
        else:
            continue
        for full in names:
            hit = next((k for k in QUERY_TOOLING if k in full.split(".")), None)
            if hit:
                out.append(Finding(f"import:{hit}", node.lineno,
                                   f"imports {QUERY_TOOLING[hit]}: count queries with the "
                                   "activity logger (db_operations), not a new tool"))
    return out


# -- where API views live --------------------------------------------------------
def api_view_rules(tree: ast.Module, path: str, _cfg: dict) -> list[Finding]:
    """API view classes defined outside views.py.

    Pagination is deliberately NOT checked here. The house mixin
    (PaginatedQueryMixin) belongs on the endpoint's QUERY SERIALIZER, which a
    view names in another file, so a diff cannot tell whether it is there. A rule
    that assumed the mixin sits on the view would have flagged every correct
    paginated view in Masterclass. It is listed as unmeasured (web.py) instead.
    """
    if path.rsplit("/", 1)[-1] == "views.py":
        return []
    out = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        bases = [dotted(b).rsplit(".", 1)[-1] for b in cls.bases]
        if not cls.name.endswith("Mixin") and any(b.endswith(API_VIEW_BASES) for b in bases):
            out.append(Finding(f"{cls.name}:location", cls.lineno,
                               f"API view {cls.name} is outside views.py: every API "
                               "class lives in its app's views.py"))
    return out


PACK = Pack(
    id="django", title="Django",
    # Imported by the change, or present in the repo: either means Django code.
    markers=("django", "rest_framework"),
    concerns=(
        Concern("models", (model_fields, model_house_fields),
                "model field problems", "new model fields follow the null/blank rules"),
        Concern("serializers", (serializer_all,),
                "'__all__' serializers", "no serializer exposes '__all__'"),
        Concern("transactions", (transaction_rules,),
                "transaction problems", "new writes keep the transaction rules"),
        Concern("signals", (signal_params,),
                "signal handler problems", "new signal handlers take only what they use"),
        Concern("query_tooling", (query_tooling,),
                "query-tooling imports", "no new query-counting or APM tooling"),
        Concern("api_views", (api_view_rules,),
                "API view problems", "new API views live in views.py"),
    ),
)
