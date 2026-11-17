"""The coding-standards checks (tools/prepush/standards/).

Every rule is tested both ways -- it flags the violation AND passes the correct
form -- and the engine is tested on a real git repository, because "flag only
what the change added" is the property that makes these fair to someone who
touches a file full of old debt.
"""
from __future__ import annotations

import ast
import subprocess
import textwrap

import pytest

from tools.prepush.change import Change
from tools.prepush.standards import api, general, web
from tools.prepush.standards.engine import Finding, added_findings
from tools.prepush.standards.packs import active_packs, check_framework_packs, django, fastapi

STAGED = {"diff_args": ["diff", "--cached"], "new_ref": ":", "base_ref": "HEAD:"}


def found(rule, src: str, path: str = "app/views.py", cfg: dict | None = None) -> list[str]:
    """The messages one rule produces for one source text."""
    return [f.message for f in rule(ast.parse(textwrap.dedent(src)), path, cfg or {})]


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    return root


def stage(root, files: dict[str, str], commit: str | None = None) -> Change:
    for path, text in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
        _git(root, "add", "--", path)
    if commit:
        _git(root, "commit", "-q", "-m", commit)
    names = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only"],
                           capture_output=True, text=True, check=True).stdout.split()
    return Change(root=root, files=names, revision=dict(STAGED))


def flags(results, check=None):
    return [r for r in results if r["status"] == "flag" and (check is None or r["check"] == check)]


# -- general: broad except -------------------------------------------------------
@pytest.mark.parametrize("handler", ["except Exception:", "except BaseException as e:",
                                     "except:", "except (ValueError, Exception):"])
def test_broad_handlers_are_found(handler):
    src = f"def f():\n    try:\n        g()\n    {handler}\n        pass\n"
    assert len(found(general.broad_excepts, src)) == 1


def test_specific_handlers_and_try_finally_are_fine():
    src = """
    import sys
    def f():
        try:
            g()
        except (KeyError, ValueError):
            pass
        try:
            g()
        finally:
            record(sys.exception())
    """
    assert found(general.broad_excepts, src) == []


# -- general: size limits --------------------------------------------------------
def test_measure_counts_like_pylint():
    fn = ast.parse(textwrap.dedent("""
    def f(self, a, b, *args, c, **kw):
        x = 1
        if a:
            return 1
        for i in b:
            pass
        else:
            pass
        try:
            pass
        except KeyError:
            pass
        finally:
            pass
        def inner():
            y = 2
            return y
        return x
    """)).body[0]
    m = general.measure(fn)
    assert m["args"] == 3                     # a, b, c (self, *args, **kw not counted)
    assert m["branches"] == 1 + 2 + 2         # if; for+else; one handler + finally
    assert m["returns"] == 2                  # inner's return is inner's
    assert "y" not in str(m) and m["locals"] == len({"self", "a", "b", "args", "c", "kw",
                                                      "x", "i", "inner"})


def test_a_function_pushed_over_a_limit_is_flagged_and_one_already_over_is_not(repo):
    big = "def f(a, b, c, d, e, g):\n    return a\n"
    stage(repo, {"m.py": big, "n.py": "def h(a):\n    return a\n"}, commit="base")
    change = stage(repo, {"m.py": big + "\n# touched\n",
                          "n.py": "def h(a, b, c, d, e, g):\n    return a\n"})
    hits = flags(general.check_size_limits(change, {}))
    assert [(h["file"], "6 args" in h["message"]) for h in hits] == [("n.py", True)]


def test_growing_an_oversized_function_is_flagged(repo):
    stage(repo, {"m.py": "def f(a, b, c, d, e, g):\n    return a\n"}, commit="base")
    change = stage(repo, {"m.py": "def f(a, b, c, d, e, g, h):\n    return a\n"})
    (hit,) = flags(general.check_size_limits(change, {}))
    assert "7 args" in hit["message"]


# -- general: match capture ------------------------------------------------------
def test_a_bare_constant_in_a_case_captures():
    src = """
    OK = 200
    def f(code):
        match code:
            case OK:
                return 1
    """
    assert "captures instead of comparing" in found(general.match_captures, src)[0]


def test_dotted_names_and_wildcards_compare_correctly():
    src = """
    def f(code):
        match code:
            case Status.OK:
                return 1
            case _:
                return 0
    """
    assert found(general.match_captures, src) == []


# -- engine: only what the change added ------------------------------------------
def test_old_debt_is_not_blamed_on_the_change(repo):
    old = "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    stage(repo, {"m.py": old}, commit="base")
    change = stage(repo, {"m.py": "import os\n\n" + old})        # moved down, not added
    assert flags(general.check_broad_except(change, {})) == []


def test_a_second_violation_in_the_same_place_is_new(repo):
    one = "    try:\n        g()\n    except Exception:\n        pass\n"
    stage(repo, {"m.py": "def f():\n" + one}, commit="base")
    change = stage(repo, {"m.py": "def f():\n" + one + one})
    assert len(flags(general.check_broad_except(change, {}))) == 1


def test_a_disabled_check_says_so(repo):
    change = stage(repo, {"m.py": "x = 1\n"})
    (r,) = general.check_broad_except(change, {"standards": {"disable": ["broad_except"]}})
    assert r["status"] == "skip" and "disabled" in r["message"]


def test_added_findings_counts_duplicates():
    a = Finding("k", 1, "m")
    assert added_findings([a, a, a], [a]) == [a, a]


# -- api -------------------------------------------------------------------------
def test_responses_outside_the_envelope_are_flagged():
    src = 'def v(r):\n    return Response({"ok": True})\n'
    assert "success, message, data" in found(api.envelope, src)[0]
    good = 'def v(r):\n    return Response({"success": True, "message": "", "data": {}})\n'
    assert found(api.envelope, good) == []


def test_a_fastapi_route_returning_a_bare_dict_is_flagged_and_webhooks_are_exempt():
    src = '@app.post("/v1/x")\nasync def x():\n    return {"status": "ok"}\n'
    assert found(api.envelope, src, "app/main.py")
    assert found(api.envelope, src, "app/hooks/stripe.py",
                 {"envelope_exempt": ["app/hooks/*"]}) == []


def test_vendor_webhooks_answer_in_the_vendors_format():
    """Webhook views share views.py with ordinary ones, so they are exempted by
    name, from both the envelope and the key-case rule."""
    src = """
    class ZoomWebhookV1(APIView):
        def post(self, request):
            return Response({"eventType": "ok"})
    class CourseListV1(APIView):
        def get(self, request):
            return Response({"eventType": "ok"})
    """
    assert len(found(api.envelope, src)) == 1
    assert len(found(api.response_keys, src)) == 1
    assert found(api.envelope, src, cfg={"envelope_exempt_scopes": ["CourseList*"]})


def test_response_keys_must_be_snake_case():
    src = ('def v(r):\n    return Response({"success": 1, "message": "", '
           '"data": {"userId": 1, "user_name": "x"}})\n')
    (msg,) = found(api.response_keys, src)
    assert "'userId'" in msg


def test_django_routes_carry_v1_in_route_name_and_class():
    src = ('urlpatterns = [path("courses/", CourseList.as_view(), name="course-list"),\n'
           '               path("v1/courses/", CourseListV1.as_view(), name="course-list-v1"),\n'
           '               path("api/", include("x.urls"))]\n')
    msgs = found(api.versioning, src, "app/urls.py")
    assert len(msgs) == 3 and all("courses/'" in m or "course-list'" in m or "CourseList " in m
                                  for m in msgs)


def test_fastapi_routes_need_v1_unless_the_router_prefix_has_it():
    src = '@router.get("/runs")\ndef runs():\n    pass\n@app.get("/healthz")\ndef h():\n    pass\n'
    assert len(found(api.versioning, src, "app/main.py")) == 1
    prefixed = 'router = APIRouter(prefix="/v1")\n' + src
    assert found(api.versioning, prefixed, "app/main.py") == []


def test_untyped_bodies_and_lax_models_are_flagged():
    src = """
    class RunIn(BaseModel):
        tenant: str
    class StrictIn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tenant: str
    @app.post("/v1/a")
    async def a(body: dict):
        pass
    @app.post("/v1/b")
    async def b(body: RunIn):
        pass
    @app.post("/v1/c")
    async def c(body: StrictIn):
        pass
    """
    msgs = found(api.strict_input, src, "app/main.py")
    assert len(msgs) == 2 and "untyped dict" in msgs[0] and "RunIn accepts" in msgs[1]


def test_django_views_reading_raw_input_are_flagged():
    src = 'def get(self, request):\n    q = request.query_params.get("q")\n'
    assert "request serializer" in found(api.strict_input, src, "app/views.py")[0]


def test_v2_never_imports_v1():
    src = "from courses.v1.services import price\nfrom courses.v2 import ok\n"
    assert len(found(api.v2_imports_v1, src, "courses/v2/views.py")) == 1
    assert found(api.v2_imports_v1, src, "courses/v1/views.py") == []


# -- packs -----------------------------------------------------------------------
def test_django_null_rules():
    src = """
    class Course(models.Model):
        title = models.CharField(max_length=9, null=True, blank=True)
        code = models.CharField(max_length=9, null=True, blank=True, unique=True)
        live = models.BooleanField(default=False, null=True, blank=True)
        note = models.TextField(blank=True, default="")
        ends = models.DateTimeField(null=True)
    """
    msgs = found(django.model_fields, src, "app/models.py")
    assert len(msgs) == 3
    assert "Course.title" in msgs[0] and "Course.live" in msgs[1] and "Course.ends" in msgs[2]


def test_django_house_fields_are_opt_in():
    src = "class C(models.Model):\n    name = models.CharField(max_length=9)\n"
    assert found(django.model_house_fields, src, "m.py") == []
    cfg = {"require_uuid_pk": True, "soft_delete_fields": ["is_deleted"]}
    assert len(found(django.model_house_fields, src, "m.py", cfg)) == 2
    based = "class C(SoftBase, models.Model):\n    pass\n"
    with_base = {**cfg, "model_bases": ["SoftBase"]}
    assert found(django.model_house_fields, based, "m.py", with_base) == []


def test_serializer_all_and_transactions():
    assert found(django.serializer_all,
                 'class S(Serializer):\n    class Meta:\n        fields = "__all__"\n')
    src = """
    async def a():
        with transaction.atomic():
            pass
    def b(order):
        with transaction.atomic():
            order.save()
            notify.delay(order.id)
            transaction.on_commit(lambda: notify.delay(order.id))
        order.views += 1
        order.save()
    """
    msgs = found(django.transaction_rules, src)
    assert len(msgs) == 3
    assert "sync-only" in msgs[0] and "notify.delay" in msgs[1] and "F()" in msgs[2]


def test_signal_handlers_take_only_what_they_use():
    src = """
    @receiver(post_save)
    def on_save(sender, instance, created, **kwargs):
        log(instance)
    @receiver(post_save)
    def ok(instance, **kwargs):
        log(instance)
    """
    msgs = found(django.signal_params, src)
    assert len(msgs) == 2 and "`sender`" in msgs[0] and "`created`" in msgs[1]


def test_query_tooling_and_view_location():
    assert found(django.query_tooling, "from django.test.utils import CaptureQueriesContext\n")
    src = "class CourseListV1(APIView):\n    pass\n"
    assert found(django.api_view_rules, src, "app/services.py")
    assert found(django.api_view_rules, src, "app/views.py") == []


def test_fastapi_blocking_calls_and_hand_rolled_auth():
    src = """
    async def a():
        requests.get("x")
        time.sleep(1)
        User.objects.get(pk=1)
        await asyncio.sleep(1)
        def sync_helper():
            requests.get("y")
    @app.get("/v1/me")
    def b(request):
        token = request.headers.get("Authorization")
    def not_a_route(request):
        token = request.headers["Authorization"]
    """
    assert len(found(fastapi.async_blocking, src)) == 3
    (msg,) = found(fastapi.auth_by_hand, src)
    assert "route b" in msg


def test_packs_run_only_for_codebases_that_use_them(repo):
    model = "class C(models.Model):\n    t = models.CharField(max_length=9, null=True)\n"
    plain = stage(repo, {"app/models.py": model})
    assert active_packs(plain, {}) == [] and check_framework_packs(plain, {}) == []
    dj = stage(repo, {"app/models.py": "from django.db import models\n" + model})
    assert [p.id for p in active_packs(dj, {})] == ["django"]
    hit = flags(check_framework_packs(dj, {}), "django:models")
    assert hit and "two empty values" in hit[0]["message"]
    declared = {"standards": {"frameworks": ["fastapi"]}}
    assert [p.id for p in active_packs(dj, declared)] == ["fastapi"]


# -- web -------------------------------------------------------------------------
def test_empty_css_rules_in_stylesheets_and_style_blocks():
    assert web.empty_rules("a.css", ".a{}\n.b{color:red}\n@media x{.c { }}\n") == [
        (".a", 1), (".c", 3)]
    page = "<script>const o={}</script><style>\n.x{}\n</style>"
    assert web.empty_rules("p.html", page) == [(".x", 2)]


def test_unmeasurable_standards_are_reported_as_not_checked(repo):
    out = web.check_standards_unmeasured(stage(repo, {"m.py": "x = 1\n"}), {})
    assert {r["status"] for r in out} == {"skip"} and len(out) == len(web.UNMEASURED)
