"""Three false positives the Merger review produced on 2026-11-24, each pinned.

  * postman_hand_edit on an honest regeneration (sample files changed, or only the
    "GENERATED ... on <date>" stamp changed, or an API file changed);
  * postman_structure ignoring `no_failure_example_ok`, because the collection
    names a request "UNIVERSAL GET faqs-v1" and the policy says "faqs-v1";
  * api_envelope in a project whose renderer adds the envelope itself.
"""
from __future__ import annotations

import ast
import json
import subprocess

import pytest

from tools import change_checks as cc
from tools.prepush.standards import api

COLLECTION = "pm/all.postman_collection.json"
PM = {"postman": {"collection": COLLECTION, "when": ["views*.py", "urls.py"],
                  "generator_inputs": ["scripts/gen.py", "docs/API Samples/"],
                  "generated_stamp": "GENERATED from the live URLconf by",
                  "no_failure_example_ok": ["faqs-v1"]}}
STAGED = {"diff_args": ["diff", "--cached"], "new_ref": ":", "base_ref": "HEAD:"}


def git(root, *args):
    """Run git in the scratch repo."""
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def collection(stamp_date="2026-11-17", body="a"):
    """A small generated collection; the stamp line carries the date."""
    return json.dumps({"info": {"description":
                                f"GENERATED from the live URLconf by gen.py on {stamp_date}."},
                       "item": [{"name": "UNIVERSAL GET faqs-v1",
                                 "request": {"description": "d"},
                                 "response": [{"code": 200, "body": body}]}]}, indent=1)


@pytest.fixture(name="repo")
def fixture_repo(tmp_path):
    """A repo whose base commit already holds the collection."""
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "Sohan Biswas")
    (root / "pm").mkdir()
    (root / COLLECTION).write_text(collection())
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "base")
    return root


def stage(root, files: dict[str, str]) -> list[dict]:
    """Stage `files` and run the Postman check on what is staged."""
    for path, text in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        git(root, "add", "--", path)
    names = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    return cc.check_postman(cc.Change(root=root, files=names, revision=dict(STAGED)), PM)


def checks(results, name):
    """The flags one check raised."""
    return [r for r in results if r["status"] == "flag" and r["check"] == name]


def test_a_stamp_only_regeneration_is_not_a_hand_edit(repo):
    """079f52f: the only collection change was the generation date."""
    assert not checks(stage(repo, {COLLECTION: collection("2026-11-18")}), "postman_hand_edit")


def test_a_sample_file_change_explains_the_collection_change(repo):
    """d454df7: example bodies changed because their API Samples file did."""
    res = stage(repo, {COLLECTION: collection(body="b"), "docs/API Samples/x.200.json": "{}"})
    assert not checks(res, "postman_hand_edit")


def test_an_api_change_explains_the_collection_change(repo):
    """Views and URLs feed the generator too."""
    res = stage(repo, {COLLECTION: collection(body="b"), "Shop/views.py": "x = 1\n"})
    assert not checks(res, "postman_hand_edit")


def test_a_real_hand_edit_is_still_flagged(repo):
    """A body changed with nothing that feeds the generator changed."""
    assert checks(stage(repo, {COLLECTION: collection(body="b")}), "postman_hand_edit")


def test_the_failure_example_exemption_matches_the_route_name(repo):
    """'faqs-v1' in the policy exempts 'UNIVERSAL GET faqs-v1' in the collection."""
    res = stage(repo, {COLLECTION: collection("2026-11-18")})
    assert not [r for r in checks(res, "postman_structure") if "failure example" in r["message"]]


def test_a_declared_envelope_renderer_turns_the_envelope_rule_off():
    """The Merger repo's EnvelopeJSONRenderer wraps every response itself."""
    tree = ast.parse('def v(r):\n    return Response({"state_boards": []})\n')
    assert api.envelope(tree, "Accounts/views.py", {})
    cfg = {"envelope_renderer": "utils.renderers.EnvelopeJSONRenderer"}
    assert not api.envelope(tree, "Accounts/views.py", cfg)
