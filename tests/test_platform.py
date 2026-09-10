"""Platform + profile resolution.

These are UNIT tests, distinct in kind from eval/harness.py. The harness pins
safety-gate behaviour at a 100%-forever bar; these cover ordinary correctness of
the plumbing. Mixing the two would slowly turn the gate suite into a general test
suite and cost it its meaning.
"""

import pytest

import config
from adapters.platforms import base


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test picks its own platform/profile; never inherit the host's."""
    monkeypatch.delenv("CRA_PLATFORM", raising=False)
    monkeypatch.delenv("CRA_PROFILE", raising=False)


def test_every_platform_supports_the_default_profile():
    """`portable` must run everywhere, or cross-machine comparison is impossible.

    Iterates the REGISTRY rather than a hardcoded list. The list version silently
    stopped covering anything added after it was written, which is the opposite of
    what a test named "every platform" promises — `studio` was registered and
    unchecked until this changed.
    """
    registered = config.registry.list_registered()["adapters"]["platform"]
    assert len(registered) >= 4, f"expected every platform to register: {registered}"
    for name in registered:
        plat = config.registry.get_adapter("platform", name)()
        assert base.DEFAULT_PROFILE in plat.profiles, f"{name} cannot run the default profile"


def test_judge_is_a_different_family_in_every_profile():
    """The invariant independent_review depends on, checked for all profiles."""
    for name, models in base.PROFILES.items():
        assert models.families_are_independent(), (
            f"profile {name}: judge {models.judge} shares a family with "
            f"diagnosis {models.diagnosis}"
        )
        assert models.diagnosis != models.judge


def test_embedding_and_reranker_are_identical_across_profiles():
    """Changing these per profile would break index portability (see base.py)."""
    embeds = {m.embedding for m in base.PROFILES.values()}
    rerankers = {m.reranker for m in base.PROFILES.values()}
    assert embeds == {base.EMBEDDING_MODEL}
    assert rerankers == {base.RERANKER_MODEL}


@pytest.mark.parametrize(
    "platform,profile,expected",
    [
        # needed = hot weights (diagnosis + judge + embedder) + kv reserve
        ("mac", "portable", "resident"),        # 11.9 GB of an 18 GB budget
        ("mac", "mac_resident", "resident"),    # 15.9 GB
        ("mac", "mac_max", "jit_swap"),         # 21.3 GB -> cannot co-reside
        ("cuda", "portable", "jit_swap"),       # 9.9 GB against 5 GB of VRAM
        # the 64 GB Studio holds its own profile with room for a 32k KV cache...
        ("studio", "studio", "resident"),       # 39.2 GB of a 48 GB budget
        # ...and comfortably holds what the 24 GB laptop has to swap for
        ("studio", "mac_max", "resident"),      # 22.3 GB
    ],
)
def test_residency_is_computed_from_the_numbers(monkeypatch, platform, profile, expected):
    monkeypatch.setenv("CRA_PLATFORM", platform)
    rt = config.get_platform().runtime(profile)
    assert rt["residency"] == expected
    mem = rt["memory"]
    # the verdict must follow from the arithmetic it reports, not be asserted
    assert (mem["needed_gb"] <= mem["budget_gb"]) is (expected == "resident")
    assert mem["needed_gb"] == pytest.approx(mem["hot_weights_gb"] + mem["kv_reserve_gb"], abs=0.01)


@pytest.mark.parametrize(
    "platform,profile",
    [
        ("cuda", "mac_max"),      # 6 GB of VRAM against ~19 GB of weights
        ("mac", "studio"),        # the 24 GB laptop against a 33 GB profile
        ("cpu", "studio"),
    ],
)
def test_oversized_profile_raises_rather_than_downgrading(monkeypatch, platform, profile):
    """A silent downgrade would label eval results with a profile that never ran.

    `studio` matters here specifically: it names the models on the REMOTE host, so
    nothing stops someone setting it on the laptop out of habit. It has to fail
    loudly rather than quietly loading 33 GB of weights into 24 GB of memory.
    """
    monkeypatch.setenv("CRA_PLATFORM", platform)
    monkeypatch.setenv("CRA_PROFILE", profile)
    with pytest.raises(ValueError, match="not supported on this platform"):
        config.get_models()


def test_unknown_profile_raises(monkeypatch):
    monkeypatch.setenv("CRA_PLATFORM", "cuda")
    monkeypatch.setenv("CRA_PROFILE", "does_not_exist")
    with pytest.raises(KeyError, match="no model profile"):
        config.get_models()


def test_large_profiles_reduce_context_to_pay_for_weights(monkeypatch):
    """mac_max cannot afford a 32k KV cache on top of ~18 GB of weights."""
    monkeypatch.setenv("CRA_PLATFORM", "mac")
    plat = config.get_platform()
    assert plat.runtime("mac_max")["context"] < plat.runtime("portable")["context"]


def test_detect_platform_returns_a_registered_name():
    assert config.detect_platform() in ("cuda", "mac", "cpu")


def test_format_flag_matches_backend(monkeypatch):
    """MLX on Apple Silicon, GGUF elsewhere -- the reason quant lives in runtime()."""
    monkeypatch.setenv("CRA_PLATFORM", "mac")
    assert config.get_platform().runtime("portable")["lms_format_flag"] == "--mlx"
    monkeypatch.setenv("CRA_PLATFORM", "cuda")
    assert config.get_platform().runtime("portable")["lms_format_flag"] == "--gguf"
