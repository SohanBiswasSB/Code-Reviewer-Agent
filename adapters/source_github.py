from registry import adapter
from tools import github_client


@adapter("source", "github")
class GitHubSource:
    """Gather evidence from GitHub. The ONLY module that talks to the API for reads.

    Thin on purpose: the HTTP details live in tools/github_client.py, and this
    adapter's job is to satisfy the EvidenceSource port so no node has to know
    which provider is in play. Swapping to GitLab is then a new adapter and an
    env change, with no node touched.
    """

    def gather(self, target: str, **opts) -> dict:
        token = opts.get("token")
        raw = github_client.fetch_issue(target, token)
        # Populate check runs for REAL rather than hardcoding [] -- classify's
        # `deterministic` branch is only reachable from a live CI result, so a
        # hardcoded empty list makes that whole route unreachable in production.
        raw["check_runs"] = github_client.get_check_runs(target, token)
        return raw

# @adapter("source", "gitlab") -> same port, GitLab REST. Nodes unchanged.
