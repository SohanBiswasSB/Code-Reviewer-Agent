from registry import adapter


@adapter("source", "github")
class GitHubSource:
    def gather(self, target: str, **opts) -> dict:
        raise NotImplementedError("fetch PR/issue diff + check_runs via github_client")

# @adapter("source", "gitlab") -> same port, GitLab REST. Surface code is unchanged.
