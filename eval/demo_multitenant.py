"""Each tenant is a CODEBASE (its own repo + prompts + rules). Escalate-to-human is
the SHARED safety net: BOTH escalate when a run isn't groundable; BOTH can approve
when it is. The per-codebase difference is prompts/risky_areas, not the safety route.
"""
from schemas.diagnosis import Diagnosis, Citation
from nodes.independent_review import independent_review
import config

store = config.get_tenant_store()

def run(tenant, cited_score):
    diag = Diagnosis("s", "medium", "c", "needs-triage", [Citation("iss-1", cited_score)])
    retr = [type("S", (), {"id": "iss-1", "score": cited_score})()]
    pol = store.policy(tenant, "code_review")
    out = independent_review({"diagnosis": diag, "_retrieved": retr, "_policy": pol})
    cb = pol["codebase"]
    return cb["name"], cb["vcs"], cb["repo"], out["review"].decision

print("GROUNDED diagnosis (cited score 0.50 >= 0.40):")
for t in ("project_a", "project_b"):
    print("  {:12} [{}:{}] -> {}".format(*run(t, 0.50)))

print("UNGROUNDED diagnosis (cited score 0.22 < 0.40):  <-- shared escalate-to-human")
for t in ("project_a", "project_b"):
    print("  {:12} [{}:{}] -> {}".format(*run(t, 0.22)))
