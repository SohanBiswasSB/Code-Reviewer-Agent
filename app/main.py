"""FastAPI surface — surface-agnostic, plugin-driven (skeleton)."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Code Reviewer Agent (plug-and-play)")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/registered")
def registered():
    """List the plugins wired in this deployment (adapters + surfaces)."""
    import config
    return config.registered()


@app.post("/run/{surface}")
def run_surface(surface: str, body: dict):
    """Drive ANY registered surface to the approval gate.
    body: {"target": "<pr_ref | issue_url | YYYY-MM-DD>", "github_token": "..."}
    Returns {thread_id, proposed_action, review_reasoning}."""
    from fastapi import HTTPException

    import config
    tenant = body.get("tenant", "_default")
    pol = config.get_tenant_store().policy(tenant, surface)
    if surface not in pol.get("enabled_surfaces", []):
        raise HTTPException(status_code=403,
                            detail=f"surface '{surface}' not enabled for tenant '{tenant}'")
    # TODO: authenticate caller + authorize caller->tenant (G3); pin prompt/policy version (G8).
    raise NotImplementedError("build_graph(surface, tenant) -> run to interrupt -> gate payload")


@app.post("/approve")
def approve(body: dict):
    """Resume a paused run. body: {"thread_id": "...", "approved": true}."""
    raise NotImplementedError("resume with Command(resume=body['approved']) on thread_id")

# GET /auth/github/callback -> store approver OAuth token on the thread (Surface A writes).
