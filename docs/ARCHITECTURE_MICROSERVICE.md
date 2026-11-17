# Plug-and-play microservice architecture

The agent is a **standalone microservice** (FastAPI) built with **ports & adapters
(hexagonal)**. The core graph depends only on *interfaces* (`ports/`); everything that
varies is a *plugin* resolved from env at startup. Nothing swaps require touching core.

```
                 ┌──────────────────────────── AGENT SERVICE ───────────────────────────┐
   HTTP / RunTask│  app/main.py (FastAPI)   jobs/ (scheduled runners)                    │
        ─────────▶                                                                       │
                 │   core graph (LangGraph) ── nodes/ ── depends ONLY on ports/          │
                 │        classify · diagnose · judge(gate) · await_approval · execute   │
                 │                              │ (ports)                                │
                 │   ┌──────────────┬───────────┴───────────┬──────────────┬──────────┐  │
                 │   ▼              ▼                       ▼              ▼          ▼  │
                 │  LLM          Retriever            EvidenceSource    ActionSink  Surface│
                 │  openai       hybrid/pgvector      github/gitlab     pr_comment  code_review│
                 │  claude…      pinecone…            s3logs            report      log_review │
                 └──────────────────────────────────────────────────────────────────────┘
```

## The ports (contracts) — `ports/interfaces.py`
`LLM`, `Retriever`, `EvidenceSource`, `ActionSink`, `Detector`, `Checkpointer`, `Surface`.
Core code imports these Protocols, never a concrete provider.

## The plugs — `registry.py`
Adapters/surfaces self-register with `@adapter(kind, name)` / `@surface(name)`.
`config.py` resolves them from env — the ONLY place a provider is chosen:

```
LLM_PROVIDER=openai      # -> adapters/llm_openai.py
RETRIEVER=indexed        # -> adapters/retriever_indexed.py (offline demo: hybrid)
SURFACES=code_review,log_review
CHECKPOINTER=postgres
```

## Add a new capability (the plug-and-play move)
- **New LLM/vector/SCM provider:** add `adapters/<x>.py` with `@adapter("llm","claude")`,
  import it in `adapters/__init__.py`, set the env var. No core change.
- **New surface** (e.g. `pr_risk_score`, `incident_postmortem`): add `surfaces/<x>.py` with
  `@surface("...")`, bind a source + prompts + sink, add it to `SURFACES`. No core change.
- **New log detector:** add a `Detector` and register it; the log surface picks it up.

## Run modes (one image)
- **web** — `uvicorn app.main:app` : `POST /v1/run/{surface}` drives any registered surface
  to the approval gate; `POST /v1/approve` resumes; `GET /v1/registered` lists plugins.
  Typed request/response models, `{success, message, data}` envelope (app/api_models.py).
- **daily** — `python -m jobs.daily_log_review` on **EventBridge → ECS RunTask** (the exact
  MilesOne pattern): runs the `log_review` surface over yesterday's logs, emits the report.
- **eval** — `python -m eval.harness`.

## Why this shape
- **Isolation:** the agent is its own service — it never imports the MilesOne codebase at
  runtime; it *reads* the repo (as evidence) and *reads* the S3 logs. Least-privilege.
- **Swappable:** GitHub↔GitLab, OpenAI↔Claude↔local, Pinecone↔pgvector — all config, not code.
- **Composable:** each surface is an independent plugin; enable/disable per deployment.
