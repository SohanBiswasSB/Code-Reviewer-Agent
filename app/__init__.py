"""The service: graph and state, the API and CLI entry points, and what observes a run.

`graph` and `state` define the run; `main` (HTTP) and `review` (terminal) drive it;
`deps` owns the checkpointer; `activity`, `live` and `monitor` record and show it;
`doctor` checks the machine before any of it is tried.
"""
