"""The graph's nodes: one module per step, wired together in `app/graph.py`.

Each node takes the run's state dict and returns only the keys it writes. The
safety design lives here -- the four-boolean gate (`independent_review`), the
in-node approval guard (`execute`), the sole writer of `approved`
(`await_approval`) and the run record (`audit`) -- so the invariants in CLAUDE.md
are enforced by these modules, not by routing alone. Underscore modules are
shared helpers, not nodes. Nodes reach providers only through `config` and the
ports, never by importing a concrete adapter (T5/G2).
"""
