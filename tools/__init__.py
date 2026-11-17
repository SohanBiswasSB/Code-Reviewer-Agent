"""Building blocks behind the adapters, the CLIs and the pre-push checks.

Indexing (`ingest`, `chunking`, `chunk_text`, `bm25`, `fusion`, `filekinds`,
`codebase`), measurement (`calibrate`), the pre-push change checks
(`change_checks`, `prepush/`), the offline retrieval demo (`retrieval`) and the
stubbed GitHub client (`github_client`). A node must not import `tools.retrieval`
or `tools.github_client` directly -- that is the T5/G2 bug class; it reaches them
through the Retriever / EvidenceSource / ActionSink ports instead.
"""
