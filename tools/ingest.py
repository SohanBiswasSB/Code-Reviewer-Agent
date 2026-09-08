"""Build/refresh the vector index (idempotent). Skeleton.

Corpus: closed issues from ~4 public repos. Chunk 800/40. Embed
text-embedding-3-small truncated to 512 dims. delete_all=True before each run.
#4: prepend a one-line context per chunk before embedding (contextual retrieval).
#6: tag each chunk with source_type in {issue, pr, ci_log, doc}.
"""

def main() -> None:
    raise NotImplementedError("implement ingest: fetch -> chunk -> (contextualize) -> embed -> upsert")


if __name__ == "__main__":
    main()
