from registry import adapter

@adapter("source", "s3logs")
class S3LogSource:
    """Read the day's CloudWatch->S3 exports for milesone-backend + celery-worker."""
    def gather(self, target: str, **opts) -> dict:
        # target = 'YYYY-MM-DD'. Return {"backend": [...lines], "celery": [...lines]}.
        raise NotImplementedError("read s3://<log-export-bucket>/daily/<date>/... and parse")
