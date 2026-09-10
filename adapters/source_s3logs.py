"""Daily log export -> a BOUNDED window (Layer 0 + G4).

## What is real here and what is not

The **pipeline is real**: stream -> detect -> aggregate -> cap -> digest, and it
is what keeps a day of logs out of the checkpoint. Only the S3 read itself is
stubbed, because it needs a bucket and credentials that no test can have. Pass
`reader=` to drive the real pipeline over any line source; the default reader
raises, loudly, rather than pretending to return data.

## Why the reduction happens HERE and not in `normalize_evidence`

The lines have to be reduced while they are still a stream. `gather()` returning
2M lines for a later node to summarise would already have put them in memory and,
the moment they landed in `_raw`, in every checkpoint of the run. There is no
later point at which the reduction can still be cheap.

So this adapter never materialises the day: it reads in windows, runs the
detectors over each, keeps running counts and a capped sample, and returns
kilobytes. `normalize_evidence` then validates that into a `LogWindow`.
"""
from __future__ import annotations

import hashlib

from registry import adapter
from schemas.evidence import MAX_LINE_CHARS

WINDOW_LINES = 5_000          # lines per detector call; bounds peak memory
MAX_SAMPLES_PER_SIGNAL = 5    # what actually reaches the prompt, per signal
MAX_CANDIDATES = 20           # distinct signals carried forward


@adapter("source", "s3logs")
class S3LogSource:
    """Read the day's CloudWatch->S3 exports and reduce them to a LogWindow.

    `target` is 'YYYY-MM-DD'. Returns the dict shape LogWindow validates.
    """

    def __init__(self, detector=None):
        self._detector = detector

    def detector(self):
        if self._detector is None:
            import config
            self._detector = config.get_detector()
        return self._detector

    def _read(self, target: str, **opts):
        """Yield (service, ref, line). The ONLY part that is stubbed.

        Real implementation: list s3://<bucket>/daily/<date>/ per service and
        stream each object's lines. Deliberately a generator so the caller never
        holds the day in memory.
        """
        raise NotImplementedError(
            f"read s3://<log-export-bucket>/daily/{target}/ per service and yield "
            f"(service, key, line). Inject `reader=` to run the pipeline offline."
        )

    def gather(self, target: str, **opts) -> dict:
        reader = opts.get("reader") or (lambda: self._read(target, **opts))
        detector = opts.get("detector") or self.detector()

        digest = hashlib.sha256()
        line_count = 0
        services: dict[str, None] = {}
        counts: dict[str, int] = {}
        severity_of: dict[str, str] = {}
        samples: dict[str, list[dict]] = {}
        truncated = False

        # Buffered per (service, ref) so each detector call sees a contiguous
        # window with correct line offsets.
        buffers: dict[tuple[str, str], list[str]] = {}
        starts: dict[tuple[str, str], int] = {}
        per_stream_count: dict[tuple[str, str], int] = {}

        def flush(key: tuple[str, str]) -> None:
            nonlocal truncated
            lines = buffers.get(key)
            if not lines:
                return
            service, ref = key
            hits = detector.detect({"service": service, "ref": ref,
                                    "start": starts[key], "lines": lines})
            for hit in hits:
                signal = hit["signal"]
                counts[signal] = counts.get(signal, 0) + 1
                severity_of.setdefault(signal, hit.get("severity", "medium"))
                bucket = samples.setdefault(signal, [])
                if len(bucket) < MAX_SAMPLES_PER_SIGNAL:
                    bucket.append({
                        "service": hit.get("service", ""),
                        "line_no": hit.get("line_no", 0),
                        "ref": hit.get("ref", ""),
                        # Truncated before it can enter state. One pathological
                        # line (a serialised payload, a base64 blob) would
                        # otherwise be carried through every checkpoint.
                        "text": (hit.get("text", "") or "")[:MAX_LINE_CHARS],
                    })
                else:
                    truncated = True
            buffers[key] = []
            starts[key] = per_stream_count[key]

        for service, ref, line in reader():
            line_count += 1
            services.setdefault(service, None)
            digest.update(line.encode("utf-8", "replace"))
            key = (service, ref)
            starts.setdefault(key, 0)
            per_stream_count[key] = per_stream_count.get(key, 0) + 1
            buffers.setdefault(key, []).append(line)
            if len(buffers[key]) >= WINDOW_LINES:
                flush(key)
        for key in list(buffers):
            flush(key)

        # Most frequent signals first: a cap has to drop the least informative.
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if len(ordered) > MAX_CANDIDATES:
            truncated = True
        candidates = [
            {"signal": signal,
             "severity": severity_of.get(signal, "medium"),
             "count": count,
             "services": sorted({s["service"] for s in samples.get(signal, [])}),
             "samples": samples.get(signal, [])}
            for signal, count in ordered[:MAX_CANDIDATES]
        ]

        return {
            "date": target,
            "ref": opts.get("ref", f"s3://<log-export-bucket>/daily/{target}/"),
            "digest": f"sha256:{digest.hexdigest()}",
            "line_count": line_count,
            "services": sorted(services),
            "counts": dict(ordered),
            "candidates": candidates,
            "truncated": truncated,
        }
