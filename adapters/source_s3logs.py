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
from dataclasses import dataclass, field

import registry
from registry import adapter
from schemas.evidence import MAX_LINE_CHARS

WINDOW_LINES = 5_000          # lines per detector call, per stream
# Lines held across ALL streams before everything is flushed. WINDOW_LINES alone
# bounded one stream, not the day: a stream only flushed at 5,000 lines, so every
# shorter S3 object stayed buffered until EOF. Measured: 400 objects of 4,999
# lines peaked at 252 MB -- the whole day, which is what this module exists to
# never hold. Flushing early is safe: `starts` keeps each stream's offsets right.
MAX_BUFFERED_LINES = 4 * WINDOW_LINES
MAX_SAMPLES_PER_SIGNAL = 5    # what actually reaches the prompt, per signal
MAX_CANDIDATES = 20           # distinct signals carried forward


def _candidates(counts: dict, severity_of: dict, samples: dict) -> tuple[list, list[dict]]:
    """(all signals by count, the capped candidate list). Most frequent first: a
    cap has to drop the least informative signals, not arbitrary ones."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ordered, [
        {"signal": signal,
         "severity": severity_of.get(signal, "medium"),
         "count": count,
         "services": sorted({s["service"] for s in samples.get(signal, [])}),
         "samples": samples.get(signal, [])}
        for signal, count in ordered[:MAX_CANDIDATES]
    ]


@dataclass
class _SignalTally:
    """What the detector found so far: counts, first severity and capped samples per signal."""

    counts: dict[str, int] = field(default_factory=dict)
    severity_of: dict[str, str] = field(default_factory=dict)
    samples: dict[str, list[dict]] = field(default_factory=dict)
    truncated: bool = False

    def fold(self, hit: dict) -> None:
        """Count one detector hit and keep it as a sample while its signal has room."""
        signal = hit["signal"]
        self.counts[signal] = self.counts.get(signal, 0) + 1
        self.severity_of.setdefault(signal, hit.get("severity", "medium"))
        bucket = self.samples.setdefault(signal, [])
        if len(bucket) >= MAX_SAMPLES_PER_SIGNAL:
            self.truncated = True
            return
        bucket.append({
            "service": hit.get("service", ""),
            "line_no": hit.get("line_no", 0),
            "ref": hit.get("ref", ""),
            # Truncated before it can enter state. One pathological line (a
            # serialised payload, a base64 blob) would otherwise be carried
            # through every checkpoint.
            "text": (hit.get("text", "") or "")[:MAX_LINE_CHARS],
        })


@dataclass
class _StreamBuffers:
    """Lines waiting for the detector, per (service, ref) stream.

    `starts` is each stream's line offset for its next window and `seen` its
    running line count, so a window flushed early still numbers its lines right.
    `buffered` is the total across every stream -- the number MAX_BUFFERED_LINES
    bounds.
    """

    lines: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    starts: dict[tuple[str, str], int] = field(default_factory=dict)
    seen: dict[tuple[str, str], int] = field(default_factory=dict)
    buffered: int = 0


class _WindowReducer:
    """The running state of one day's reduction: the stream buffers, and the
    tally of what the detector found so far.

    Buffered per (service, ref) so each detector call sees a contiguous window
    with correct line offsets. Memory is bounded by MAX_BUFFERED_LINES across all
    streams, never by the size of the day.
    """

    def __init__(self, detector):
        """Start empty; `detector` is called once per flushed window."""
        self.detector = detector
        self.digest = hashlib.sha256()
        self.line_count = 0
        self.services: dict[str, None] = {}
        self.tally = _SignalTally()
        self.streams = _StreamBuffers()

    def add(self, service: str, ref: str, line: str) -> None:
        """Take one line: hash it, buffer it, and flush when a cap is reached.

        A stream flushes alone at WINDOW_LINES; when ALL streams together reach
        MAX_BUFFERED_LINES, every stream flushes (see MAX_BUFFERED_LINES).
        """
        self.line_count += 1
        self.services.setdefault(service, None)
        self.digest.update(line.encode("utf-8", "replace"))
        key, s = (service, ref), self.streams
        s.starts.setdefault(key, 0)
        s.seen[key] = s.seen.get(key, 0) + 1
        s.lines.setdefault(key, []).append(line)
        s.buffered += 1
        if len(s.lines[key]) >= WINDOW_LINES:
            self.flush(key)
        elif s.buffered >= MAX_BUFFERED_LINES:
            self.flush_all()

    def flush_all(self) -> None:
        """Flush every stream's buffer, in the order the streams first appeared."""
        for key in list(self.streams.lines):
            self.flush(key)

    def flush(self, key: tuple[str, str]) -> None:
        """Run the detector over one stream's buffer and fold its hits in.

        Empties the buffer and advances that stream's start offset so the next
        window's line numbers continue where this one stopped.
        """
        s = self.streams
        lines = s.lines.get(key)
        if not lines:
            return
        s.buffered -= len(lines)
        service, ref = key
        hits = self.detector.detect({"service": service, "ref": ref,
                                     "start": s.starts[key], "lines": lines})
        for hit in hits:
            self.tally.fold(hit)
        s.lines[key] = []
        s.starts[key] = s.seen[key]


@adapter("source", "s3logs")
class S3LogSource:
    """Read the day's CloudWatch->S3 exports and reduce them to a LogWindow.

    `target` is 'YYYY-MM-DD'. Returns the dict shape LogWindow validates.
    """

    def __init__(self, detector=None):
        """`detector` is injectable for tests; otherwise resolved on first use."""
        self._detector = detector

    def detector(self):
        """The injected Detector, else the configured one (`config.get_detector`).

        Through `registry.provider`, not `import config`: config imports
        `adapters`, which imports this module.
        """
        if self._detector is None:
            self._detector = registry.provider("detector")
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
        """The `EvidenceSource` port: one day of logs reduced to a LogWindow-shaped dict.

        `opts["reader"]` is a zero-argument callable yielding (service, ref, line);
        without it the stubbed S3 read raises `NotImplementedError`.
        `opts["detector"]` overrides the detector for this call. Returns `date`,
        `ref`, a sha256 `digest` of every line read, `line_count`, `services`,
        per-signal `counts`, at most MAX_CANDIDATES `candidates` (most frequent
        first, each with at most MAX_SAMPLES_PER_SIGNAL samples), and `truncated`,
        which is True when any cap dropped something. Memory is bounded by
        MAX_BUFFERED_LINES, never by the size of the day.
        """
        reader = opts.get("reader") or (lambda: self._read(target, **opts))
        window = _WindowReducer(opts.get("detector") or self.detector())
        for service, ref, line in reader():
            window.add(service, ref, line)
        window.flush_all()

        tally = window.tally
        ordered, candidates = _candidates(tally.counts, tally.severity_of, tally.samples)

        return {
            "date": target,
            "ref": opts.get("ref", f"s3://<log-export-bucket>/daily/{target}/"),
            "digest": f"sha256:{window.digest.hexdigest()}",
            "line_count": window.line_count,
            "services": sorted(window.services),
            "counts": dict(ordered),
            "candidates": candidates,
            "truncated": tally.truncated or len(ordered) > MAX_CANDIDATES,
        }
