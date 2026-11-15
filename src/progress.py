"""Progress + ETA for long runs. Written to a JSON file that scripts/status_v2.py aggregates across processes.

ETA = sum over call kinds of (remaining calls x mean latency of that kind) / workers, multiplied by a calibration
factor (observed wall time / predicted wall time for the uncached calls so far). Calls differ a lot in cost
(MMLU-Pro initial answers with reasoning vs. one-line verdict-first follow-ups), so a single calls/second rate
would be badly biased while the cheap follow-ups and expensive initial calls are interleaved.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict

# Prior seconds per call (latency at 4 concurrent requests, small model) until >= MIN_OBS calls are observed.
PRIORS = {"initial": {"truthfulqa_mc4": 15.0, "mmlu_pro_10": 45.0}, "followup": 8.0}
MIN_OBS = 5


class Progress:
    def __init__(self, path: str, model: str, workers: int, prior_scale: float = 1.0, log=print,
                 log_every_s: float = 120):
        self.path, self.model, self.workers = path, model, workers
        self.prior_scale, self.log, self.log_every_s = prior_scale, log, log_every_s
        self.expected: dict[tuple, int] = defaultdict(int)
        self.completed: dict[tuple, int] = defaultdict(int)
        self.lat: dict[tuple, list[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.t_start = time.time()
        self.t_first_live: float | None = None
        self.live_latency_sum = 0.0
        self.tokens_out = 0
        self.n_cached = 0
        self.status = "running"
        self.current = ""
        self._last_log = 0.0
        self._last_write = 0.0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def expect(self, ds: str, kind: str, n: int) -> None:
        with self.lock:
            self.expected[(ds, kind)] += n

    def cancel(self, ds: str, kind: str, n: int) -> None:
        with self.lock:
            self.expected[(ds, kind)] -= n

    def _mean_latency(self, key: tuple) -> float:
        obs = self.lat[key]
        if len(obs) >= MIN_OBS:
            return sum(obs[-200:]) / len(obs[-200:])
        ds, kind = key
        prior = PRIORS["initial"].get(ds, 30.0) if kind == "initial" else PRIORS["followup"]
        return prior * self.prior_scale

    def done(self, ds: str, kind: str, rec: dict) -> None:
        with self.lock:
            key = (ds, kind)
            self.completed[key] += 1
            if rec.get("cached"):
                self.n_cached += 1
            elif rec.get("latency_s") is not None:
                lat = float(rec["latency_s"])
                if self.t_first_live is None:
                    self.t_first_live = time.time() - lat
                self.lat[key].append(lat)
                self.live_latency_sum += lat
                self.tokens_out += rec.get("output_tokens") or 0
            now = time.time()
            if now - self._last_write > 10:
                self._write(now)
            if now - self._last_log > self.log_every_s:
                self._last_log = now
                s = self.snapshot(now)
                self.log(f"[progress] {self.model}: {s['done_calls']}/{s['expected_calls']} calls "
                         f"({s['pct']:.1f}%), ETA {fmt_dur(s['eta_s'])} ({s['eta_clock']})")

    def snapshot(self, now: float | None = None) -> dict:
        now = now or time.time()
        remaining_pred = sum(max(0, self.expected[k] - self.completed[k]) * self._mean_latency(k)
                             for k in self.expected) / self.workers
        calib = 1.0
        if self.t_first_live is not None and self.live_latency_sum > 0:
            predicted_so_far = self.live_latency_sum / self.workers
            calib = min(3.0, max(0.33, (now - self.t_first_live) / predicted_so_far))
        eta = remaining_pred * calib if self.status == "running" else 0.0
        done = sum(self.completed.values())
        exp = max(done, sum(self.expected.values()))
        per_ds = {}
        for (ds, kind), n in self.expected.items():
            d = per_ds.setdefault(ds, {"done": 0, "expected": 0})
            d["done"] += self.completed[(ds, kind)]
            d["expected"] += n
        return {"model": self.model, "status": self.status, "current": self.current,
                "started": self.t_start, "updated": now, "elapsed_s": now - self.t_start,
                "done_calls": done, "expected_calls": exp, "pct": 100 * done / exp if exp else 0.0,
                "eta_s": eta, "eta_clock": time.strftime("%a %H:%M", time.localtime(now + eta)),
                "calibration": calib, "cached_calls": self.n_cached, "tokens_out": self.tokens_out,
                "per_dataset": per_ds}

    def _write(self, now: float) -> None:
        self._last_write = now
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.snapshot(now), f, indent=1)
        os.replace(tmp, self.path)

    def finish(self, status: str = "done") -> None:
        with self.lock:
            self.status = status
            self._write(time.time())


def fmt_dur(s: float) -> str:
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"
