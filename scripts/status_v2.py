"""One-line-per-model progress + overall ETA for a (possibly running) v2 study.

    python scripts/status_v2.py [results/v2]
Models not yet started are estimated from the per-call wall time of finished/running models, scaled by
the config's eta_prior_scale.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402

from src.progress import fmt_dur  # noqa: E402

out = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "results/v2")
cfg = yaml.safe_load(open(os.path.join(ROOT, "config_v2.yaml")))
order = ["llama3.2-3b-instruct", "qwen3-vl-4b-instruct", "llama3.1-8b-instruct"]
scale = {m["name"]: m.get("eta_prior_scale", 1.0) for m in cfg["models"]}
snaps = {}
for m in order:
    p = os.path.join(out, "progress", f"{m}.json")
    if os.path.exists(p):
        snaps[m] = json.load(open(p))
now = time.time()
# Queued models: projected total duration of the most advanced started model, scaled by eta_prior_scale ratio.
ref = None
for m in order:
    s = snaps.get(m)
    if s and s["done_calls"] > 50 and s["cached_calls"] < s["done_calls"] / 2:
        ref = (m, s["elapsed_s"] + (s["eta_s"] if s["status"] != "done" else 0.0))
total_eta, lines = 0.0, []
for m in order:
    s = snaps.get(m)
    if s and s["status"] == "done":
        lines.append(f"  {m:24s} done      {s['done_calls']} calls in {fmt_dur(s['elapsed_s'])}")
    elif s:
        stale = now - s["updated"]
        eta = max(0.0, s["eta_s"] - stale)
        total_eta += eta
        lines.append(f"  {m:24s} {s['status']:9s} {s['done_calls']}/{s['expected_calls']} ({s['pct']:.1f}%)"
                     f", ETA {fmt_dur(eta)}" + (f"  [no update for {fmt_dur(stale)}]" if stale > 300 else ""))
    else:
        est = ref[1] * scale[m] / scale[ref[0]] if ref else None
        total_eta += est or 0
        lines.append(f"  {m:24s} queued    est. {fmt_dur(est) if est else '?'} (scaled from {ref[0] if ref else '-'})")
print(f"[{time.strftime('%a %H:%M')}] overall ETA {fmt_dur(total_eta)} -> finishes ~"
      f"{time.strftime('%a %H:%M', time.localtime(now + total_eta))}")
print("\n".join(lines))
