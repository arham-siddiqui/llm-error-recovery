"""Run the v2 main study for one or more models (one process per model is the intended use).

    python scripts/run_v2.py --config config_v2.yaml --models llama3.2-3b-instruct
    python scripts/run_v2.py --config config_v2.yaml --n 6 --output-dir results/v2_smoke   # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from src import prompts as P  # noqa: E402
from src.datasets import load_all  # noqa: E402
from src.experiments_v2 import (CALLS_PER_ITEM_MULTI, CALLS_PER_ITEM_SINGLE, assign_design,  # noqa: E402
                                initial_correct_rate, run_dataset_v2)
from src.models import ResponseCache, make_client  # noqa: E402
from src.progress import Progress, fmt_dur  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v2.yaml"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=None, help="override items per dataset (smoke runs)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(ROOT, args.output_dir or cfg["output_dir"])
    seed = cfg["seed"]
    ds_cfg = {k: v for k, v in cfg["datasets"].items() if not args.datasets or k in args.datasets}
    data_dir = os.path.join(ROOT, cfg["data_dir"]) if args.n is None else os.path.join(out, "data")
    examples, load_report = load_all(ds_cfg, seed, data_dir, n_override=args.n)
    for name, exs in examples.items():
        mt = ds_cfg[name].get("multiturn_n", 0) if args.n is None else max(1, args.n // 2)
        assign_design(exs, mt)
        load_report[name]["multiturn_n"] = sum(e["metadata"]["multiturn"] for e in exs)
        print(f"[load] {name}: {len(exs)} items ({load_report[name]['multiturn_n']} in multi-turn subset)",
              flush=True)

    for m in [m for m in cfg["models"] if not args.models or m["name"] in args.models]:
        m = dict(m)
        prior_scale = m.pop("eta_prior_scale", 1.0)
        cache = ResponseCache(os.path.join(ROOT, "results", "cache", f"{m['name']}.jsonl"))
        client = make_client(m, cfg["generation"], seed, cache)
        prog = Progress(os.path.join(out, "progress", f"{client.name}.json"), client.name, cfg["concurrency"],
                        prior_scale, log=lambda s: print(s, flush=True))
        for name, exs in examples.items():
            prog.expect(name, "initial", len(exs))
            prog.expect(name, "followup", len(exs) * CALLS_PER_ITEM_SINGLE
                        + sum(e["metadata"]["multiturn"] for e in exs) * CALLS_PER_ITEM_MULTI)
        t0 = time.time()
        print(f"[model] {client.name} ({client.backend}:{client.model})", flush=True)
        n_calls = 0
        try:
            for name, exs in examples.items():
                prog.current = name
                tt = time.time()
                recs = run_dataset_v2(client, exs, seed, cfg["concurrency"],
                                      os.path.join(out, "raw", client.name, f"{name}.jsonl"), prog)
                n_calls += len(recs)
                print(f"  [{name}] {len(recs)} calls ({sum(r['cached'] for r in recs)} cached) in "
                      f"{fmt_dur(time.time() - tt)}; initial {initial_correct_rate(recs)}", flush=True)
        except BaseException:
            prog.finish("failed")
            raise
        prog.finish("done")
        manifest = {"model": client.describe(), "seed": seed, "config_path": os.path.relpath(args.config, ROOT),
                    "generation": cfg["generation"], "datasets": load_report, "n_override": args.n,
                    "n_calls": n_calls, "elapsed_s": time.time() - t0, "finished_unix": time.time(),
                    "prompts": {"system": P.SYSTEM_PROMPT, "followup_format": P.FOLLOWUP_FORMAT,
                                "assert": P.ASSERT_TEMPLATES, "neutral": P.NEUTRAL_TEMPLATES,
                                "pressure": P.PRESSURE_ARMS},
                    "python": sys.version.split()[0], "platform": platform.platform()}
        json.dump(manifest, open(os.path.join(out, "raw", client.name, "manifest.json"), "w"), indent=1,
                  default=str)
        print(f"[done] {client.name}: {n_calls} calls in {fmt_dur(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()
