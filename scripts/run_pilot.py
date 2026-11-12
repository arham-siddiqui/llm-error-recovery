"""Run the pilot end-to-end.

    python scripts/run_pilot.py --config config.yaml
    python scripts/run_pilot.py --config config.yaml --n 2 --output-dir results/smoke   # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # project root only; src/ itself is never on sys.path

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.datasets import load_all  # noqa: E402
from src.experiments import run_graded_dataset, run_sycophancy_dataset  # noqa: E402
from src.metrics import compute_summary, flatten  # noqa: E402
from src.models import ResponseCache, make_client  # noqa: E402


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, example) with every condition's outcome."""
    rows = {}
    for _, r in df.iterrows():
        key = (r.model_name, r.dataset, r.example_id)
        row = rows.setdefault(key, {"model_name": r.model_name, "dataset": r.dataset, "example_id": r.example_id,
                                    "gold": r.gold})
        col = r.condition if r.condition != "pressure" else f"pressure_r{r['round']}"
        row[f"{col}_parsed"] = r.parsed
        row[f"{col}_status"] = r.status
        if r.offered_answer is not None and r.condition != "initial":
            row[f"{col}_offered"] = r.offered_answer
            row[f"{col}_adopted_offered"] = r.adopted_offered
    return pd.DataFrame(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--n", type=int, default=None, help="override examples per dataset (e.g. 2 for smoke)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(ROOT, args.output_dir or cfg["output_dir"])
    os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]
    ds_cfg = {k: v for k, v in cfg["datasets"].items() if not args.datasets or k in args.datasets}
    data_dir = os.path.join(ROOT, cfg["data_dir"]) if args.n is None else os.path.join(out, "data")

    t0 = time.time()
    print(f"[load] datasets: {list(ds_cfg)}")
    examples, load_report = load_all(ds_cfg, seed, data_dir, n_override=args.n)
    for k, v in load_report.items():
        print(f"  {k}: {v['loaded']}/{v['requested']} ({v['status'][:90]})")

    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    records, manifest_models = [], []
    for m in models:
        # one cache file per model -> separate processes can run different models concurrently
        cache = ResponseCache(os.path.join(ROOT, "results", "cache", f"{m['name']}.jsonl"))
        client = make_client(m, cfg["generation"], seed, cache)
        manifest_models.append(client.describe())
        print(f"[model] {client.name} ({client.backend}:{client.model})")
        for ds, exs in examples.items():
            tt = time.time()
            print(f"  [{ds}] {len(exs)} examples")
            if ds == "sycophancy_fp":
                recs = run_sycophancy_dataset(client, exs, cfg["concurrency"])
            else:
                pmax = ds_cfg[ds].get("pressure_max", cfg.get("multiturn", {}).get("max_per_dataset"))
                recs = run_graded_dataset(client, exs, seed, cfg["concurrency"], pmax,
                                          log=lambda s: print(s, flush=True))
            records.extend(recs)
            print(f"    {len(recs)} calls ({sum(r['cached'] for r in recs)} cached) in {time.time() - tt:.0f}s",
                  flush=True)

    with open(os.path.join(out, "raw_responses.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(records)
    graded = df[df.dataset != "sycophancy_fp"].copy()
    wide_table(graded).to_csv(os.path.join(out, "graded_results.csv"), index=False)
    summary = compute_summary(graded)
    flatten(summary).to_csv(os.path.join(out, "summary_metrics.csv"), index=False)
    summary.to_pickle(os.path.join(out, "summary.pkl"))
    manifest = {"started_unix": t0, "elapsed_s": time.time() - t0, "seed": seed, "config_path": args.config,
                "n_override": args.n, "generation": cfg["generation"], "models": manifest_models,
                "datasets": load_report, "n_calls": len(records),
                "n_cached_calls": int(df.cached.sum()), "python": sys.version.split()[0],
                "platform": platform.platform(),
                "total_tokens_in": int(df.input_tokens.fillna(0).sum()),
                "total_tokens_out": int(df.output_tokens.fillna(0).sum()),
                "total_cost_usd": float(df.cost_usd.fillna(0).sum())}
    json.dump(manifest, open(os.path.join(out, "run_manifest.json"), "w"), indent=2, default=str)
    print(f"[done] {len(records)} calls, {time.time() - t0:.0f}s. Outputs in {out}")

    if not args.no_report:
        from src.reporting import build_reports
        build_reports(out, os.path.join(ROOT, "reports") if args.n is None else os.path.join(out, "reports"))


if __name__ == "__main__":
    main()
