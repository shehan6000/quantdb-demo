import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time, numpy as np, pandas as pd, json, argparse
from src.db.engine import QuantDB
from src.utils import load_config, log

def run_benchmark(n_queries=100, k=20, cfg_path="config/config.yaml"):
    cfg = load_config(cfg_path)
    db  = QuantDB(cfg_path)
    _, all_lats = db.store.get_all_latents()
    n   = len(all_lats)
    log.info(f"Benchmark: n={n:,} queries={n_queries} k={k}")

    rng     = np.random.default_rng(42)
    sample  = rng.choice(n, size=min(n_queries, n), replace=False)
    results = []

    for qi, si in enumerate(sample):
        z_q = all_lats[si]
        # Ground truth
        t0 = time.perf_counter()
        dists = np.linalg.norm(all_lats - z_q[None], axis=1)
        gt    = set(np.argsort(dists)[:k].tolist())
        gt_ms = (time.perf_counter() - t0) * 1000
        # HNSW
        t0 = time.perf_counter()
        _, I = db.faiss_idx.search(z_q, k)
        q_ms = (time.perf_counter() - t0) * 1000
        recall = len(set(I[I>=0].tolist()) & gt) / len(gt)
        results.append({"latency_ms": q_ms, "recall": recall,
                         "speedup": gt_ms / (q_ms + 1e-9)})
        if qi % 10 == 0:
            log.info(f"  Q{qi:3d}: {q_ms:.2f}ms recall@{k}={recall:.3f} "
                     f"speedup={gt_ms/(q_ms+1e-9):.1f}x")

    df = pd.DataFrame(results)
    summary = {
        "n_total": n, "n_queries": n_queries,
        "p50_ms":  float(df.latency_ms.quantile(.50)),
        "p95_ms":  float(df.latency_ms.quantile(.95)),
        "p99_ms":  float(df.latency_ms.quantile(.99)),
        "recall":  float(df.recall.mean()),
        "speedup": float(df.speedup.mean()),
    }
    print(json.dumps(summary, indent=2))
    df.to_csv(f"{cfg['data']['logs_path']}/bench_results.csv", index=False)
    db.close()
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k",       type=int, default=20)
    a  = ap.parse_args()
    run_benchmark(a.queries, a.k)
