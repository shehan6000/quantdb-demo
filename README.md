# QUANT-DB

**A Hybrid Vector-Relational Query Engine for Semantic Financial Pattern Search**

QUANT-DB is a research database engine that makes shape similarity a first-class SQL operation for financial time-series data. It lets analysts search historical price data for recurring patterns — consolidations, breakouts, mean-reversion setups — using a single declarative SQL predicate, instead of hand-written scripts over raw price arrays.

```sql
SELECT ticker, start_date, similarity_score
FROM price_windows
WHERE PATTERN_LIKE(bars, :query, threshold => 0.85)
  AND sector = 'Technology'
  AND volume > 1000000
ORDER BY similarity_score DESC
LIMIT 20;
```

## Problem

Quant analysts routinely look for recurring price patterns in historical market data, but no database system treats this as a native query operation. Current options fall short in different ways:

- **Dynamic Time Warping (DTW)** — exact, but O(N·T²) and computationally intractable at scale.
- **General vector search (pgvector, Pinecone)** — approximate, but domain-agnostic with no financial structure.
- **Charting tools (e.g. TrendSpider)** — usable, but application-specific and non-programmable.

No existing system combines SQL expressiveness, domain-adapted vector similarity, and cost-based hybrid query optimization.

**Core research question:** How do you make shape similarity a first-class SQL operation, at sub-millisecond latency, across 50M+ stored windows, without sacrificing relational expressiveness?

## Architecture

QUANT-DB is built as three integrated layers, each independently publishable:

| Layer | Component | Description |
|---|---|---|
| 1 | **Domain-adapted compression** | A VQ-VAE encodes each 60-bar OHLCV window (T=60, F=5) into a 64-dim latent vector, quantized to one of 256 codebook entries. ~10× storage reduction, <4% reconstruction error, and an L1-resident distance table for fast approximate scans. |
| 2 | **Hybrid Query Optimizer (HQO)** | A cost-based optimizer that jointly reasons over vector ANN scans and relational filters, selecting among four execution plans (HNSW-first, vector-first, filter-first, fused scan) using a geometric KDE cardinality estimator. Extends the System R optimization framework (Selinger et al., 1979) to the vector-relational setting. |
| 3 | **Hardware-aware execution engine** | An L1-resident distance table (1,024 B for K=256), a NumPy-vectorized BitScanKernel (C++ AVX-512 planned for Year 1), and a FusedScanKernel that interleaves vector and relational predicate evaluation in a single pass. Integrates with DuckDB via a `PATTERN_LIKE` scalar UDF. |

## Prototype status

A working prototype exists prior to formal research commencement. All figures below are measured, not estimated.

| Metric | Value | Conditions |
|---|---|---|
| Query latency (p50) | 0.198 ms | 50 queries, k=20, N=14,410 windows |
| Query latency (p95) | 0.235 ms | Same benchmark |
| Query latency (p99) | 0.335 ms | Same benchmark |
| Recall@20 | 1.000 | Perfect recall across all 50 runs |
| Speedup vs. brute-force | 14.67× | vs. exact NumPy L2 scan |
| Reconstruction MSE | 0.038 | Normalized scale; target < 0.05 |
| Distance table size | 1,024 B | K=256; confirmed L1-resident |

**Stack:** PyTorch (VQ-VAE) · FAISS HNSW (m=16, ef_construction=200) · HybridQueryOptimizer + GeometricEstimator · Layer3Engine (DistanceTable, BitScanKernel) · DuckDB (`PATTERN_LIKE`, `SIMILARITY_SCORE` UDFs) · FastAPI backend · Streamlit demo (six tabs)


- Wang et al. (2021), *Milvus: A purpose-built vector data management system*, SIGMOD.

---

**Author:** Shehan Liyanaarachchi
