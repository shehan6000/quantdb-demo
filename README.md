# QUANT-DB — Semantic Financial Time-Series Database

## Project Overview

QUANT-DB is a research-oriented database prototype designed for semantic pattern search over financial time-series data. It extends SQL-based analytical querying with similarity-based retrieval, enabling users to identify historically similar market patterns while applying conventional relational filters.

The system combines deep-learning-based time-series compression, approximate nearest-neighbor search, hybrid query optimization, and cache-aware execution to improve the efficiency of financial pattern retrieval.

**Project focus:** Database Systems · AI/ML · Time-Series Analytics · Vector Search · Query Optimization

## Key Features

- **Semantic financial pattern search:** Retrieve historical price windows with shapes similar to a given query pattern.
- **Domain-adapted compression:** Represent financial time-series windows using learned discrete codes.
- **Approximate nearest-neighbor retrieval:** Use HNSW indexing to accelerate similarity search.
- **SQL-integrated similarity queries:** Combine pattern matching with relational conditions such as ticker, sector, and trading volume.
- **Hybrid query optimization:** Select execution strategies according to estimated query costs and predicate selectivity.
- **Cache-aware execution:** Use compact distance lookup tables to reduce computational and memory-access overhead.
- **Interactive research interface:** Explore the pipeline, submit pattern queries, inspect retrieved windows, and review system benchmarks.
- **System health monitoring:** Track codebook utilization, assignment entropy, and database initialization status.

## System Architecture

The architecture consists of interconnected data ingestion, representation learning, indexing, query planning, and execution components.

### 1. Financial Data Processing

The prototype processes financial OHLCV data consisting of:

- Open price
- High price
- Low price
- Close price
- Trading volume

Time-series observations are divided into overlapping sliding windows of 60 bars, with a stride of one bar. Each window contains 300 numerical values across five features.

Per-window z-score normalization reduces sensitivity to absolute price levels, allowing the system to focus on the relative shape of market movements. Volume is log-transformed to reduce the influence of extreme trading-volume observations.

### 2. Deep Learning-Based Compression

A Vector-Quantized Variational Autoencoder (VQ-VAE) learns compact representations of financial patterns.

The encoder uses one-dimensional convolutional layers and residual downsampling to transform each input window into a 64-dimensional latent representation. Vector quantization maps these representations to entries in a learned codebook containing 256 prototypes.

The decoder reconstructs the original time-series window, allowing reconstruction quality to be evaluated during model development.

This representation-learning stage supports compact pattern storage and provides the basis for similarity estimation.

### 3. Vector Indexing and Similarity Retrieval

The prototype integrates FAISS with a Hierarchical Navigable Small World (HNSW) index for approximate nearest-neighbor search over latent representations.

The index uses a graph-based search structure to identify candidate windows without requiring exhaustive distance calculations against every stored representation.

The implementation also includes a compact distance-table approach for estimating distances between quantized representations.

### 4. Hardware-Aware Query Execution

The execution layer explores ways to reduce the cost of similarity calculations through compact data structures and cache-efficient access patterns.

Its main components include:

- **Distance Table:** A 256 × 256 table of precomputed distances between codebook entries.
- **BitScanKernel:** A vectorized scanning approach implemented using NumPy operations.
- **FusedScanKernel:** An execution strategy designed to combine similarity evaluation and relational filtering in a single pass.

The distance table is designed to fit within the CPU's L1 cache, reducing memory-access overhead during lookup-intensive operations.

### 5. Hybrid Query Optimizer

The Hybrid Query Optimizer (HQO) evaluates alternative query execution strategies for workloads combining vector similarity and relational filtering.

The prototype supports four execution-plan approaches:

- HNSW-first retrieval
- Vector-first linear scanning
- Filter-first relational evaluation
- Fused similarity and relational scanning

A kernel density estimation (KDE)-based cardinality estimator models the distribution of latent representations to estimate the number of candidates satisfying a similarity predicate.

The optimizer uses these estimates to select an execution plan according to the expected workload characteristics.

### 6. SQL Integration and Analytical Storage

DuckDB provides the relational analytical engine, while FAISS supports approximate vector retrieval.

Custom SQL user-defined functions expose pattern matching and similarity scoring through the database interface. This enables similarity-based retrieval to be combined with conventional relational predicates and result ordering.

The query pipeline coordinates pattern encoding, similarity scoring, relational filtering, result ranking, and visualization.

### 7. Interactive Research Dashboard

A Streamlit interface provides an interactive environment for investigating the system.

The dashboard includes:

- Architecture and pipeline exploration
- Financial pattern visualization
- Live pattern-query demonstrations
- Similarity-result inspection
- Codebook utilization analysis
- System health monitoring
- Benchmark and research-contribution summaries

Plotly is used to visualize financial windows, retrieval results, and system metrics.

## Technical Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | PyTorch, VQ-VAE |
| Numerical computing | NumPy |
| Data processing | Pandas |
| Analytical database | DuckDB |
| Vector search | FAISS, HNSW |
| Query optimization | KDE-based cardinality estimation, cost-based plan selection |
| Query interface | SQL, custom user-defined functions |
| Interactive application | Streamlit |
| Visualization | Plotly |
| Financial data | OHLCV time-series, yfinance-compatible ingestion |
| Performance engineering | SIMD-enabled NumPy operations, cache-aware lookup tables |

## Research Contributions

### SQL-Native Semantic Pattern Queries

Integrates time-series similarity into a relational query workflow, allowing users to combine semantic pattern matching with conventional database predicates.

### Learned Discrete Representations

Applies VQ-VAE-based compression to financial time-series, producing compact representations suitable for similarity estimation and indexing.

### Cost-Aware Hybrid Query Planning

Explores adaptive execution planning for queries combining approximate vector retrieval and relational filtering.

### Cache-Efficient Similarity Computation

Investigates compact distance tables and fused scanning strategies to reduce computational and memory-access overhead.

### Integrated AI and Database Architecture

Combines representation learning, vector indexing, relational analytics, query optimization, and interactive visualization within a unified research prototype.

## Benchmark Results

The following results are reported by the prototype's existing benchmark configuration.

| Metric | Reported result |
|---|---:|
| Dataset | 14,410 synthetic time-series windows |
| Financial tickers represented | 10 |
| Window length | 60 bars |
| Latent representation dimension | 64 |
| Codebook size | 256 |
| Median query latency (p50) | 0.198 ms |
| 95th-percentile latency (p95) | 0.235 ms |
| 99th-percentile latency (p99) | 0.335 ms |
| Recall@20 | 1.000 |
| Reported speedup over brute-force search | 14.67× |
| Reported reconstruction MSE | 0.038 |
| Distance-table row size | 1,024 bytes |

The reported query benchmarks use a workload of 50 queries with a top-20 retrieval setting. Query execution does not require a GPU in the described prototype.

**Benchmark scope:** These measurements were obtained on a small synthetic dataset. They demonstrate the prototype's reported performance under its current evaluation conditions; they do not establish equivalent performance on large-scale real-world financial data.

The reconstruction metric is reported separately from the query-performance measurements. It describes the model's reconstruction objective and should not be interpreted as a general guarantee of financial-pattern accuracy.

## Current Scope and Future Research

The current prototype establishes an experimental foundation for investigating semantic financial time-series retrieval.

Potential research extensions include:

- Evaluation on historical market data from a broader range of financial instruments.
- Scaling experiments across hundreds of thousands to millions of time-series windows.
- Native C++ execution kernels with AVX-512 vectorization.
- Improved approximate nearest-neighbor indexing and cardinality estimation.
- Disk-resident storage and large-scale index partitioning.
- Freehand and image-based financial pattern queries.
- Further integration of the compression, indexing, storage, and execution components into a native database engine.
- Development of a reproducible public benchmark suite.

These are proposed research directions rather than completed capabilities of the current prototype.

## Project Significance

QUANT-DB explores the intersection of database systems, machine learning, and performance engineering. Its central objective is to investigate how learned representations and vector-search techniques can be integrated with relational query processing to support efficient semantic retrieval over financial time-series.

The project demonstrates an end-to-end research approach encompassing data preparation, neural representation learning, vector indexing, query optimization, execution-engine design, benchmarking, and interactive system evaluation.

**Primary research areas:** Database Internals · Query Processing · Vector Databases · Time-Series Data Management · Representation Learning · Approximate Nearest-Neighbor Search · AI-Driven Analytics
