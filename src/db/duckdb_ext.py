"""
src/db/duckdb_ext.py

DuckDB PATTERN_LIKE Integration
================================
Registers PATTERN_LIKE as a real DuckDB scalar UDF, making it a first-class
SQL predicate. After registration, queries like:

    SELECT row_id, ticker, ts_start, similarity_score
    FROM   windows
    WHERE  pattern_like(row_id, 0.85) = TRUE
      AND  sector = 'Technology'
    ORDER BY pattern_like_score(row_id) DESC
    LIMIT  20

execute natively inside DuckDB's query engine.

Proposal reference: Section 3.9
"QUANT-DB uses this [DuckDB extension API] to inject three new operator types
into DuckDB's physical plan: VectorScan, FusedScan, and StagingUnion."

Implementation note:
    DuckDB's Python API exposes create_function() for scalar UDFs and
    create_table_function() for table-valued functions. This module uses
    scalar UDFs to implement PATTERN_LIKE as a boolean predicate and
    SIMILARITY_SCORE as a float-valued function. The VectorScan / FusedScan
    table-valued operators described in the proposal are the C++ DuckDB
    extension target (PhD Year-1); the scalar UDF approach here delivers
    identical query semantics with a simpler registration path.
"""

import numpy as np
import duckdb
from typing import Callable, Optional
from src.utils import log


# ─────────────────────────────────────────────────────────────────────────────
# Global encoder state — set once at DB startup, used by all UDF calls
# ─────────────────────────────────────────────────────────────────────────────

_encode_fn:     Optional[Callable]   = None   # (window_array) → latent (d,)
_all_lats:      Optional[np.ndarray] = None   # (N, d) — all stored latent vectors
_all_ids:       Optional[np.ndarray] = None   # (N,)   — row IDs matching rows in DB
_query_lat:     Optional[np.ndarray] = None   # (d,)   — current query latent
_max_dist:      Optional[float]      = None   # kept for back-compat; not used for scoring
_id_to_pos:     dict                 = {}
_rank_scores:   dict                 = {}     # row_id → rank-based similarity score


def register(
    conn:       duckdb.DuckDBPyConnection,
    encode_fn:  Callable,
    all_lats:   np.ndarray,
    all_ids:    np.ndarray,
) -> None:
    """
    Register PATTERN_LIKE and SIMILARITY_SCORE as DuckDB scalar functions.

    Call once at server startup after the VQ-VAE model is loaded.

    Args:
        conn:       DuckDB connection (the same one used by BitPackedStore)
        encode_fn:  callable(window: np.ndarray) → latent: np.ndarray (d,)
                    wraps model.encode(); used to encode the query window
        all_lats:   (N, d) float32 — all stored latent vectors, indexed by row_id
        all_ids:    (N,)   int64   — row IDs for all_lats rows
    """
    global _encode_fn, _all_lats, _all_ids
    _encode_fn = encode_fn
    _all_lats  = all_lats.astype(np.float32)
    _all_ids   = all_ids.astype(np.int64)

    # Build fast row_id → position lookup
    global _id_to_pos
    _id_to_pos = {int(rid): i for i, rid in enumerate(_all_ids)}

    # ── Precompute all distances at query time for normalisation ─────────────
    # _max_dist is set by set_query_window() so similarity scores are always
    # in [0, 1] relative to the actual distance distribution of the dataset.
    # threshold=0.85 then genuinely means "top 15% most similar".

    # ── SIMILARITY_SCORE(row_id) ─────────────────────────────────────────────
    def _similarity_score(row_id: int) -> float:
        # Use precomputed rank-based score: 1.0 = closest row, 0.0 = furthest.
        # Populated by set_query_window(), so threshold=0.85 *guarantees*
        # the top 15% of rows pass — regardless of the query's distance
        # distribution.  The old percentile-normalisation failed when all
        # windows were equidistant from a synthetic query (all scores ≈0.5).
        return _rank_scores.get(int(row_id), 0.0)

    # ── PATTERN_LIKE(row_id, threshold) ─────────────────────────────────────
    # threshold is now a normalised similarity in [0, 1]:
    #   0.85 → top 15% most similar windows pass
    #   0.50 → top 50% pass  (lenient)
    def _pattern_like(row_id: int, threshold: float) -> bool:
        return _similarity_score(row_id) >= threshold

    # Register both functions
    try:
        conn.remove_function("similarity_score")
    except Exception:
        pass
    try:
        conn.remove_function("pattern_like")
    except Exception:
        pass

    # create_function() signature differs across DuckDB versions.
    # Try the modern typed form first; fall back to the untyped form for older releases.
    try:
        import duckdb.typing as _dt
        conn.create_function("similarity_score", _similarity_score,
                             [_dt.BIGINT], _dt.FLOAT)
        conn.create_function("pattern_like", _pattern_like,
                             [_dt.BIGINT, _dt.FLOAT], _dt.BOOLEAN)
    except AttributeError:
        # DuckDB < 0.9 — create_function infers types from Python annotations
        def _similarity_score_typed(row_id: int) -> float:
            return _similarity_score(row_id)
        def _pattern_like_typed(row_id: int, threshold: float) -> bool:
            return _pattern_like(row_id, threshold)
        conn.create_function("similarity_score", _similarity_score_typed)
        conn.create_function("pattern_like", _pattern_like_typed)

    log.info(
        f"DuckDB PATTERN_LIKE registered. "
        f"K-space: {len(_all_ids):,} rows  |  "
        f"Table: similarity_score(row_id), pattern_like(row_id, threshold)"
    )


def set_query_window(query_latent: np.ndarray) -> None:
    """
    Set the current query latent vector before executing a PATTERN_LIKE query.
    Precomputes rank-based similarity scores for every stored row so that
    threshold=0.85 always means "top 15% most similar by rank", regardless
    of the query's absolute distance distribution.

    In the C++ DuckDB extension (PhD Year-1), this will be replaced by passing
    the query latent as a bind parameter to the VectorScan table function.
    """
    global _query_lat, _max_dist, _rank_scores
    _query_lat = query_latent.astype(np.float32)
    if _all_lats is not None and len(_all_lats) > 0:
        dists = np.linalg.norm(_all_lats - _query_lat[None, :], axis=1)
        _max_dist = float(np.percentile(dists, 95))   # kept for back-compat

        # Rank-based similarity: closest row → score=1.0, furthest → score=0.0.
        # argsort gives positions from smallest to largest distance, so
        # rank 0 = nearest neighbour = score 1.0.
        N = len(dists)
        order = np.argsort(dists)          # (N,) indices: nearest first
        scores = np.empty(N, dtype=np.float32)
        scores[order] = 1.0 - np.arange(N, dtype=np.float32) / max(N - 1, 1)
        _rank_scores = {int(_all_ids[i]): float(scores[i]) for i in range(N)}


def refresh_latents(all_lats: np.ndarray, all_ids: np.ndarray) -> None:
    """
    Update the latent matrix and ID map (called after new windows are ingested).
    """
    global _all_lats, _all_ids, _id_to_pos
    _all_lats  = all_lats.astype(np.float32)
    _all_ids   = all_ids.astype(np.int64)
    _id_to_pos = {int(rid): i for i, rid in enumerate(_all_ids)}
    log.info(f"DuckDB UDF latent cache refreshed: {len(_all_ids):,} rows")


def execute_pattern_sql(
    conn:      duckdb.DuckDBPyConnection,
    sql:       str,
    query_lat: np.ndarray,
    limit:     int = 20,
) -> list[dict]:
    """
    Execute a PATTERN-SQL query using the registered DuckDB UDFs.

    Translates PATTERN_LIKE(price_bars, :query, threshold => X) syntax
    to the registered UDF form pattern_like(row_id, X) and executes
    the full query through DuckDB's native engine.

    Args:
        conn:      DuckDB connection with registered UDFs
        sql:       PATTERN-SQL string (may contain PATTERN_LIKE predicate)
        query_lat: (d,) encoded query latent
        limit:     max rows to return

    Returns:
        list of result dicts with row_id, ticker, ts_start, ts_end,
        avg_volume, sector, similarity_score
    """
    set_query_window(query_lat)

    # Translate PATTERN_LIKE(price_bars, :query_window, threshold => X)
    # → pattern_like(row_id, X) AND similarity_score(row_id) AS similarity_score
    import re
    threshold_match = re.search(
        r'PATTERN_LIKE\s*\([^,]+,\s*:[^,]+,\s*threshold\s*=>\s*([\d.]+)',
        sql, re.IGNORECASE
    )
    threshold = float(threshold_match.group(1)) if threshold_match else 0.80

    # Build the translated query
    extra_clauses = _extract_extra_clauses(sql)
    extra_sql = f" AND {extra_clauses}" if extra_clauses.strip() else ""

    translated = f"""
        SELECT
            row_id,
            ticker,
            ts_start,
            ts_end,
            avg_volume,
            sector,
            similarity_score(row_id) AS similarity_score
        FROM windows
        WHERE pattern_like(row_id, {threshold}){extra_sql}
        ORDER BY similarity_score(row_id) DESC
        LIMIT {limit}
    """

    try:
        rows = conn.execute(translated).fetchall()
        cols = ["row_id", "ticker", "ts_start", "ts_end",
                "avg_volume", "sector", "similarity_score"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        log.error(f"DuckDB PATTERN_LIKE query failed: {e}")
        raise


def _extract_extra_clauses(sql: str) -> str:
    """
    Extract relational WHERE clauses from PATTERN-SQL, excluding
    the PATTERN_LIKE predicate itself.

    Handles three positions of PATTERN_LIKE within WHERE:
        WHERE PATTERN_LIKE(...) AND col = x   → "col = x"
        WHERE col = x AND PATTERN_LIKE(...)   → "col = x"
        WHERE PATTERN_LIKE(...)               → ""
    """
    import re

    # Remove PATTERN_LIKE together with any immediately surrounding AND keyword.
    # Three alternation branches, tried left-to-right:
    #   1. "AND PATTERN_LIKE(...)" — PATTERN_LIKE is not the first predicate
    #   2. "PATTERN_LIKE(...) AND" — PATTERN_LIKE is first, more predicates follow
    #   3. bare "PATTERN_LIKE(...)" — only predicate
    cleaned = re.sub(
        r'\s+AND\s+PATTERN_LIKE\s*\([^)]+\)'
        r'|PATTERN_LIKE\s*\([^)]+\)\s*AND\s+'
        r'|PATTERN_LIKE\s*\([^)]+\)',
        '', sql, flags=re.IGNORECASE
    )

    # Extract only what sits between WHERE and ORDER BY / LIMIT.
    # Deliberately omit "$" as a stop-anchor so we never capture ORDER BY
    # or LIMIT keywords when PATTERN_LIKE was the sole WHERE predicate.
    where_match = re.search(
        r'\bWHERE\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b)',
        cleaned, re.IGNORECASE | re.DOTALL
    )
    if not where_match:
        return ""

    clause = where_match.group(1).strip()
    # Strip any residual trailing AND / OR using a word-boundary sub
    # (avoids the rstrip('AND') char-stripping bug).
    clause = re.sub(r'\s*\b(AND|OR)\s*$', '', clause, flags=re.IGNORECASE).strip()
    return clause