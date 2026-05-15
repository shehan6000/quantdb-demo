"""
api/server.py  (updated — Layer 3 + DuckDB PATTERN_LIKE endpoints)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.db.engine import QuantDB
from src.utils import load_config, zscore_normalize, log

app = FastAPI(title="QUANT-DB API", version="0.3.0")
_db: QuantDB = None


@app.on_event("startup")
def _startup():
    global _db
    _db = QuantDB()


@app.on_event("shutdown")
def _shutdown():
    if _db:
        _db.close()


# ─── Request models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    sql:          str
    query_window: list = None

class Layer3QueryRequest(BaseModel):
    query_window: list
    threshold:    float = 0.85
    top_k:        int   = 20
    rel_where:    str   = None

class IngestBar(BaseModel):
    ticker: str; open: float; high: float; low: float
    close: float; volume: float; ts: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query")
def run_query(req: QueryRequest):
    """Standard PATTERN-SQL query — routes through HQO or DuckDB UDF."""
    if _db is None:
        raise HTTPException(503, "DB not ready")
    params = {}
    if req.query_window:
        w = zscore_normalize(np.array(req.query_window, dtype=np.float32))
        params["query_window"] = w
    try:
        return _db.query(req.sql, params)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/query/layer3")
def run_layer3_query(req: Layer3QueryRequest):
    """
    Direct Layer 3 query path — BitScanKernel or FusedScanKernel.
    Demonstrates Layer 3 independently of FAISS.
    """
    if _db is None:
        raise HTTPException(503, "DB not ready")
    w = zscore_normalize(np.array(req.query_window, dtype=np.float32))
    try:
        return _db.query_with_layer3(
            query_window=w,
            threshold=req.threshold,
            top_k=req.top_k,
            rel_where=req.rel_where,
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/layer3/info")
def layer3_info():
    """Return Layer 3 engine metadata."""
    if _db is None:
        raise HTTPException(503, "DB not ready")
    return _db.layer3.table_info()


@app.get("/layer3/benchmark")
def layer3_benchmark(n_queries: int = 50):
    """
    Run Layer 3 vs brute-force benchmark on demand.
    Returns timing comparison for display in the demo.
    """
    if _db is None:
        raise HTTPException(503, "DB not ready")
    try:
        return _db.benchmark_layer3(n_queries=n_queries)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest")
def ingest(bar: IngestBar):
    if _db is None:
        raise HTTPException(503, "DB not ready")
    _db.start_ingestion()
    _db.ingest_bar(bar.ticker, bar.dict())
    return {"status": "queued"}


@app.get("/health")
def health():
    if _db is None:
        return {"status": "not_ready"}
    n   = _db.store.count()
    cb  = _db.check_codebook_health()
    lag = (_db.pipeline.encoding_lag_ms
           if _db.pipeline._thread.is_alive() else "pipeline_idle")
    return {
        "status":         "ok",
        "n_windows":      n,
        "codebook":       cb,
        "encoding_lag_ms": lag,
        "layer3": {
            "active":       True,
            "table_bytes":  _db.layer3.dist_table.size_bytes(),
            "fits_l1":      _db.layer3.dist_table.fits_in_l1(),
            "duckdb_udf":   True,
        },
    }


if __name__ == "__main__":
    cfg = load_config()["demo"]
    uvicorn.run("api.server:app", host=cfg["api_host"],
                port=cfg["api_port"], reload=False)
