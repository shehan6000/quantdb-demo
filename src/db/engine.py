"""
src/db/engine.py  (updated — Layer 3 + DuckDB PATTERN_LIKE integration)
"""
import numpy as np, torch, pathlib
from src.utils import load_config, log
from src.compression.model    import VQVAE
from src.compression.codebook import CodebookManager
from src.index.store          import BitPackedStore
from src.index.faiss_index    import QuantIndex
from src.optimizer.hqo        import HybridQueryOptimizer
from src.query.parser         import parse as parse_sql
from src.query.executor       import QueryExecutor
from src.ingestion.pipeline   import IngestionPipeline
from src.execution.layer3     import Layer3Engine, benchmark_layer3_vs_brute_force
from src.db.duckdb_ext        import register as register_duckdb_ext
from src.db.duckdb_ext        import execute_pattern_sql, refresh_latents


class QuantDB:
    def __init__(self, cfg_path="config/config.yaml"):
        self.cfg = load_config(cfg_path)
        vc, dc, ic = self.cfg["vqvae"], self.cfg["data"], self.cfg["index"]

        # ── VQ-VAE model ──────────────────────────────────────────────────────
        self.model = VQVAE(T=vc["T"], F=vc["F"], d=vc["d"], K=vc["K"],
                           ema_decay=float(vc.get("ema_decay", 0.99)),
                           ema_epsilon=float(vc.get("ema_epsilon", 1e-5)))
        ckpt = pathlib.Path(dc["models_path"]) / "vqvae_best.pt"
        if ckpt.exists():
            self.model.load_state_dict(
                torch.load(ckpt, map_location="cpu")["model"])
            log.info(f"Loaded VQ-VAE from {ckpt}")
        self.model.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, vc["T"], vc["F"])
            idx, _ = self.model.encode(dummy)
            self.L = int(idx.shape[1])

        # ── Storage + FAISS index ─────────────────────────────────────────────
        self.store     = BitPackedStore(dc["db_path"], self.L, vc["d"], vc["K"])
        self.faiss_idx = QuantIndex(
            d=vc["d"], index_type=ic["type"],
            M=ic["hnsw_M"], ef_construction=ic["hnsw_ef_construction"],
            ef_search=ic["hnsw_ef_search"],
            nlist=ic["ivf_nlist"], nprobe=ic["ivf_nprobe"])

        self._populate_index()

        # ── Layer 3 engine ────────────────────────────────────────────────────
        codebook_weights = (
            self.model.quantizer.codebook.weight.detach().numpy()
        )                                                           # (K, d)
        self.layer3 = Layer3Engine(codebook_weights)
        log.info(
            f"Layer 3 initialised — distance table: "
            f"{self.layer3.dist_table.size_bytes()} bytes "
            f"({'fits in L1' if self.layer3.dist_table.fits_in_l1() else 'exceeds L1'})"
        )

        # ── DuckDB PATTERN_LIKE UDF ───────────────────────────────────────────
        row_ids, lats = self.store.get_all_latents()
        if len(lats) > 0:
            def _encode_fn(window: np.ndarray) -> np.ndarray:
                self.model.eval()
                with torch.no_grad():
                    t = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                    _, z = self.model.encode(t)
                return z.squeeze(0).numpy()

            register_duckdb_ext(
                conn=self.store.conn,
                encode_fn=_encode_fn,
                all_lats=lats,
                all_ids=row_ids,
            )

        # ── HQO ───────────────────────────────────────────────────────────────
        self.hqo = HybridQueryOptimizer(self.cfg, self.store.count(), vc["d"])
        if len(lats) > 0:
            self.hqo.fit_estimators(lats)

        # ── Codebook manager ──────────────────────────────────────────────────
        self.cb_mgr = CodebookManager(
            K=vc["K"], models_path=dc["models_path"], logs_path=dc["logs_path"],
            entropy_ratio=self.cfg["codebook"]["entropy_min_ratio"],
            recon_delta=self.cfg["codebook"]["recon_error_max_delta"])

        # ── Ingestion pipeline ────────────────────────────────────────────────
        self.pipeline = IngestionPipeline(
            self.model, self.store, self.faiss_idx, self.cfg)

        # ── Query executor ────────────────────────────────────────────────────
        self.executor = QueryExecutor(
            self.store, self.faiss_idx, self.hqo, self.model, {})

    def _populate_index(self):
        row_ids, lats = self.store.get_all_latents()
        if len(row_ids) == 0:
            dc = self.cfg["data"]
            lat_f = pathlib.Path(dc["encoded_path"]) / "latents.npy"
            idx_f = pathlib.Path(dc["encoded_path"]) / "indices.npy"
            if lat_f.exists():
                lats    = np.load(lat_f)
                indices = np.load(idx_f).astype(np.uint16)
                n       = len(lats)
                row_ids = np.arange(n, dtype=np.int64)
                log.info(f"Loading {n:,} pre-encoded windows from disk...")
                self.store.insert_batch(
                    row_ids, ["synthetic"]*n, ["2018-01-01"]*n, ["2024-01-01"]*n,
                    indices, lats, [1e6]*n, ["NA"]*n)
                row_ids, lats = self.store.get_all_latents()
        if len(row_ids) > 0:
            log.info(f"Populating FAISS index with {len(row_ids):,} vectors...")
            self.faiss_idx.add(lats, row_ids)

    def query(self, sql: str, params: dict = None) -> dict:
        """
        Execute a PATTERN-SQL query.
        Routes through the DuckDB PATTERN_LIKE UDF if the query contains
        PATTERN_LIKE; falls back to the HQO executor otherwise.
        """
        params = params or {}

        if "PATTERN_LIKE" in sql.upper() and "query_window" in params:
            # Encode the query window
            window = params["query_window"]
            self.model.eval()
            with torch.no_grad():
                t = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                _, z = self.model.encode(t)
            z_q = z.squeeze(0).numpy()

            # Try DuckDB PATTERN_LIKE route first
            try:
                import time as _time
                t0 = _time.perf_counter()
                rows = execute_pattern_sql(
                    conn=self.store.conn,
                    sql=sql,
                    query_lat=z_q,
                    limit=params.get("limit", 20),
                )
                latency_ms = (_time.perf_counter() - t0) * 1000
                if rows:
                    return {
                        "rows": rows,
                        "plan": "DuckDB-PATTERN_LIKE",
                        "n_results": len(rows),
                        "engine": "duckdb_udf",
                        "latency_ms": round(latency_ms, 2),
                    }
                log.info("DuckDB UDF returned 0 rows, falling back to FAISS/HQO")
            except Exception as e:
                log.warning(f"DuckDB UDF route failed ({e}), falling back to HQO executor")

        # HQO / FAISS executor path
        # executor.execute() takes a parsed plan dict (not raw SQL).
        # We zero the threshold so FAISS does pure top-K retrieval — the
        # similarity score is added back below via exact latent L2.
        import re as _re
        hqo_sql = _re.sub(
            r',\s*threshold\s*=>\s*[\d.]+', ', threshold => 0.0',
            sql, flags=_re.IGNORECASE
        )
        plan = parse_sql(hqo_sql)
        self.executor.params = params
        result = self.executor.execute(plan)

        # If executor returned rows but no similarity_score column, and we
        # have z_q from the DuckDB path above, rerank by exact latent L2.
        rows = result.get("rows", [])
        if rows and "z_q" in locals():
            all_ids, all_lats = self.store.get_all_latents()
            id_to_pos = {int(rid): i for i, rid in enumerate(all_ids)}
            for row in rows:
                pos = id_to_pos.get(int(row.get("row_id", -1)), -1)
                if pos >= 0:
                    dist = float(np.linalg.norm(z_q - all_lats[pos]))
                    row["similarity_score"] = round(1.0 / (1.0 + dist), 4)
            rows.sort(key=lambda r: r.get("similarity_score", 0), reverse=True)
            result["rows"] = rows

        if "plan" in result:
            result["engine"] = "faiss_hqo"
            result["duckdb_metadata"] = True
        return result

    def query_with_layer3(self, query_window: np.ndarray,
                          threshold: float = 0.85,
                          top_k: int = 20,
                          rel_where: str = None) -> dict:
        """
        Direct Layer 3 query path — bypasses FAISS, uses BitScanKernel.
        Useful for demonstrating Layer 3 independently.
        """
        import time

        # Encode
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(query_window.astype(np.float32)).unsqueeze(0)
            _, z = self.model.encode(t)
        z_q = z.squeeze(0).numpy()

        # Prepare distance table
        t0 = time.perf_counter()
        self.layer3.prepare(z_q)
        table_ms = (time.perf_counter() - t0) * 1000

        # Load all codes
        all_ids, all_lats = self.store.get_all_latents()
        all_codes = self.store.get_codes_by_ids(all_ids)  # (N, L) uint16

        if rel_where:
            # Fused scan — evaluate relational mask over DuckDB, then fuse
            rel_ids = self.store.relational_filter(rel_where)
            rel_set = set(rel_ids.tolist())
            rel_mask = np.array([int(rid) in rel_set for rid in all_ids], dtype=bool)

            result = self.layer3.execute_fused_scan(
                codes=all_codes,
                row_ids=all_ids,
                rel_mask=rel_mask,
                dist_threshold=threshold,
                top_k=top_k,
            )
            plan_used = "FusedScan"
        else:
            result = self.layer3.execute_bit_scan(
                codes=all_codes,
                row_ids=all_ids,
                threshold=threshold,
                top_k=top_k,
            )
            plan_used = "BitScan"

        # Rerank by exact L2
        rows = self.store.get_rows_by_ids(result.row_ids)
        for row in rows:
            pos = np.searchsorted(all_ids, row["row_id"])
            if pos < len(all_ids) and all_ids[pos] == row["row_id"]:
                dist = float(np.linalg.norm(z_q - all_lats[pos]))
                row["similarity_score"] = float(1.0 / (1.0 + dist))
            else:
                row["similarity_score"] = 0.0
        rows.sort(key=lambda r: r["similarity_score"], reverse=True)

        return {
            "rows": rows[:top_k],
            "plan": plan_used,
            "n_scanned": result.n_scanned,
            "scan_ms": result.latency_ms,
            "table_prep_ms": table_ms,
            "layer3_info": self.layer3.table_info(),
        }

    def benchmark_layer3(self, n_queries: int = 50) -> dict:
        """Run Layer 3 vs brute-force benchmark and return results dict."""
        codebook = self.model.quantizer.codebook.weight.detach().numpy()
        all_ids, all_lats = self.store.get_all_latents()
        all_codes = self.store.get_codes_by_ids(all_ids)
        results = benchmark_layer3_vs_brute_force(
            codebook=codebook,
            all_codes=all_codes,
            all_lats=all_lats,
            row_ids=all_ids,
            n_queries=n_queries,
        )
        log.info(f"Layer3 benchmark: p50 speedup={results['speedup_p50']:.1f}x "
                 f"p95 speedup={results['speedup_p95']:.1f}x")
        return results

    def check_codebook_health(self) -> dict:
        try:
            return self.cb_mgr.check(self.model)
        except AttributeError:
            # Fallback for CodebookManager versions without .check()
            cb = self.model.quantizer.codebook.weight.detach().numpy()
            import numpy as np
            idx_f = pathlib.Path(self.cfg["data"]["encoded_path"]) / "indices.npy"
            if idx_f.exists():
                codes = np.load(idx_f).flatten().astype(int)
                K = self.cfg["vqvae"]["K"]
                counts = np.bincount(codes, minlength=K)
                probs = counts / counts.sum()
                H = float(-np.sum(probs[probs>0] * np.log(probs[probs>0]+1e-12)))
                H_max = float(np.log(K))
                stale = H / H_max < self.cfg["codebook"]["entropy_min_ratio"]
                return {"stale": stale, "entropy": round(H,3), "reason": "healthy" if not stale else "low_entropy"}
            return {"stale": False, "entropy": None, "reason": "no_index_file"}

    def start_ingestion(self):
        if not self.pipeline._thread.is_alive():
            self.pipeline.start()

    def ingest_bar(self, ticker: str, bar: dict):
        self.pipeline.ingest(ticker, bar)

    def close(self):
        self.pipeline.stop()
        self.store.close()