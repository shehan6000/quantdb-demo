import numpy as np, torch, time
from src.index.bitpack import batch_approx_distances
from src.utils import log

class QueryExecutor:
    def __init__(self, store, faiss_index, hqo, model, params):
        self.store  = store
        self.idx    = faiss_index
        self.hqo    = hqo
        self.model  = model
        self.params = params

    def execute(self, plan):
        t0  = time.perf_counter()
        pl  = plan.get("pattern_like")
        k   = plan.get("limit", 20)
        rel = plan.get("rel_where", "").strip()

        if pl is None:
            ids  = self.store.relational_filter(rel or "1=1")
            rows = self.store.get_rows_by_ids(ids[:k])
            return {"rows": rows, "latency_ms": (time.perf_counter()-t0)*1000,
                    "plan": "relational_only"}

        param_key = pl["param"].lstrip(":")
        qw = self.params.get(param_key)
        if qw is None:
            raise ValueError(f"Parameter '{param_key}' not provided")

        qw_t = torch.from_numpy(qw.astype(np.float32)).unsqueeze(0)
        _, z_q = self.model.encode(qw_t)
        z_q = z_q.squeeze(0).numpy()
        dist_table = self.model.distance_table(
            torch.from_numpy(qw.astype(np.float32))).numpy()

        hqo_plan  = self.hqo.plan(z_q, pl["threshold"],
                                   plan.get("rel_predicates", []), k)
        strategy  = hqo_plan["plan"]
        kc        = hqo_plan["k_candidates"]

        if strategy == "filter_first":
            result_ids = self._filter_first(z_q, dist_table, rel, kc)
        elif strategy == "fused_scan":
            result_ids = self._fused_scan(z_q, rel, kc)
        elif strategy == "vector_first":
            result_ids = self._vector_first(z_q, rel, kc)
        else:  # hnsw_first
            result_ids = self._hnsw_first(z_q, rel, kc)

        rows = self._rerank(result_ids, z_q, k)
        return {"rows": rows, "latency_ms": (time.perf_counter()-t0)*1000,
                "plan": strategy, "hqo_plan": hqo_plan}

    def _hnsw_first(self, z_q, rel, kc):
        subset = self.store.relational_filter(rel) if rel else None
        if subset is not None and len(subset) == 0:
            return np.array([], dtype=np.int64)
        _, ids = self.idx.search(z_q, kc, id_subset=subset)
        return ids[ids >= 0]

    def _vector_first(self, z_q, rel, kc):
        _, ids = self.idx.search(z_q, kc)
        ids = ids[ids >= 0]
        if rel and len(ids):
            rel_set = set(self.store.relational_filter(rel).tolist())
            ids = np.array([i for i in ids if i in rel_set], dtype=np.int64)
        return ids

    def _filter_first(self, z_q, dist_table, rel, kc):
        rel_ids = self.store.relational_filter(rel) if rel else None
        if rel_ids is not None and len(rel_ids) == 0:
            return np.array([], dtype=np.int64)
        if rel_ids is not None:
            codes = self.store.get_codes_by_ids(rel_ids)
            dists = batch_approx_distances(codes, dist_table)
            n     = min(kc, len(dists))
            top   = np.argpartition(dists, n-1)[:n] if n > 0 else []
            return rel_ids[top]
        _, ids = self.idx.search(z_q, kc)
        return ids[ids >= 0]

    def _fused_scan(self, z_q, rel, kc):
        all_ids, all_lats = self.store.get_all_latents()
        if len(all_ids) == 0:
            return np.array([], dtype=np.int64)
        mask = np.ones(len(all_ids), dtype=bool)
        if rel:
            rel_set = set(self.store.relational_filter(rel).tolist())
            mask = np.array([i in rel_set for i in all_ids], dtype=bool)
        fids, flats = all_ids[mask], all_lats[mask]
        if len(fids) == 0:
            return np.array([], dtype=np.int64)
        dists = np.linalg.norm(flats - z_q[None], axis=1)
        n     = min(kc, len(dists))
        top   = np.argpartition(dists, n-1)[:n]
        return fids[top]

    def _rerank(self, result_ids, z_q, k):
        if len(result_ids) == 0:
            return []
        rows = self.store.get_rows_by_ids(result_ids[:k])
        lat_rows = self.store.conn.execute(
            f"SELECT row_id, latent FROM windows WHERE row_id IN "
            f"({','.join(map(str, result_ids[:k].tolist()))})"
        ).fetchall()
        id2lat = {r[0]: np.array(list(r[1]), dtype=np.float32) for r in lat_rows}
        for row in rows:
            z = id2lat.get(row["row_id"], np.zeros_like(z_q))
            row["similarity_score"] = float(1.0 / (1.0 + np.linalg.norm(z_q - z)))
        rows.sort(key=lambda r: r["similarity_score"], reverse=True)
        return rows[:k]
