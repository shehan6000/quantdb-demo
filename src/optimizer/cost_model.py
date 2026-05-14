class CostModel:
    def __init__(self, cfg):
        h = cfg.get("hqo", {})
        self.c_vec    = h.get("vector_scan_cost_per_row", 1.0)
        self.c_rel    = h.get("filter_cost_per_row",      0.1)
        self.c_fused  = h.get("fused_cost_per_row",       0.6)
        self.c_hnsw   = 50.0

    def select_plan(self, n_total, vec_card, rel_card, n_rel, k, cheap_preds):
        c = {
            "vector_first": self.c_vec * n_total + self.c_rel * vec_card * max(1, n_rel),
            "filter_first": self.c_rel * n_total * max(1, n_rel) + self.c_vec * rel_card,
            "hnsw_first":   self.c_hnsw + self.c_rel * k * 5 * max(1, n_rel),
            "fused_scan":   self.c_fused * n_total * max(1, n_rel)
                            if (cheap_preds and n_rel <= 2) else float("inf"),
        }
        best = min(c, key=c.__getitem__)
        return best, c[best]
