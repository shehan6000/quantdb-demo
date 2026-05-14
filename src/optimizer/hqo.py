import numpy as np
from src.optimizer.cardinality import GeometricEstimator
from src.optimizer.cost_model  import CostModel
from src.utils import log

CHEAP_OPS = {"=", ">", "<", ">=", "<=", "BETWEEN", "IN"}

class HybridQueryOptimizer:
    def __init__(self, cfg, n_total, d):
        self.n_total = n_total
        self.d       = d
        self.cm      = CostModel(cfg)
        self.geo     = GeometricEstimator()
        self.alpha   = cfg.get("hqo", {}).get("alpha", 5)

    def fit_estimators(self, latents):
        self.geo.fit(latents)

    def _est_vec(self, z, eps):
        return self.geo.estimate(z, eps)

    def _est_rel(self, predicates):
        sel = 1.0
        for p in predicates:
            op = str(p.get("op", "=")).upper()
            sel *= 0.05 if op == "=" else 0.15 if op in ("IN",) else 0.3
        return max(1, int(sel * self.n_total))

    def plan(self, z_query, eps, predicates, k):
        vec_card = self._est_vec(z_query, eps)
        rel_card = self._est_rel(predicates)
        n_rel    = len(predicates)
        cheap    = all(str(p.get("op","=")).upper() in CHEAP_OPS for p in predicates)

        plan_name, cost = self.cm.select_plan(
            self.n_total, vec_card, rel_card, n_rel, k, cheap)
        log.info(f"HQO plan={plan_name} cost={cost:.1f} "
                 f"vec_card={vec_card} rel_card={rel_card}")
        return {"plan": plan_name, "cost": cost, "k_candidates": k * self.alpha,
                "vec_card_est": vec_card, "rel_card_est": rel_card}
