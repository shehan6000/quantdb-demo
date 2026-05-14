import numpy as np, math, json, pathlib, time
from src.utils import log

class CodebookManager:
    def __init__(self, K=256, models_path="models", logs_path="logs",
                 entropy_ratio=0.5, recon_delta=0.30):
        self.K = K
        self.version_log = pathlib.Path(logs_path) / "codebook_versions.json"
        self.entropy_ratio = entropy_ratio
        self.recon_delta   = recon_delta
        self.versions = json.load(open(self.version_log)) if self.version_log.exists() else []

    def compute_assignment_entropy(self, code_indices):
        counts = np.bincount(code_indices.flatten().astype(int), minlength=self.K)
        probs  = counts / (counts.sum() + 1e-9)
        probs  = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def is_stale(self, code_indices, current_recon, baseline_recon):
        H      = self.compute_assignment_entropy(code_indices)
        H_min  = self.entropy_ratio * math.log(self.K)
        delta  = (current_recon - baseline_recon) / (abs(baseline_recon) + 1e-9)
        reasons = []
        if H < H_min:
            reasons.append(f"low_entropy H={H:.3f}<{H_min:.3f}")
        if delta > self.recon_delta:
            reasons.append(f"high_recon_delta={delta:.2%}")
        return bool(reasons), "; ".join(reasons) if reasons else "healthy"

    def latest_version(self):
        return max((v["version"] for v in self.versions), default=0)
