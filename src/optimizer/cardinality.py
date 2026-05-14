import numpy as np
from sklearn.neighbors import KernelDensity
from src.utils import log

class GeometricEstimator:
    def __init__(self, bandwidth=0.5):
        self.kde     = KernelDensity(bandwidth=bandwidth, algorithm='ball_tree')
        self.fitted  = False
        self.n_total = 0

    def fit(self, latents):
        self.n_total = len(latents)
        sample = latents if len(latents) <= 50_000 else latents[
            np.random.choice(len(latents), 50_000, replace=False)]
        log.info("Fitting geometric cardinality estimator...")
        self.kde.fit(sample)
        self.fitted = True

    def estimate(self, z_query, eps):
        if not self.fitted:
            return self.n_total // 10
        log_d  = self.kde.score_samples(z_query.reshape(1, -1))[0]
        frac   = min(1.0, float(np.exp(log_d)) * (eps ** 2) * z_query.shape[0])
        return max(1, int(frac * self.n_total))

class LearnedEstimator:
    """Placeholder — trained on query logs after system is live."""
    def __init__(self):
        self.fitted = False

    def estimate(self, z_query, eps):
        return -1
