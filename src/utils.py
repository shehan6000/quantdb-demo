import yaml, pathlib, numpy as np, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quantdb")

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def zscore_normalize(w):
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True) + 1e-8
    return (w - mu) / sd

def sliding_windows(data, T, stride=1):
    M = (len(data) - T) // stride + 1
    idx = np.arange(T)[None] + stride * np.arange(M)[:, None]
    return data[idx]

def ensure_dirs(*paths):
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)
