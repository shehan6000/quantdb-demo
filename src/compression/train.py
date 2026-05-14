import torch, numpy as np, pandas as pd, pathlib, json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.utils import load_config, zscore_normalize, sliding_windows, log, ensure_dirs
from src.compression.model import VQVAE

def _synthetic_ohlcv(n_bars, seed):
    """Regime-switching GBM: (n_bars, 5) OHLCV. Reproducible."""
    rng = np.random.default_rng(seed)
    regimes = [(0.0003, 0.008), (-0.0002, 0.009), (0.0001, 0.005), (0.0, 0.020)]
    closes = [100.0]
    for i in range(n_bars - 1):
        mu, sigma = regimes[(i // 60) % len(regimes)]
        closes.append(max(closes[-1] * (1 + rng.normal(mu, sigma)), 1.0))
    closes = np.array(closes)
    noise  = rng.uniform(0.002, 0.012, n_bars)
    highs  = closes * (1 + noise)
    lows   = closes * (1 - noise)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    volume = rng.lognormal(13.5, 0.8, n_bars)
    return np.stack([opens, highs, lows, closes, volume], axis=1).astype(np.float32)

def build_data(cfg):
    raw   = pathlib.Path(cfg["data"]["raw_path"])
    raw.mkdir(parents=True, exist_ok=True)
    tickers = [t for t in cfg["data"]["tickers"] if isinstance(t, str) and t.strip()]
    dfs = []
    for i, t in enumerate(tickers):
        fp = raw / f"{t}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
        else:
            log.info(f"Generating synthetic data for {t}...")
            arr = _synthetic_ohlcv(1500, seed=i * 17 + 42)
            idx = pd.date_range("2018-01-01", periods=1500, freq="B")
            df  = pd.DataFrame(arr, index=idx, columns=["open","high","low","close","volume"])
            df.to_parquet(fp)
        df["ticker"] = t
        dfs.append(df[["open","high","low","close","volume","ticker"]])
    log.info(f"Loaded {sum(len(d) for d in dfs):,} bars across {len(tickers)} tickers.")
    return pd.concat(dfs, ignore_index=True)

def build_windows(df, T):
    windows = []
    for _, g in df.groupby("ticker", sort=False):
        arr = g[["open","high","low","close","volume"]].values.astype(np.float32)
        arr[:, 4] = np.log1p(arr[:, 4])
        ws = sliding_windows(arr, T)
        windows.append(np.array([zscore_normalize(w) for w in ws], dtype=np.float32))
    return np.concatenate(windows, axis=0)

def train(cfg_path="config/config.yaml"):
    cfg = load_config(cfg_path)
    vc, dc = cfg["vqvae"], cfg["data"]
    T, F, d, K = vc["T"], vc["F"], vc["d"], vc["K"]
    ensure_dirs(dc["models_path"], dc["encoded_path"], dc["logs_path"])

    df      = build_data(cfg)
    windows = build_windows(df, T)
    log.info(f"Total windows: {len(windows):,}")

    split = int(len(windows) * vc["train_split"])
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(windows[:split])),
                       batch_size=vc["batch_size"], shuffle=True, num_workers=0)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(windows[split:])),
                       batch_size=vc["batch_size"], shuffle=False, num_workers=0)

    model = VQVAE(T=T, F=F, d=d, K=K, beta=vc["beta"],
                  ema_decay=float(vc["ema_decay"]),
                  ema_epsilon=float(vc["ema_epsilon"]))
    opt   = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=vc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, vc["epochs"])

    best_val, history = float("inf"), []
    mpath = pathlib.Path(dc["models_path"])

    for epoch in range(1, vc["epochs"] + 1):
        model.train()
        tr_loss = sum(
            (lambda loss: (opt.zero_grad(), loss.backward(), opt.step(), loss.item())[-1])(
                model(xb)[2])
            for (xb,) in tr_dl
        ) / len(tr_dl)

        model.eval()
        va_loss = va_recon = 0.0
        all_idx = []
        with torch.no_grad():
            for (xb,) in va_dl:
                _, idx, loss, info = model(xb)
                va_loss  += loss.item()
                va_recon += info["recon"]
                all_idx.append(idx.reshape(-1))
        va_loss  /= len(va_dl)
        va_recon /= len(va_dl)
        sched.step()

        unique = torch.cat(all_idx).unique().numel()
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss,
                         "recon": va_recon, "codes_used": unique})
        log.info(f"E{epoch:03d} train={tr_loss:.4f} val={va_loss:.4f} "
                 f"recon={va_recon:.4f} codes={unique}/{K}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "cfg": vc}, mpath / "vqvae_best.pt")

    torch.save({"model": model.state_dict(), "cfg": vc}, mpath / "vqvae_final.pt")
    with open(pathlib.Path(dc["logs_path"]) / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Pre-encoding {len(windows):,} windows...")
    model.eval()
    all_idx, all_z = [], []
    with torch.no_grad():
        for i in range(0, len(windows), vc["batch_size"]):
            xb = torch.from_numpy(windows[i:i+vc["batch_size"]])
            idx, z = model.encode(xb)
            all_idx.append(idx.numpy()); all_z.append(z.numpy())
    indices = np.concatenate(all_idx).astype(np.uint16)
    latents = np.concatenate(all_z).astype(np.float32)
    np.save(f"{dc['encoded_path']}/indices.npy", indices)
    np.save(f"{dc['encoded_path']}/latents.npy",  latents)
    log.info(f"Saved {len(indices):,} encoded windows to {dc['encoded_path']}/")
    log.info("Training complete.")
    return model

if __name__ == "__main__":
    train()
