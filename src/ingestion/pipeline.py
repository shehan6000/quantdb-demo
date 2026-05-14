import threading, time, numpy as np, torch
from collections import deque
from src.utils import zscore_normalize, log

class IngestionPipeline:
    """Ring buffer -> batch VQ-VAE encoder -> staged HNSW merge."""
    def __init__(self, model, store, faiss_index, cfg):
        self.model  = model
        self.store  = store
        self.idx    = faiss_index
        self.T      = cfg["vqvae"]["T"]
        self.flush  = cfg["index"]["flush_threshold"]
        self._buf   = deque(maxlen=cfg["index"]["staging_max_size"])
        self._lock  = threading.Lock()
        self._stage_lats = []
        self._stage_ids  = []
        self._next_id    = store.count()
        self._lag_ms     = 0.0
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        log.info("Ingestion pipeline started.")

    def stop(self):
        self._stop.set()

    def ingest(self, ticker, bar):
        with self._lock:
            self._buf.append({"ticker": ticker, **bar, "_t": time.time()})

    def _loop(self):
        window_buf = {}
        while not self._stop.is_set():
            with self._lock:
                records = [self._buf.popleft()
                           for _ in range(min(256, len(self._buf)))]
            if not records:
                time.sleep(0.02)
                continue
            self._lag_ms = (time.time() - min(r["_t"] for r in records)) * 1000
            for r in records:
                t = r["ticker"]
                window_buf.setdefault(t, []).append(
                    [r["open"], r["high"], r["low"], r["close"],
                     np.log1p(r["volume"])])
                if len(window_buf[t]) >= self.T:
                    w = zscore_normalize(
                        np.array(window_buf[t][-self.T:], dtype=np.float32))
                    self._encode_stage(t, w)
            if len(self._stage_ids) >= self.flush:
                self._merge()

    def _encode_stage(self, ticker, window):
        self.model.eval()
        with torch.no_grad():
            _, z = self.model.encode(torch.from_numpy(window).unsqueeze(0))
        rid = self._next_id; self._next_id += 1
        self._stage_lats.append(z.squeeze(0).numpy())
        self._stage_ids.append(rid)

    def _merge(self):
        if not self._stage_lats: return
        lats = np.array(self._stage_lats, dtype=np.float32)
        ids  = np.array(self._stage_ids,  dtype=np.int64)
        self.idx.add(lats, ids)
        self._stage_lats.clear(); self._stage_ids.clear()
        log.info(f"Merged {len(ids)} new windows into index.")

    @property
    def encoding_lag_ms(self): return self._lag_ms
