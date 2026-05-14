import numpy as np, duckdb, pathlib
from src.index.bitpack import pack_codes, unpack_codes
from src.utils import log

class BitPackedStore:
    def __init__(self, db_path, L, d, K):
        self.L, self.d, self.K = L, d, K
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS windows (
                row_id     BIGINT PRIMARY KEY,
                ticker     VARCHAR,
                ts_start   VARCHAR,
                ts_end     VARCHAR,
                codes_blob BLOB,
                latent     DOUBLE[],
                codebook_v SMALLINT DEFAULT 0,
                avg_volume DOUBLE,
                sector     VARCHAR
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON windows(ticker)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts     ON windows(ts_start)")
        self.conn.commit()

    def insert_batch(self, row_ids, tickers, ts_starts, ts_ends,
                     codes, latents, volumes, sectors, codebook_v=0):
        rows = [
            (int(row_ids[i]), str(tickers[i]), str(ts_starts[i]), str(ts_ends[i]),
             pack_codes(codes[i]),
             latents[i].astype(np.float64).tolist(),
             int(codebook_v), float(volumes[i]), str(sectors[i]))
            for i in range(len(row_ids))
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def get_all_latents(self, codebook_v=None):
        q = "SELECT row_id, latent FROM windows"
        if codebook_v is not None:
            q += f" WHERE codebook_v = {int(codebook_v)}"
        rows = self.conn.execute(q).fetchall()
        if not rows:
            return np.array([], dtype=np.int64), np.zeros((0, self.d), dtype=np.float32)
        ids  = np.array([r[0] for r in rows], dtype=np.int64)
        lats = np.array([list(r[1]) for r in rows], dtype=np.float32)
        return ids, lats

    def get_codes_by_ids(self, row_ids):
        ids_str = ",".join(map(str, row_ids.tolist()))
        rows    = self.conn.execute(
            f"SELECT row_id, codes_blob FROM windows WHERE row_id IN ({ids_str})"
        ).fetchall()
        id2blob = {r[0]: bytes(r[1]) for r in rows}
        codes   = np.zeros((len(row_ids), self.L), dtype=np.uint16)
        for i, rid in enumerate(row_ids):
            if rid in id2blob:
                codes[i] = unpack_codes(id2blob[rid], self.L)
        return codes

    def relational_filter(self, sql_where):
        rows = self.conn.execute(
            f"SELECT row_id FROM windows WHERE {sql_where}").fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def get_rows_by_ids(self, row_ids):
        if len(row_ids) == 0:
            return []
        ids_str = ",".join(map(str, row_ids.tolist()))
        rows = self.conn.execute(
            f"SELECT row_id, ticker, ts_start, ts_end, avg_volume, sector "
            f"FROM windows WHERE row_id IN ({ids_str})"
        ).fetchall()
        return [dict(zip(["row_id","ticker","ts_start","ts_end","avg_volume","sector"], r))
                for r in rows]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM windows").fetchone()[0]

    def close(self):
        self.conn.close()
