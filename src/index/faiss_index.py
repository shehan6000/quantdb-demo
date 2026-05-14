import faiss, numpy as np
from src.utils import log

class QuantIndex:
    def __init__(self, d=64, index_type="hnsw", M=16, ef_construction=200,
                 ef_search=64, nlist=64, nprobe=8):
        self.d, self.index_type = d, index_type
        self.ef_search, self.nprobe = ef_search, nprobe
        if index_type == "hnsw":
            base = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
            base.hnsw.efConstruction = ef_construction
        else:
            quantizer = faiss.IndexFlatL2(d)
            base = faiss.IndexIVFPQ(quantizer, d, nlist, d // 4, 8)
        self.index = faiss.IndexIDMap2(base)   # IDMap2 supports reconstruct()

    def _set_search_params(self):
        if self.index_type == "hnsw":
            faiss.downcast_index(self.index.index).hnsw.efSearch = self.ef_search
        else:
            faiss.downcast_index(self.index.index).nprobe = self.nprobe

    def train(self, vectors):
        if self.index_type == "ivfpq":
            self.index.index.train(vectors.astype(np.float32))

    def add(self, vectors, row_ids):
        self.index.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32),
            row_ids.astype(np.int64))

    def search(self, query, k, id_subset=None):
        """query: (d,) -> (distances, row_ids) shape (k,)"""
        self._set_search_params()
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)

        if id_subset is not None and 0 < len(id_subset) < self.index.ntotal:
            # Build a temporary flat index over the filtered subset
            tmp = faiss.IndexIDMap2(faiss.IndexFlatL2(self.d))
            vecs = np.zeros((len(id_subset), self.d), dtype=np.float32)
            for i, rid in enumerate(id_subset):
                try:
                    self.index.reconstruct(int(rid), vecs[i])
                except Exception:
                    pass
            tmp.add_with_ids(vecs, id_subset.astype(np.int64))
            D, I = tmp.search(q, min(k, len(id_subset)))
        else:
            D, I = self.index.search(q, k)

        return D[0], I[0]

    def save(self, path):
        faiss.write_index(self.index, str(path))

    def load(self, path):
        self.index = faiss.read_index(str(path))

    @property
    def ntotal(self):
        return self.index.ntotal
