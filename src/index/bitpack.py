import numpy as np

BITS = 8   # K=256 needs 8 bits (was 10 for K=1024)

def pack_codes(indices):
    """indices: (L,) uint8/uint16 -> bytes"""
    return indices.astype(np.uint8).tobytes()

def unpack_codes(packed, L):
    """bytes -> (L,) uint16"""
    return np.frombuffer(packed, dtype=np.uint8).astype(np.uint16)

def batch_approx_distances(all_codes, dist_table):
    """all_codes: (N,L) uint16 | dist_table: (K,) -> (N,) float"""
    return dist_table[all_codes].mean(axis=1)
