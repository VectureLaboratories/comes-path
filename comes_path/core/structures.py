"""
Vecture Laboratories // Frontier Partitioning Structures

Operational Directive:
Maintain sub-logarithmic overhead via quantized distance mapping.
"""

import numpy as np
from numba import int64, float64, uint64, njit
from numba.experimental import jitclass

spec = [
    ('buckets', int64[:, :]),
    ('bucket_counts', int64[:]),
    ('bucket_width', float64),
    ('num_buckets', int64),
    ('current_index', int64),
    ('count_in_frontier', int64),
    ('bitmask', uint64[:]),
]

@jitclass(spec)
class FrontierBucket:
    def __init__(self, num_buckets, bucket_width, initial_capacity=4096):
        self.num_buckets = ((num_buckets + 63) // 64) * 64
        self.bucket_width = bucket_width
        self.buckets = np.full((self.num_buckets, initial_capacity), -1, dtype=np.int64)
        self.bucket_counts = np.zeros(self.num_buckets, dtype=np.int64)
        self.current_index = 0
        self.count_in_frontier = 0
        self.bitmask = np.zeros(self.num_buckets // 64, dtype=np.uint64)

    def _set_bit(self, idx):
        self.bitmask[idx >> 6] |= (uint64(1) << (uint64(idx) & uint64(63)))

    def _clear_bit(self, idx):
        self.bitmask[idx >> 6] &= ~(uint64(1) << (uint64(idx) & uint64(63)))

    def insert(self, node_id, distance):
        idx = int(distance / self.bucket_width) % self.num_buckets
        count = self.bucket_counts[idx]
        
        # Robust resize logic
        if count >= self.buckets.shape[1]:
            new_cap = self.buckets.shape[1] * 2
            new_buckets = np.full((self.num_buckets, new_cap), -1, dtype=np.int64)
            # Copy all buckets
            for i in range(self.num_buckets):
                if self.bucket_counts[i] > 0:
                    new_buckets[i, :self.bucket_counts[i]] = self.buckets[i, :self.bucket_counts[i]]
            self.buckets = new_buckets
            
        self.buckets[idx, count] = node_id
        self.bucket_counts[idx] = count + 1
        self.count_in_frontier += 1
        self._set_bit(idx)

    def pop_min(self):
        if self.count_in_frontier == 0:
            return -1
        while True:
            idx = self.current_index % self.num_buckets
            if self.bucket_counts[idx] > 0:
                break
            mask_idx = idx >> 6
            if self.bitmask[mask_idx] == 0:
                self.current_index = (self.current_index + 64) & ~63
            else:
                self.current_index += 1
        actual_idx = self.current_index % self.num_buckets
        self.bucket_counts[actual_idx] -= 1
        node_id = self.buckets[actual_idx, self.bucket_counts[actual_idx]]
        if self.bucket_counts[actual_idx] == 0:
            self._clear_bit(actual_idx)
        self.count_in_frontier -= 1
        return node_id

    def is_empty(self):
        return self.count_in_frontier == 0