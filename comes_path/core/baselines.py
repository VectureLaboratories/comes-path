"""
Vecture Laboratories // Algorithmic Baselines

Operational Directive:
Provide JIT-optimized standard Dijkstra implementation for clinical comparison.
"""

import numpy as np
from numba import njit, int64, float64

@njit(fastmath=True, cache=True)
def heappush(heap, counts, item_dist, item_node):
    idx = counts[0]
    heap[idx, 0] = item_dist
    heap[idx, 1] = item_node
    counts[0] += 1
    
    curr = idx
    while curr > 0:
        parent = (curr - 1) // 2
        if heap[curr, 0] < heap[parent, 0]:
            tmp_d = heap[curr, 0]
            tmp_n = heap[curr, 1]
            heap[curr, 0] = heap[parent, 0]
            heap[curr, 1] = heap[parent, 1]
            heap[parent, 0] = tmp_d
            heap[parent, 1] = tmp_n
            curr = parent
        else:
            break

@njit(fastmath=True, cache=True)
def heappop(heap, counts):
    if counts[0] <= 0:
        return -1.0, -1
    
    res_dist = heap[0, 0]
    res_node = int(heap[0, 1])
    
    counts[0] -= 1
    last_idx = counts[0]
    heap[0, 0] = heap[last_idx, 0]
    heap[0, 1] = heap[last_idx, 1]
    
    curr = 0
    while True:
        left = 2 * curr + 1
        right = 2 * curr + 2
        smallest = curr
        
        if left < counts[0] and heap[left, 0] < heap[smallest, 0]:
            smallest = left
        if right < counts[0] and heap[right, 0] < heap[smallest, 0]:
            smallest = right
            
        if smallest != curr:
            tmp_d = heap[curr, 0]
            tmp_n = heap[curr, 1]
            heap[curr, 0] = heap[smallest, 0]
            heap[curr, 1] = heap[smallest, 1]
            heap[smallest, 0] = tmp_d
            heap[smallest, 1] = tmp_n
            curr = smallest
        else:
            break
            
    return res_dist, res_node

@njit(fastmath=True, cache=True)
def numba_dijkstra(indices, indptr, data, source, num_nodes):
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    distances[source] = 0.0
    settled = np.zeros(num_nodes, dtype=np.bool_)
    
    # Ensure heap is large enough for all potential relaxations
    # In worst case, every edge could be a push.
    heap = np.empty((len(indices) + num_nodes, 2), dtype=np.float64)
    counts = np.zeros(1, dtype=np.int64)
    
    heappush(heap, counts, 0.0, float(source))
    
    while counts[0] > 0:
        d, u_f = heappop(heap, counts)
        u = int(u_f)
        
        if settled[u]:
            continue
        settled[u] = True
        
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            weight = data[i]
            new_dist = d + weight
            
            if new_dist < distances[v]:
                distances[v] = new_dist
                heappush(heap, counts, new_dist, float(v))
                
    return distances