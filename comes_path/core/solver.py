"""
Vecture Laboratories // Comes Solver Execution Pipeline

Operational Directive:
Govern the transition between Comes Partitioning and Hybrid Fallback protocols.
"""

import numpy as np
from .structures import FrontierBucket
from .relaxation import relax_pivots, identify_pivots
from .partitioning import partition_graph
from numba import njit
import heapq

class ComesSolver:
    """
    Absolute shortest-path derivation engine.
    
    Orchestrates CSR-based traversal utilizing the Comes breakthrough.
    Maintains clinical precision through the Dial-Comes Invariant.
    """
    def __init__(self, adjacency_matrix_csr=None):
        self.indices = None
        self.indptr = None
        self.data = None
        self.pivots = None
        self.params = None
        self.is_sparse_fallback = False
        if adjacency_matrix_csr is not None:
            self.set_graph(adjacency_matrix_csr)

    def set_graph(self, csr_matrix):
        """Ingest CSR topology and determine operational mode."""
        self.indices = csr_matrix.indices
        self.indptr = csr_matrix.indptr
        self.data = csr_matrix.data
        n = len(self.indptr) - 1
        m = len(self.indices)
        
        if m < 2 * n:
            self.is_sparse_fallback = True
        else:
            self.is_sparse_fallback = False
            self.params = partition_graph(csr_matrix)
            self.pivots = identify_pivots(self.indptr)

    def shortest_path(self, source, target=None):
        """
        Derive shortest distance from source node.
        Returns full distance array or single scalar if target is specified.
        """
        num_nodes = len(self.indptr) - 1
        distances = np.full(num_nodes, np.inf, dtype=np.float64)
        settled = np.zeros(num_nodes, dtype=np.bool_)
        distances[source] = 0.0
        
        if self.is_sparse_fallback:
            res = self._dijkstra_fallback(source, target, distances, settled)
            return res[target] if target is not None else res
        
        frontier = FrontierBucket(
            num_buckets=self.params["num_buckets"], 
            bucket_width=self.params["bucket_width"]
        )
        frontier.insert(source, 0.0)
        
        res = self._solve(
            source, target, distances, settled, self.indices, self.indptr, self.data, self.pivots, frontier
        )
        return res[target] if target is not None else res

    def _dijkstra_fallback(self, source, target, distances, settled):
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if settled[u]: continue
            settled[u] = True
            if target is not None and u == target: break
            for i in range(self.indptr[u], self.indptr[u+1]):
                v = self.indices[i]
                weight = self.data[i]
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    heapq.heappush(pq, (distances[v], v))
        return distances

    @staticmethod
    @njit
    def _solve(source, target, distances, settled, indices, indptr, data, pivots, frontier):
        while not frontier.is_empty():
            u = frontier.pop_min()
            if u == -1: break
            if settled[u]: continue
            settled[u] = True
            if target is not None and u == target: break
            
            relax_pivots(u, distances, indices, indptr, data, pivots, frontier, lookahead_depth=2)
        return distances

def shortest_path(G, source, target=None):
    """Protocol Entry Point: Standard interface for ComesSolver."""
    solver = ComesSolver(G)
    return solver.shortest_path(source, target)
