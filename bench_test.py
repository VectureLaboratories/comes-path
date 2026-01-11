import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from comes_path.core.solver import ComesSolver
from comes_path.core.baselines import numba_dijkstra

def generate_grid_graph(dim):
    n = dim * dim
    print(f"Generating grid graph with {n} nodes...")
    rows = []
    cols = []
    data = []
    for r in range(dim):
        for c in range(dim):
            u = r * dim + c
            if c + 1 < dim:
                v = r * dim + (c + 1)
                rows.append(u); cols.append(v)
                data.append(np.random.rand() + 1.0)
            if r + 1 < dim:
                v = (r + 1) * dim + c
                rows.append(u); cols.append(v)
                data.append(np.random.rand() + 1.0)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    return adj.tocsr()

def run_benchmark():
    dim = 1000
    n = dim * dim
    source = 0
    adj_csr = generate_grid_graph(dim)
    
    print("Starting SciPy Dijkstra (C++ Heap)...")
    start = time.time()
    d_scipy = dijkstra(adj_csr, indices=source, directed=False)
    print(f"SciPy Dijkstra: {time.time() - start:.4f}s")
    
    print("\nStarting Numba Dijkstra (Baseline Heap)...")
    numba_dijkstra(adj_csr.indices, adj_csr.indptr, adj_csr.data, source, 10)
    start = time.time()
    d_numba = numba_dijkstra(adj_csr.indices, adj_csr.indptr, adj_csr.data, source, n)
    numba_time = time.time() - start
    print(f"Numba Dijkstra: {numba_time:.4f}s")
    
    print("\nStarting ComesSolver (Frontier Partitioning)...")
    solver = ComesSolver(adj_csr)
    solver.shortest_path(source, target=10)
    start = time.time()
    d_comes = solver.shortest_path(source)
    comes_time = time.time() - start
    print(f"ComesSolver: {comes_time:.4f}s")
    
    improvement_vs_numba = (numba_time - comes_time) / numba_time * 100
    print(f"\nAlgorithmic Dominance (Comes vs Numba-Heap): {improvement_vs_numba:.2f}%")
    
    mask = np.isfinite(d_scipy)
    if np.allclose(d_scipy[mask], d_comes[mask], atol=1e-5):
        print("Validation PASSED.")
    else:
        print("Validation FAILED.")

if __name__ == "__main__":
    run_benchmark()
