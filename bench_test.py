import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from comes_path.core.solver import ComesSolver

def generate_grid_graph(dim):
    """
    Generate a 2D grid graph (dim x dim nodes) with random weights.
    """
    n = dim * dim
    print(f"Generating grid graph with {n} nodes...")
    rows = []
    cols = []
    data = []
    
    for r in range(dim):
        for c in range(dim):
            u = r * dim + c
            # Horizontal adjacency settlement
            if c + 1 < dim:
                v = r * dim + (c + 1)
                rows.append(u); cols.append(v)
                data.append(np.random.rand() + 1.0)
            # Vertical adjacency settlement
            if r + 1 < dim:
                v = (r + 1) * dim + c
                rows.append(u); cols.append(v)
                data.append(np.random.rand() + 1.0)
                
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    return adj.tocsr()

def run_benchmark():
    # Configuration: 10^6 node grid topology
    dim = 1000
    source = 0
    target = dim * dim - 1
    
    adj_csr = generate_grid_graph(dim)
    
    print("Starting SciPy Dijkstra...")
    start = time.time()
    d_scipy = dijkstra(adj_csr, indices=source, directed=False)
    end = time.time()
    scipy_time = end - start
    print(f"SciPy Dijkstra: {scipy_time:.4f}s")
    
    print("Starting ComesSolver...")
    solver = ComesSolver(adj_csr)
    
    # Trigger JIT compilation
    solver.shortest_path(source, target=10)
    
    start = time.time()
    d_comes = solver.shortest_path(source)
    end = time.time()
    comes_time = end - start
    print(f"ComesSolver (warm): {comes_time:.4f}s")
    
    improvement = (scipy_time - comes_time) / scipy_time * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Validation of result integrity
    mask = np.isfinite(d_scipy)
    if np.allclose(d_scipy[mask], d_comes[mask], atol=1e-5):
        print("Validation PASSED.")
    else:
        print("Validation FAILED.")

if __name__ == "__main__":
    run_benchmark()