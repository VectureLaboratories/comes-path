import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from comes_path.core.solver import ComesSolver
from comes_path.core.baselines import numba_dijkstra
import networkx as nx

def generate_scale_free_graph(n, m_edges):
    """
    Generate a Barabási–Albert scale-free graph.
    Hub nodes will act as strategic pivots.
    """
    print(f"Generating Scale-Free Graph (n={n}, m={m_edges})...")
    G = nx.barabasi_albert_graph(n, m_edges)
    # Add random weights
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = np.random.rand() + 0.1
    return nx.to_scipy_sparse_array(G, format='csr')

def run_benchmark():
    n = 100_000 
    m_param = 5 # Number of edges to attach from a new node to existing nodes
    source = 0
    
    adj_csr = generate_scale_free_graph(n, m_param)
    
    print("Starting SciPy Dijkstra (C++ Heap)...")
    start = time.time()
    d_scipy = dijkstra(adj_csr, indices=source, directed=False)
    print(f"SciPy Dijkstra: {time.time() - start:.4f}s")
    
    print("\nStarting Numba Dijkstra (Baseline Heap)...")
    numba_dijkstra(adj_csr.indices, adj_csr.indptr, adj_csr.data, source, 10) # Warm-up
    start = time.time()
    d_numba = numba_dijkstra(adj_csr.indices, adj_csr.indptr, adj_csr.data, source, n)
    numba_time = time.time() - start
    print(f"Numba Dijkstra: {numba_time:.4f}s")
    
    print("\nStarting ComesSolver (Frontier Partitioning)...")
    solver = ComesSolver(adj_csr)
    solver.shortest_path(source, target=10) # Trigger JIT compilation
    start = time.time()
    d_comes = solver.shortest_path(source)
    comes_time = time.time() - start
    print(f"ComesSolver: {comes_time:.4f}s")
    
    improvement_vs_numba = (numba_time - comes_time) / numba_time * 100
    print(f"\nAlgorithmic Dominance (Comes vs Numba-Heap): {improvement_vs_numba:.2f}%")
    
    # Validation of result integrity
    mask = np.isfinite(d_scipy)
    if np.allclose(d_scipy[mask], d_comes[mask], atol=1e-5):
        print("Validation PASSED.")
    else:
        print("Validation FAILED.")

if __name__ == "__main__":
    run_benchmark()