import numpy as np
from scipy.sparse import csr_matrix
from comes_path.core.solver import ComesSolver
from comes_path.core.partitioning import partition_graph

def technical_verification():
    print("[VECTURE] Initializing Technical Verification Loop...")
    
    # Controlled 5-node test topology
    # 0 --(1.0)--> 1 --(2.0)--> 2
    # 0 --(5.0)--> 3 --(1.0)--> 2
    # 2 --(1.0)--> 4
    rows = np.array([0, 1, 0, 3, 2])
    cols = np.array([1, 2, 3, 2, 4])
    weights = np.array([1.0, 2.0, 5.0, 1.0, 1.0])
    
    adj = csr_matrix((weights, (rows, cols)), shape=(5, 5))
    # Symmetrization
    adj = adj + adj.T
    
    print("[VECTURE] Graph Ingested. CSR Metadata:")
    print(f" - Nodes: {adj.shape[0]}")
    print(f" - Edges: {adj.nnz}")
    
    # Step 1: Calibration (Partitioning)
    params = partition_graph(adj)
    print(f"[VECTURE] Partitioning Parameters: {params}")
    
    # Step 2: Solver Initialization and Execution
    solver = ComesSolver(adj)
    source = 0
    target = 4
    
    print(f"[VECTURE] Executing Shortest Path Loop: {source} -> {target}")
    distances = solver.shortest_path(source)
    
    # Step 3: Result Integrity Verification
    # Expected path: 0 -> 1 -> 2 -> 4 = 1.0 + 2.0 + 1.0 = 4.0
    # Alternative path: 0 -> 3 -> 2 -> 4 = 5.0 + 1.0 + 1.0 = 7.0
    expected_dist = 4.0
    actual_dist = distances[target]
    
    print(f"[VECTURE] Result: {actual_dist}")
    
    if np.isclose(actual_dist, expected_dist):
        print("[VECTURE] Verification: SUCCESS. Loop closed without anomalies.")
        return True
    else:
        print(f"[VECTURE] Verification: FAILURE. Expected {expected_dist}, got {actual_dist}")
        return False

if __name__ == "__main__":
    success = technical_verification()
    if not success:
        exit(1)