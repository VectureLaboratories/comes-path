import unittest
import numpy as np
from scipy.sparse import csr_matrix
from comes_path.core.solver import ComesSolver

class TestComesPath(unittest.TestCase):
    def test_simple_path(self):
        # 0 -> 1 (w=1.0), 1 -> 2 (w=2.0)
        row = np.array([0, 1])
        col = np.array([1, 2])
        data = np.array([1.0, 2.0])
        adj = csr_matrix((data, (row, col)), shape=(3, 3))
        
        solver = ComesSolver(adj)
        distances = solver.shortest_path(0)
        self.assertEqual(distances[2], 3.0)

    def test_disconnected_graph(self):
        # Topology: 0 -> 1 (w=1.0), Node 2 isolated.
        row = np.array([0])
        col = np.array([1])
        data = np.array([1.0])
        adj = csr_matrix((data, (row, col)), shape=(3, 3))
        
        solver = ComesSolver(adj)
        distances = solver.shortest_path(0)
        self.assertEqual(distances[1], 1.0)
        self.assertTrue(np.isinf(distances[2]))

    def test_target_search(self):
        row = np.array([0, 1])
        col = np.array([1, 2])
        data = np.array([1.0, 2.0])
        adj = csr_matrix((data, (row, col)), shape=(3, 3))
        
        solver = ComesSolver(adj)
        dist = solver.shortest_path(0, target=2)
        self.assertEqual(dist, 3.0)

    def test_sparse_fallback(self):
        # Sparse configuration: m < 2n.
        row = np.array([0, 1, 2, 3, 4])
        col = np.array([1, 2, 3, 4, 5])
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        adj = csr_matrix((data, (row, col)), shape=(10, 10))
        
        solver = ComesSolver(adj)
        self.assertTrue(solver.is_sparse_fallback)
        distances = solver.shortest_path(0)
        self.assertEqual(distances[5], 5.0)

if __name__ == '__main__':
    unittest.main()