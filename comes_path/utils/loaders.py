"""
Vecture Laboratories // Data Ingestion Protocols

Operational Directive:
Translate raw topological data (OSM, GraphML, ADJ) into CSR format.
"""

import numpy as np
from scipy.sparse import csr_matrix
import xml.etree.ElementTree as ET

def load_adj(file_path):
    """Ingest adjacency lists into CSR format."""
    data = np.loadtxt(file_path)
    u = data[:, 0].astype(np.int64)
    v = data[:, 1].astype(np.int64)
    weights = data[:, 2]
    n = max(u.max(), v.max()) + 1
    return csr_matrix((weights, (u, v)), shape=(n, n))

def load_graphml(file_path):
    """Parse GraphML XML structures into CSR format."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    node_map = {}
    current_idx = 0
    edges = []
    for edge in root.findall('.//g:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        if source not in node_map:
            node_map[source] = current_idx
            current_idx += 1
        if target not in node_map:
            node_map[target] = current_idx
            current_idx += 1
        weight_data = edge.find('.//g:data', ns)
        weight = float(weight_data.text) if weight_data is not None else 1.0
        edges.append((node_map[source], node_map[target], weight))
    edges = np.array(edges)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    w = edges[:, 2]
    return csr_matrix((w, (u, v)), shape=(current_idx, current_idx))

def load_osm(file_path):
    """
    Extract topological data from OpenStreetMap (XML).
    Converts spatial coordinates into Euclidean edge weights.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    nodes = {}
    for node in root.findall('node'):
        nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
    edges = []
    node_id_map = {old_id: i for i, old_id in enumerate(nodes.keys())}
    for way in root.findall('way'):
        is_highway = any(tag.get('k') == 'highway' for tag in way.findall('tag'))
        if not is_highway: continue
        way_nodes = [nd.get('ref') for nd in way.findall('nd')]
        for i in range(len(way_nodes) - 1):
            u_id, v_id = way_nodes[i], way_nodes[i+1]
            if u_id in nodes and v_id in nodes:
                u_lat, u_lon = nodes[u_id]
                v_lat, v_lon = nodes[v_id]
                dist = np.sqrt((u_lat - v_lat)**2 + (u_lon - v_lon)**2)
                edges.append((node_id_map[u_id], node_id_map[v_id], dist))
    if not edges: return csr_matrix((0,0))
    edges = np.array(edges)
    return csr_matrix((edges[:, 2], (edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64))), shape=(len(nodes), len(nodes)))
