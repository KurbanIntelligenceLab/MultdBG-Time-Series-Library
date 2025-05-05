import random

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance
from torch_geometric.utils import from_networkx


class dBGMasker:
    def __init__(self, dBG, augment_prob=0.1, augment=True):
        self.dBG = dBG
        self.augment_prob = augment_prob
        self.augment = augment
        str_nodes = list(self.dBG.graph.nodes)
        self.str_to_idx = {node: idx for idx, node in enumerate(str_nodes)}
        self.idx_to_str = {idx: node for node, idx in self.str_to_idx.items()}
        self.G_relabel = nx.relabel_nodes(self.dBG.graph, self.str_to_idx)
        self.n_nodes = len(self.G_relabel)

        inv_weight_graph = self.G_relabel.copy()
        all_weights = [d['weight'] for _, _, d in inv_weight_graph.edges(data=True)]
        w_max = max(all_weights)

        for u, v, d in inv_weight_graph.edges(data=True):
            d['weight'] = (w_max - d['weight']) + 1e-6

        self.w_max = max(inv_weight_graph)
        self.adj = nx.to_scipy_sparse_array(
            inv_weight_graph,
            nodelist=range(self.n_nodes),
            weight='weight',
            format='csr'
        )

        self.similar_node_cache = {}

    def generate_mask(self, values):
        k_1 = self.dBG.k - 1
        selected_nodes = set()

        for dim in range(values.shape[0]):
            for i in range(values.shape[1] - k_1 + 1):
                mer = tuple(values[dim, i:i + k_1])
                query = (dim, mer)

                if query in self.str_to_idx:
                    selected_nodes.add(query)
                elif query in self.similar_node_cache:
                    selected_nodes.add(self.similar_node_cache[query])
                else:
                    min_dist = float('inf')
                    best_match = None
                    for node in self.dBG.graph.nodes:
                        if query[0] == node[0]:
                            dist = distance.cityblock(mer, node[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_match = node
                    if best_match is not None:
                        self.similar_node_cache[query] = best_match
                        selected_nodes.add(best_match)

        # Make a copy to modify during iteration
        selected_nodes_augmented = set(selected_nodes)

        if self.augment and self.augment_prob > 0:
            for node in list(selected_nodes):
                if random.random() < self.augment_prob:
                    if node in self.similar_node_cache:
                        best_match = self.similar_node_cache[node]
                    else:
                        min_dist = float('inf')
                        best_match = None
                        for candidate in self.dBG.graph.nodes:
                            if candidate not in selected_nodes_augmented and candidate[0] == node[0]:
                                dist = distance.cityblock(node[1], candidate[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = candidate
                        if best_match is not None:
                            self.similar_node_cache[node] = best_match

                    if best_match is not None:
                        selected_nodes_augmented.discard(node)
                        selected_nodes_augmented.add(best_match)

        selected_idxs = [self.str_to_idx[n] for n in selected_nodes_augmented if n in self.str_to_idx]
        soft_mask = self.__gaussian_mask(selected_idxs)

        mask = np.zeros(self.n_nodes, dtype=np.float32)
        for idx, val in soft_mask.items():
            mask[idx] = val

        return mask

    def __gaussian_mask(self, idxs, sigma=100, limit=0.05):
        dist_matrix = dijkstra(csgraph=self.adj,
                               directed=True,
                               unweighted=False,
                               indices=idxs,
                               min_only=True)
        mask = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
        return {i: float(mask[i]) for i in range(self.n_nodes)}





class dBGNeighborLoader:
    def __init__(self, dBG, num_neighbors: list):
        self.dBG = dBG
        self.num_neighbors = num_neighbors

        self.layer_mask = [set() for _ in range(self.dBG.dimensions)]
        self.query_map = dict()

        for node in self.dBG.graph.nodes:
            self.layer_mask[node[0]].add(node)

    def sample(self, values):
        visited_nodes = set()
        for dim, kmer in enumerate(values):
            query = (dim, tuple(kmer.tolist()))
            if query in self.dBG.graph:
                seed = query
            elif query in self.query_map.keys():
                seed = self.query_map[query]
            else:
                min_dist = float('inf')
                min_node = None
                for node in self.dBG.graph.nodes:
                    manhattan_dist = distance.cityblock(query[1], node[1])
                    if manhattan_dist < min_dist:
                        min_node = node
                        min_dist = manhattan_dist
                self.query_map[query] = min_node
                seed = min_node
            visited_nodes.update(self.sample_layer(seed, dim))
        subgraph = self.dBG.graph.subgraph(visited_nodes)
        subgraph = from_networkx(subgraph)
        return subgraph

    def sample_layer(self, nodes, layer):
        target_size = sum(self.num_neighbors) + 1
        sampled_nodes = {nodes}
        current_nodes = {nodes}
        remaining_needed = target_size - len(sampled_nodes)
        depth = 0

        while remaining_needed > 0 and current_nodes:
            next_nodes = set()
            num_to_sample = min(self.num_neighbors[min(depth, len(self.num_neighbors) - 1)], remaining_needed)

            potential_neighbors = set()
            for node in current_nodes:
                neighbors = list(self.dBG.graph.neighbors(node))
                neighbors = [n for n in neighbors if n in self.layer_mask[layer] and n not in sampled_nodes]
                potential_neighbors.update(neighbors)

            if potential_neighbors:
                sampled = random.sample(list(potential_neighbors), min(num_to_sample, len(potential_neighbors)))
                next_nodes.update(sampled)

            sampled_nodes.update(next_nodes)
            current_nodes = next_nodes
            remaining_needed = target_size - len(sampled_nodes)
            depth += 1

        return sampled_nodes
