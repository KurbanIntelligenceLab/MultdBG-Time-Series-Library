import random

from scipy.spatial import distance
from torch_geometric.utils import from_networkx


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
