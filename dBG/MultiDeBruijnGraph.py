import copy
from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class MultivariateDeBruijnGraph:
    def __init__(self,
                 k: int,
                 dimensions: int,
                 disc_functions: list):
        assert k >= 2, 'k-mer length must be more than 1!'
        assert dimensions >= 1, 'Number of dimensions must be a positive number!'

        self.k = k
        self.graph = nx.DiGraph()
        self.dimensions = dimensions

        assert len(disc_functions) == self.dimensions or len(disc_functions) == 1, 'disc_functions dimension mismatch'

        if len(disc_functions) != dimensions:
            self.discretizers = [copy.deepcopy(disc_functions[0]) for _ in range(self.dimensions)]
        else:
            self.discretizers = disc_functions

        assert len(self.discretizers) == self.dimensions, 'disc_functions dimension mismatch'

    def __str__(self):
        return f"{self.__class__.__name__} with " \
               f"{self.graph.number_of_nodes()} nodes " \
               f"and {self.graph.number_of_edges()} edges"

    def discretize_data(self, sequences):
        discrete_sequences = [None] * len(sequences)
        for i, dim in enumerate(sequences):
            values = np.array(dim).reshape(-1, 1)
            discrete_sequences[i] = self.discretizers[i].fit_transform(values).astype(int).flatten()
        return discrete_sequences

    def insert(self, sequences):
        assert self.dimensions == len(sequences), f'Dimension mismatch. Expected {self.dimensions} but got {len(sequences)}'
        sequences = [np.array(row) for row in sequences]
        disc_sequences = self.discretize_data(sequences)
        for i, (kmers, raw_kmers) in enumerate(zip(self.__multivariate_sliding_window(disc_sequences),
                                                   self.__multivariate_sliding_window(sequences))):
            updated_prefix_nodes = set()
            updated_suffix_nodes = set()

            for dim, (kmer, raw_kmer) in enumerate(zip(kmers, raw_kmers)):
                if kmer[0] is None:
                    continue

                prefix = (dim, kmer[:-1])
                if i == 0:
                    updated_prefix_nodes.add(prefix)

                suffix = (dim, kmer[1:])
                updated_suffix_nodes.add(suffix)
                self.__add_edge(prefix, suffix, {'kmer': np.array(kmer), 'type': 'ktuple'}, raw=raw_kmer)

            if i == 0:
                self.__add_hyper_edges(updated_prefix_nodes)
            self.__add_hyper_edges(updated_suffix_nodes)

        for u, v in self.graph.edges:
            self.graph[u][v].pop('raw', None)  # Removes 'weight' if it exists

    def __add_edge(self, source: tuple, target: tuple, edge_attributes: dict, raw: list = None):
        if not self.graph.has_edge(source, target):
            if 'weight' not in edge_attributes.keys():
                edge_attributes['weight'] = 1
            if edge_attributes['type'] == 'ktuple':
                edge_attributes['raw'] = [raw]
            else:
                edge_attributes['raw'] = []
            self.graph.add_edge(source, target, **edge_attributes)
        else:
            if edge_attributes['type'] == 'ktuple':
                self.graph[source][target]['raw'].append(raw)
            self.graph[source][target]['weight'] += 1

    def __add_hyper_edges(self, nodes):
        for node1, node2 in combinations(nodes, 2):
            self.__add_edge(node1, node2, {'type': 'hyper', 'kmer': np.zeros(self.k, dtype=int)})
            self.__add_edge(node2, node1, {'type': 'hyper', 'kmer': np.zeros(self.k, dtype=int)})

    def __multivariate_sliding_window(self, sequences):
        window_num = max(len(seq) for seq in sequences) - self.k + 1
        i = 0
        while True:
            current_windows = []
            for seq in sequences:
                if i + self.k <= len(seq):
                    current_windows.append(tuple(seq[i: i + self.k]))
                else:
                    current_windows.append((None,) * self.k)

            yield current_windows
            i += 1
            if i >= window_num:
                return

"""
if __name__ == "__main__":
    test = MultivariateDeBruijnGraph(k=3,
                                     dimensions=3,
                                     disc_functions=[KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')])
    test.insert([[1.1, 2.3, 1.1, 3.3, 1.1, 3.3, 1.1, 3.3],
                 [112.33, 123.11, 223.3, 211.1223, 334.31, 223.23],
                 [52.213, 123.32, 424.324, 122.23]])

    print(test)
    print("Nodes:")
    print(sorted(set(test.graph.nodes), key=lambda x: str(x)))  # Sort nodes as strings to avoid type errors
    print("Edges:")

    for edge in sorted(test.graph.edges(data=True), key=lambda x: (str(x[0]), str(x[1]))):
        print(edge[0], "->", edge[1], edge[2])
"""