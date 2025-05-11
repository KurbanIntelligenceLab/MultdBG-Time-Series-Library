import copy
from itertools import combinations

import networkx as nx
import numpy as np


class MultivariateDeBruijnGraph:
    def __init__(self,
                 k: int,
                 dimensions: int,
                 disc_functions: list):
        assert k >= 2, 'k-mer length must be > 1'
        assert dimensions >= 1, 'Number of dimensions must be positive'
        self.k = k
        self.dimensions = dimensions
        self.graph = nx.DiGraph()

        # build one discretizer per dimension
        assert len(disc_functions) in (1, dimensions), 'disc_functions dimension mismatch'
        if len(disc_functions) == 1:
            self.discretizers = [copy.deepcopy(disc_functions[0]) for _ in range(dimensions)]
        else:
            self.discretizers = disc_functions

    def __str__(self):
        return f"{self.__class__.__name__}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"

    def discretize_data(self, sequences):
        discrete = []
        for i, seq in enumerate(sequences):
            vals = np.array(seq).reshape(-1, 1)
            disc = self.discretizers[i].fit_transform(vals).astype(int).flatten()
            discrete.append(disc)
        return discrete

    def insert(self, sequences):
        assert len(sequences) == self.dimensions
        seqs = [np.array(s) for s in sequences]
        disc_seqs = self.discretize_data(seqs)

        for i, (kmers, raw_kmers) in enumerate(zip(
                self.__multivariate_sliding_window(disc_seqs),
                self.__multivariate_sliding_window(seqs))):

            updated_prefix = set()
            updated_suffix = set()

            for dim, (kmer, raw) in enumerate(zip(kmers, raw_kmers)):
                if kmer[0] is None:
                    continue

                prefix = (dim, kmer[:-1])
                suffix = (dim, kmer[1:])

                self.__add_node_feature(prefix, raw[:-1])
                self.__add_node_feature(suffix, raw[1:])

                updated_prefix.add(prefix)
                updated_suffix.add(suffix)

                self.__add_edge(prefix, suffix, {'kmer': np.array(kmer), 'type': 'ktuple'})

            if i == 0:
                self.__add_hyper_edges(updated_prefix)
            self.__add_hyper_edges(updated_suffix)

        node_features = [feat for (_, feat) in self.graph.nodes(data="features")]

        # Removed cuz dynamic features are not supported by pytorch geometric
        for n, data in self.graph.nodes(data=True):
            data.pop('feat', None)

        return node_features

    def __add_node_feature(self, node, raw_k_1):
        arr = np.array(raw_k_1).reshape(1, -1)
        if not self.graph.has_node(node):
            self.graph.add_node(node, features=arr)
        else:
            feats = self.graph.nodes[node].get('features', [])
            feats = np.vstack([feats, arr])
            self.graph.nodes[node]['features'] = feats

    def __add_edge(self, source, target, attrs):
        w = attrs.get('weight', 0) + 1
        attrs['weight'] = w
        if not self.graph.has_edge(source, target):
            self.graph.add_edge(source, target, **attrs)
        else:
            # just bump the weight; other attrs (like kmer/type) remain
            self.graph[source][target]['weight'] = w

    def __add_hyper_edges(self, nodes):
        for n1, n2 in combinations(nodes, 2):
            # hyper‐edges still only carry type/kmer, no raw
            self.__add_edge(n1, n2, {'type': 'hyper', 'kmer': np.zeros(self.k, int)})
            self.__add_edge(n2, n1, {'type': 'hyper', 'kmer': np.zeros(self.k, int)})

    def __multivariate_sliding_window(self, sequences):
        max_len = max(len(s) for s in sequences)
        window_count = max_len - self.k + 1
        for i in range(window_count):
            current = []
            for s in sequences:
                if i + self.k <= len(s):
                    current.append(tuple(s[i:i + self.k]))
                else:
                    current.append((None,) * self.k)
            yield current


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
