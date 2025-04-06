import networkx as nx


class DeBruijnGraph:
    def __init__(self, k: int):
        if k < 2:
            raise ValueError('k-mer length must be more than 1!')

        self.k = k
        self.alphabet = set()
        self.graph = nx.DiGraph()

    def __str__(self):
        return f"{self.__class__.__name__} with " \
               f"{self.graph.number_of_nodes()} nodes " \
               f"and {self.graph.number_of_edges()} edges"

    def insert(self, new_sequences: list):
        for seq in new_sequences:
            self.alphabet.update(seq)
            for i in range(len(seq) - self.k + 1):
                kmer = tuple(seq[i : i + self.k])
                prefix, suffix = kmer[:-1], kmer[1:]
                edge_attributes = {
                    'weight': 1,
                    'kmer': kmer,
                }
                if not self.graph.has_edge(prefix, suffix):
                    self.graph.add_edge(prefix, suffix, **edge_attributes)
                else:
                    self.graph[prefix][suffix]["weight"] += 1


if __name__ == '__main__':
    test = DeBruijnGraph(k=3)
    test.insert(['ABAB'])
    print(test)
    print(test.graph.nodes)