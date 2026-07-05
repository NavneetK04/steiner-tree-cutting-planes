# core.py -- shared helpers for the Steiner cut code.
# Two things live here: a reader for the SteinLib .stp files, and an Instance class
# that turns the edge list into the directed-arc form the models need.

import numpy as np


def read_stp(filename):
    """Read a SteinLib .stp file. Returns the edge list [(u, v, cost), ...] and the
       list of terminal nodes."""
    edges, terminals = [], []
    reading_graph = reading_terminals = False
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "SECTION Graph":
                reading_graph = True; continue
            if line == "SECTION Terminals":
                reading_graph = False; reading_terminals = True; continue
            if line == "END":
                reading_graph = reading_terminals = False; continue
            if reading_graph and line.startswith("E "):
                p = line.split()
                edges.append((int(p[1]), int(p[2]), float(p[3])))
            if reading_terminals and line.startswith("T "):
                terminals.append(int(line.split()[1]))
    return edges, terminals


class Instance:
    """Holds the graph in the directed-arc form the models use.
       Each undirected edge e becomes two arcs: a forward arc (id e) and a reverse arc
       (id e + a). The first terminal is the root; the rest are the commodity sinks
       (one commodity per terminal, each sent from the root to its terminal)."""
    def __init__(self, edges, terminals, name=""):
        self.name = name
        # relabel the nodes to 1..n
        nodes = sorted({u for u, v, c in edges} | {v for u, v, c in edges}
                       | set(terminals))
        self.node_map = {nd: i + 1 for i, nd in enumerate(nodes)}
        self.n = len(nodes)

        sf, tf, c1 = [], [], []
        for u, v, c in edges:
            sf.append(self.node_map[u]); tf.append(self.node_map[v]); c1.append(c)
        m = len(edges)
        # forward arcs first, then the reverse arcs
        self.sf = np.array(sf + tf, dtype=int)          # start node of each arc (length 2a)
        self.tf = np.array(tf + sf, dtype=int)          # end node of each arc
        self.c1 = np.array(c1, dtype=float)             # edge cost (length a)
        self.a = m                                       # number of undirected edges
        self.A = 2 * m                                   # number of directed arcs

        term = [self.node_map[t] for t in terminals]
        self.root = term[0]                              # first terminal is the root
        self.commodity_term = term[1:]                   # one commodity per remaining terminal
        self.K = len(self.commodity_term)
        self.terminals = term

        # for each node, the arcs going in and the arcs going out (arc ids are 0-based)
        self.in_arcs  = [[] for _ in range(self.n + 1)]
        self.out_arcs = [[] for _ in range(self.n + 1)]
        for arc in range(self.A):
            self.out_arcs[self.sf[arc]].append(arc)
            self.in_arcs[self.tf[arc]].append(arc)

    def edge_of_arc(self, arc):
        """Given an arc id (0..2a-1), return the undirected edge id (0..a-1) it belongs to."""
        return arc % self.a