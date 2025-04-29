import logging
import math
import queue

import networkx as nx
import torch

from hesp.hierarchy.hierarchy_helpers import json2rels, hierarchy_pos
from hesp.hierarchy.node import Node
from hesp.visualize.visualize_helpers import colour_nodes

# models/hierarchy/tree.py

import os
import math
import queue
import torch
import numpy as np
import networkx as nx
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tree:
    """ Tree object containing the target classes and their hierarchical relationships.
    Args:
        i2c: dict linking label indices to concept names. 
        json: json representation of hierarchical relationships. Empty dict assumes no hierarchical relationships.
    """

    def __init__(self, i2c, json_hierarchy=None):
        self.i2c = i2c
        # Number of leaf classes
        self.K = max(i2c.keys()) + 1  

        # Build or default the JSON hierarchy
        if not json_hierarchy:
            # flat: all classes under 'root'
            json_hierarchy = {"root": {c: {} for _, c in i2c.items()}}
        self.json = json_hierarchy

        self.root = next(iter(self.json))
        self.target_classes = np.array(list(i2c.keys()), dtype=np.int64)
        self.c2i = {c: i for i, c in i2c.items()}

        # Build nodes and index maps
        self.nodes = self._init_nodes()
        self.train_classes = list(self.i2n.keys())
        self.M = self._compute_M()

        # Build hierarchy and sibling matrices
        self._init_matrices()

        # Color nodes (for visualization), init graph & metric families
        self.nodes = colour_nodes(self.nodes, self.root)
        self._init_graph()
        self._init_metric_families()

    def _compute_M(self):
        # M = number of leaf classes (K) + number of internal (ancestor) nodes, excluding the root
        ancestor_nodes = [n for n in self.nodes if n not in self.c2i]
        return self.K + len(ancestor_nodes) - 1

    def _init_nodes(self):
        """Initialize Node objects and index mappings."""
        idx_counter = self.K
        self.i2n = self.i2c.copy()  # idx→name
        q = queue.Queue()
        nodes = {}

        # Root
        root_node = Node(
            name=self.root,
            parent=None,
            children=list(self.json[self.root].keys()),
            ancestors=[],
            siblings=[],
            depth=0,
            sub_hierarchy=self.json[self.root],
            idx=-1,
        )
        nodes[self.root] = root_node
        q.put(root_node)

        # BFS over hierarchy
        while not q.empty():
            parent = q.get()
            for child_name in parent.children:
                if child_name in self.c2i:
                    idx = self.c2i[child_name]
                else:
                    idx = idx_counter
                    idx_counter += 1

                child_node = Node(
                    name=child_name,
                    parent=parent.name,
                    children=list(parent.sub_hierarchy[child_name].keys()),
                    ancestors=parent.ancestors + [parent.name],
                    siblings=parent.children,
                    depth=parent.depth + 1,
                    sub_hierarchy=parent.sub_hierarchy[child_name],
                    idx=idx,
                )
                if idx not in self.i2n:
                    self.i2n[idx] = child_name
                nodes[child_name] = child_node
                q.put(child_node)

        # Build reverse map: name→idx
        self.n2i = {name: idx for idx, name in self.i2n.items()}
        return nodes

    def _init_matrices(self):
        """
        Build:
         - hmat [M×M]: hierarchy connectivity (ancestors + self)
         - sibmat [M×M]: sibling connections
        Then trim both to [K×K] so they match your logits dimension.
        """
        M = self.M
        K = self.K
        hmat = np.zeros((M, M), dtype=np.float32)
        sibmat = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            if i in self.i2n:
                name = self.i2n[i]
                node = self.nodes[name]

                # siblings
                sib_indices = [ self.n2i[sib] for sib in node.siblings ]
                sibmat[i, sib_indices] = 1.0

                # hierarchy: self + ancestors (excluding root)
                hier_indices = [i] + [ self.n2i[a] for a in node.ancestors if a != self.root ]
                hmat[i, hier_indices] = 1.0

        # Convert to torch and trim to leaf classes only
        # self.hmat = torch.from_numpy(hmat[:K, :K])
        # self.sibmat = torch.from_numpy(sibmat[:K, :K])
        self.hmat = torch.from_numpy(hmat)
        self.sibmat = torch.from_numpy(sibmat)

    def _init_graph(self):
        """Initializes networkx graph, used for visualization."""
        rels = json2rels(self.json)
        self.G = nx.Graph()
        self.G.add_edges_from(rels)
        pos = hierarchy_pos(self.G, self.root, width=2 * math.pi)
        self.pos = {u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()}

    @property
    def levels(self):
        """Returns nodes grouped by depth (excluding root)."""
        max_depth = max(n.depth for n in self.nodes.values())
        return [[n for n in self.nodes.values() if n.depth == d] for d in range(1, max_depth + 1)]

    def get_sibling_nodes(self, node_name):
        return [ self.nodes[sib] for sib in self.nodes[node_name].siblings ]

    def get_parent_node(self, node_name):
        return self.nodes[self.nodes[node_name].parent]

    def is_hyponym_of(self, key, target):
        parent = self.nodes[key].parent
        if parent is None:
            return False
        if parent == target:
            return True
        return self.is_hyponym_of(parent, target)

    def metric_family(self, concept_idx):
        name = self.i2c[concept_idx]
        node = self.nodes[name]
        siblings = [i for i in self.target_classes if self.is_hyponym_of(self.i2c[i], node.parent)]
        cousins = [i for i in self.target_classes if self.is_hyponym_of(self.i2c[i], self.nodes[node.parent].parent)]
        return siblings, cousins

    def _init_metric_families(self):
        """Attach metric_siblings and metric_cousins to each leaf node."""
        for i in self.target_classes:
            name = self.i2c[i]
            node = self.nodes[name]
            sibs, cous = self.metric_family(i)
            if node.parent != self.root:
                node.metric_siblings = sibs
                # if parent of parent is root, cousins = siblings
                node.metric_cousins = cous if self.nodes[node.parent].parent != self.root else sibs
            else:
                node.metric_siblings = [i]
                node.metric_cousins = [i]


