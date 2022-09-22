import os
import random
from collections import deque

import numpy as np
import networkx as nx


class GraphNodeSampler:
    """Randomly samples an existing edge in a specified graph, extracts the connecting nodes, then expands these
    nodes around a certain neighborhood using the original graph.
    """

    def __init__(self, full_graph, node_list=None, num_edge_types=2):
        """Initializes the sampler.

        Args:
            full_graph: The graph used for generating subgraphide_diedgeshs
            node_list: (Optional) A list of start node IDs to sample subgraphs for. Default is None, which will use
                all the nodes in the graph.
            num_edge_types: (Optional) The total number of edge types or IDs. Default is 3.
        """
        self._full_graph = full_graph
        self._node_subset = list(self._full_graph.nodes) if node_list is None else node_list
        self._num_edge_types = num_edge_types

    def get_config(self):
        return {
            "full_graph": self._full_graph,
            "node_list": self._node_subset,
            "num_edge_types": self._num_edge_types
        }

    def _expand_subgraph(self, node_list, max_depth, max_neighbours, max_nodes, ignore_warnings=True):
        """Expands the subgraph around the 2 nodes of an edge, subject to constraints
        """
        assert max_nodes >= len(node_list)

        queue = deque([(node, 0) for node in node_list])
        node_set = set(node_list)
        filter_edges = []

        if max_depth > 0:
            while len(queue) > 0:
                node, depth = queue.popleft()
                all_neighbours = set(self._full_graph.neighbors(node))
                extra_neighbours = all_neighbours - node_set
                existing_neighbours = all_neighbours - extra_neighbours

                max_existing_neighbours = min(max_neighbours, len(existing_neighbours))
                if max_existing_neighbours < len(existing_neighbours):
                    filter_edges.extend([(node, nb) for nb in
                                         random.sample(existing_neighbours, len(existing_neighbours) - max_existing_neighbours)])

                max_extra_neighbours = min(max_neighbours - max_existing_neighbours, len(extra_neighbours))
                if max_extra_neighbours < len(extra_neighbours):
                    sample_extra_neighbours = set(random.sample(extra_neighbours, max_extra_neighbours))
                    filter_edges.extend([(node, nb) for nb in (extra_neighbours - sample_extra_neighbours)])
                else:
                    sample_extra_neighbours = extra_neighbours

                if depth < max_depth:
                    if (len(node_set) + len(sample_extra_neighbours)) > max_nodes:
                        if ignore_warnings:
                            sample_extra_neighbours = random.sample(sample_extra_neighbours, max_nodes - len(node_set))
                        else:
                            raise ValueError("Increase max nodes. Current: %d" % max_nodes)
                    queue.extend([(nb, depth + 1) for nb in sample_extra_neighbours])
                    node_set.update(sample_extra_neighbours)

        induced_nodes = nx.filters.show_nodes(self._full_graph.nbunch_iter(node_set))
        # if self._full_graph.is_directed():
        #    hide_edges = nx.filters.hide_diedges(filter_edges)
        # else:
        hide_edges = nx.filters.hide_edges(filter_edges)

        return nx.graphviews.subgraph_view(self._full_graph, induced_nodes, hide_edges)

    def _sample_node(self, max_depth, max_neighbours, max_nodes):
        node = random.choice(self._node_subset)

        subgraph = self._expand_subgraph([node], max_depth, max_neighbours, max_nodes)
        return subgraph, node

    def _convert_to_np(self, subgraph, start_node, max_nodes, max_neighbours):
        num_features = len(subgraph.nodes[start_node]["feat"])
        node_features = np.zeros((max_nodes, num_features), dtype=np.float32)
        node_adj = np.full((max_nodes, self._num_edge_types + 1, max_neighbours), -1, dtype=np.int32)
        id_remap = {node: i for i, node in enumerate(subgraph.nodes)}

        for i, node in enumerate(subgraph.nodes):
            # Add features
            node_features[i] = subgraph.nodes[node]["feat"]

            # Add neighbour indices
            cnt = [0, 0]
            for nb in nx.neighbors(subgraph, node):
                #print("i: %d, cnt: %d" % (nb, cnt))
                edge_type = subgraph[node][nb]["type_id"]
                node_adj[i][edge_type][cnt[edge_type]] = id_remap[nb]
                cnt[edge_type] = cnt[edge_type] + 1

            # Add self-loop
            node_adj[i][self._num_edge_types][0] = i

        return node_adj, node_features, id_remap[start_node], id_remap

    def to_numpy(self, max_depth, max_neighbours, max_nodes, ignore_warnings=True):
        subgraph = self._expand_subgraph(self._node_subset,
                                         max_depth, max_neighbours, max_nodes,
                                         ignore_warnings=ignore_warnings)
        dummy = self._node_subset[len(self._node_subset) // 2]
        return self._convert_to_np(subgraph, dummy, max_nodes, max_neighbours)

    def sample(self, max_depth, max_neighbours, max_nodes):
        """Randomly samples an edge and associated subgraph and transforms these to
        node adjacency tensors and feature tensors, subject to various constraints.

        If the number of nodes exceed the constraints, nodes will be randomly sampled
        to meet that constraint.

        Args:
            max_depth: Maximum number of hops around the edge nodes to expand when building a subgraph
            max_neighbours: Maximum number of direct neighbours each node can have
            max_nodes: Maximum number of nodes in a subgraph

        Returns:
            A tuple of numpy tensors. Index 0 is the node  adjacency tensor,
            index 1 is the features tensor, index 2 is the 1st node in the sampled edge, index 3 is the 2nd node,
            index 4 is a dict mapping the original node ID of the graph to a node index in the adjacency / features
            tensors.
        """

        subgraph, node = self._sample_node(max_depth, max_neighbours, max_nodes)
        return self._convert_to_np(subgraph, node, max_nodes, max_neighbours)

    def iter_nodes(self, max_depth, max_neighbours, max_nodes):
        """Iterates through all start nodes and generates subgraphs

        Args:
            max_depth: Maximum number of hops around the edge nodes to expand when building a subgraph
            max_neighbours: Maximum number of direct neighbours each node can have
            max_nodes: Maximum number of nodes in a subgraph

        Returns:
            An iterator containing tuples of numpy tensors. Index 0 is the node  adjacency tensor,
            index 1 is the features tensor, index 2 is the 1st node in the sampled edge, index 3 is the 2nd node,
            index 4 is a dict mapping the original node ID of the graph to a node index in the adjacency / features
            tensors.
        """
        for node in self._node_subset:
            subgraph = self._expand_subgraph([node], max_depth, max_neighbours, max_nodes)
            yield self._convert_to_np(subgraph, node, max_nodes, max_neighbours)
