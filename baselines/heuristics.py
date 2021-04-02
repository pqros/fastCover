import logging
import time
import igraph
import heapq
from collections import deque
import numpy as np
import torch


class Node:
    def __init__(self, id, value):
        self.id = id
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        # we store the negative value to use max heap.
        return f"{self.id}: {-self.value}"


def bfs(graph: igraph.Graph, d: int):
    if d <= 1:
        return graph.copy()

    n = graph.vcount()
    es = []

    for v in range(n):
        layers = [0] * n
        visited, queue = set([v]), deque([v])
        while queue:
            vertex = queue.popleft()
            for neighbor in graph.successors(vertex):
                if neighbor not in visited and layers[vertex] < d:
                    visited.add(neighbor) 
                    queue.append(neighbor)
                    layers[neighbor] = layers[vertex] + 1
        visited.remove(v)
        es.extend([(v, u) for u in visited])
    
    extended_graph = igraph.Graph(n=n, directed=True)
    extended_graph.add_edges(es)
    return extended_graph


def extend_closure(graph: igraph.Graph, debug=False):
    graph_temp = graph.copy()
    n = graph.vcount()
    new_edges = []
    for v in range(n):        
        v_next = graph.successors(v)
        
        v_next_next = set()
        for u in v_next:
            v_next_next = v_next_next.union(set(graph.successors(u)))
        
        v_next_next = v_next_next.difference(v_next)
        v_next_next = v_next_next.difference([v])
        new_edges.extend([(v, u) for u in v_next_next])
    graph_temp.add_edges(new_edges)
    return graph_temp


def d_closure(graph: igraph.Graph, d: int, debug=False):
    graph_temp = graph.copy()
    for i in range(1, d):
        if debug:
            print(f"Extending {i}/{d - 1}...", flush=True)
        graph_temp = extend_closure(graph_temp, debug=debug)
    return graph_temp


def d_greedy(graph: igraph.Graph, k: int, d: int, debug=False):
    """find k-max d-hop cover with greedy

    Args:
        graph (igraph.Graph): graph
        k (int): the number of seeds k
        debug (bool): debug mode

    Returns:
        seeds, covernum: selected seeds, and the seed count
    """
    seeds = []
    # closed_graph = d_closure(graph, d, debug)
    closed_graph = bfs(graph, d)

    nodes_num = closed_graph.vcount()
    covered = [False] * nodes_num
    cover_num = 0

    inf_list = [deg + 1 for deg in closed_graph.outdegree()]

    node_queue = [Node(i, -inf_list[i]) for i in range(nodes_num)]
    heapq.heapify(node_queue)
    i = 0

    while i < k and cover_num < nodes_num:  # while there's still free point or unused budget

        # Find the node with max marginal utility
        max_inf_node = heapq.heappop(node_queue)
        if inf_list[max_inf_node.id] != - max_inf_node.value:
            max_inf_node.value = -inf_list[max_inf_node.id]
            heapq.heappush(node_queue, max_inf_node)
            continue

        i += 1
        seeds.append(max_inf_node.id)
        if not covered[max_inf_node.id]:  # Update predecessors
            covered[max_inf_node.id] = True  # 1. mark max_node as covered
            cover_num += 1
            inf_list[max_inf_node.id] -= 1
            # 2. all the preds have influence -1
            for predecessor in closed_graph.predecessors(max_inf_node.id):
                inf_list[predecessor] -= 1

        # Update successors
        for successor in closed_graph.successors(max_inf_node.id):
            if not covered[successor]:
                # 1. mark all the successors as covered
                covered[successor] = True
                cover_num += 1
                # 2. all the successors have influence -1 (since there is no unitility to cover themselves)
                inf_list[successor] -= 1
                # 3. all the (predecessors of successors) have influence -1
                for predecessor in closed_graph.predecessors(successor):
                    inf_list[predecessor] -= 1

        if debug:
            print(
                f"Round {i}: {max_inf_node.id} is selected. {cover_num} nodes are covered.")
                
    return seeds, cover_num


def neighbors(graph: igraph.Graph, v: int, d: int):
    if d == 1:
        return graph.successors(v)

    n = graph.vcount()
    layers = [0] * n
    visited, queue = set([v]), deque([v])
    while queue:
        vertex = queue.popleft()
        for neighbor in graph.successors(vertex):
            if neighbor not in visited and layers[vertex] < d:
                visited.add(neighbor) 
                queue.append(neighbor)
                layers[neighbor] = layers[vertex] + 1
    
    visited = list(visited)
    return visited


def uncovered_neighbors(graph, k):
    pass


def remove_isolated(graph, k):  # algorithm 2
    r = set()  # vertices to be removed
    n = graph.vcount()
    f = [True] * n

    for v in range(n):
        if f[n]:
            neighbor_k, neighbor_k_plus = neighbors(v, k), neighbors(v, k+1)
            if len(neighbor_k) == len(neighbor_k_plus):
                r = r.union(set(neighbor_k))
            for u in neighbor_k:
                f[u] = False

    reduced_graph = graph.delete_vertices(r)
    return reduced_graph


def heu1(graph, k, d):
    "Naive Solution Construction"
    n = graph.vcount()
    is_covered = [False] * n
    nodes = sorted([Node(i, -deg) for i, deg in zip(range(n), graph.outdegree())])
    seeds = set()

    for node in nodes:
        v = node.id
        if len(seeds) >= k:
            break
        
        if not is_covered[v]:
            seeds.add(v)
            for u in neighbors(graph, v, d):
                is_covered[u] = True

    seeds = list(seeds)
    n_covered = np.sum(is_covered)
    return list(seeds), n_covered


def heu2(graph, k, theta, t_limit):
    "Advanced Solution Construciton"
    n = graph.vcount()
    is_covered = [False] * n
    nodes = [Node(i, -deg) for i, deg in zip(range(n), graph.outdegree())]
    d = set()

    t_start = time.time()
    for node in nodes:
        if time.time() - t_start > t_limit or all(is_covered):
            break

        v = node.id
        if not is_covered[v] or len(uncovered_neighbors(graph, k)) > theta:
            # theta are general 0 to 4
            d.add(v)
            for u in neighbors(k, v):
                is_covered[u] = True

    if not all(is_covered):
        graph_temp = graph.copy()
        uncovered_nodes = [v for v in range(n)if not is_covered[v]]
        graph_temp.delete_vertices(uncovered_nodes)
        d2 = heu1(graph_temp, k)
        d = d.union(d2)

    return d


def get_influence(graph, seeds):
    if torch.is_tensor(seeds):
        seeds = seeds.int().tolist()

    covered = set()
    for seed in seeds:
        covered.add(int(seed))
        for u in graph.successors(seed):  # Add all the covered seeds
            covered.add(u)
    return len(covered)


def get_influence_d(graph, seeds, d):
    if d == 1:
        return get_influence(graph, seeds)
    
    if torch.is_tensor(seeds):
        seeds = seeds.int().tolist()
    
    covered = set(seeds)
    added = set(seeds)
    for _ in range(d):
        temp_set = set()
        for seed in added:
            for u in graph.successors(seed):  # Add all the covered seeds
                temp_set.add(u)
        added = temp_set.difference(covered)
        covered = covered.union(added)

    if len(covered) > graph.vcount():
        logging.debug(f"Seeds: {seeds}")  # temp
    return len(covered)

def greedy(closed_graph: igraph.Graph, k: int, debug=False):
    """find k-max d-hop cover with greedy

    Args:
        graph (igraph.Graph): graph
        k (int): the number of seeds k
        debug (bool): debug mode

    Returns:
        seeds, covernum: selected seeds, and the seed count
    """
    seeds = []

    nodes_num = closed_graph.vcount()
    covered = [False] * nodes_num
    cover_num = 0

    inf_list = [deg + 1 for deg in closed_graph.outdegree()]

    node_queue = [Node(i, -inf_list[i]) for i in range(nodes_num)]
    heapq.heapify(node_queue)
    i = 0

    while i < k and cover_num < nodes_num:  # while there's still free point or unused budget

        # Find the node with max marginal utility
        max_inf_node = heapq.heappop(node_queue)
        if inf_list[max_inf_node.id] != - max_inf_node.value:
            max_inf_node.value = -inf_list[max_inf_node.id]
            heapq.heappush(node_queue, max_inf_node)
            continue

        i += 1
        seeds.append(max_inf_node.id)
        if not covered[max_inf_node.id]:  # Update predecessors
            covered[max_inf_node.id] = True  # 1. mark max_node as covered
            cover_num += 1
            inf_list[max_inf_node.id] -= 1
            # 2. all the preds have influence -1
            for predecessor in closed_graph.predecessors(max_inf_node.id):
                inf_list[predecessor] -= 1

        # Update successors
        for successor in closed_graph.successors(max_inf_node.id):
            if not covered[successor]:
                # 1. mark all the successors as covered
                covered[successor] = True
                cover_num += 1
                # 2. all the successors have influence -1 (since there is no unitility to cover themselves)
                inf_list[successor] -= 1
                # 3. all the (predecessors of successors) have influence -1
                for predecessor in closed_graph.predecessors(successor):
                    inf_list[predecessor] -= 1

        if debug:
            print(
                f"Round {i}: {max_inf_node.id} is selected. {cover_num} nodes are covered.")
                
    return seeds, cover_num
