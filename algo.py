import heapq
import igraph


class Node:
    def __init__(self, id, value):
        self.id = id
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        # we store the negative value to use max heap.
        return f"{self.id}: {-self.value}"


def greedy(graph: igraph.Graph, k: int, debug=False):
    """find k-max cover with greedy

    Args:
        graph (igraph.Graph): graph
        k (int): the number of seeds k
        debug (bool): debug mode

    Returns:
        seeds, covernum: selected seeds, and the seed count
    """
    seeds = []
    nodes_num = graph.vcount()
    covered = [False] * nodes_num
    cover_num = 0

    # inf_list = graph.outdegree()
    # inf_list = [inf_list[i] + 1 for i in range(nodes_num)]
    # the value (influence) of a vertex equals its outdegree +1 (itself)
    inf_list = [deg + 1 for deg in graph.outdegree()]

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
            for predecessor in graph.predecessors(max_inf_node.id):
                inf_list[predecessor] -= 1

        # Update successors
        for successor in graph.successors(max_inf_node.id):
            if not covered[successor]:
                # 1. mark all the successors as covered
                covered[successor] = True
                cover_num += 1
                # 2. all the successors have influence -1 (since there is no unitility to cover themselves)
                inf_list[successor] -= 1
                # 3. all the (predecessors of successors) have influence -1
                for predecessor in graph.predecessors(successor):
                    inf_list[predecessor] -= 1

        if debug:
            print(
                f"Round {i}: {max_inf_node.id} is selected. {cover_num} nodes are covered.")
                
    return seeds, cover_num


def getCover(seed, covered, graph):
    return [covered[succ] for succ in graph.successors(seed)] + [covered[seed]]


def globalImprove(graph, seeds):
    n = graph.vcount()

    # 所有节点的种子父节点数量（包括自己）
    covered = [0]*n
    for seed in seeds:
        covered[seed] += 1
        for succ in graph.successors(seed):
            covered[succ] += 1
    
    # 当前种子和非种子
    curSeeds = set(seeds)
    curNotSeeds = set(range(n)) - set(seeds)
    # 两个用来计数的
    uniqueCover = [getCover(node, covered, graph).count(1)
                   for node in range(n)]
    NotSeedCover = [getCover(node, covered, graph).count(0)
                    for node in range(n)]
    # 两个队列，当前种子集和当前非种子集
    SeedsList = [Node(seed, uniqueCover[seed]) for seed in seeds]
    heapq.heapify(SeedsList)
    NotSeedsList = [Node(notseed, -NotSeedCover[notseed])
                    for notseed in curNotSeeds]
    heapq.heapify(NotSeedsList)
    # 独立覆盖最小的种子，和独立覆盖最大的非种子进行交换
    # 涉及到两个节点覆盖的所有节点的父节点的独立覆盖数量
    count = 0
    while True:
        minseed = heapq.heappop(SeedsList)
        if minseed.inf != uniqueCover[minseed.id]:
            minseed.inf = uniqueCover[minseed.id]
            heapq.heappush(SeedsList, minseed)
            continue
        maxnotseed = heapq.heappop(NotSeedsList)
        if maxnotseed.inf != -NotSeedCover[maxnotseed.id]:
            maxnotseed.inf = -NotSeedCover[maxnotseed.id]
            heapq.heappush(NotSeedsList, maxnotseed)
            heapq.heappush(SeedsList, minseed)
            continue
        # print('Added', NotSeedCover[maxnotseed.id], uniqueCover[minseed.id])
        if uniqueCover[minseed.id] > NotSeedCover[maxnotseed.id]:
            print('Changed:', count)
            break
        if uniqueCover[minseed.id] < 0:
            break
        # 对于删除的种子来说，可能有一部分覆盖节点不再被覆盖，有一部分节点变成新的单一父种子节点
        # covered变了，两个队列都要更新
        curSeeds.remove(minseed.id)
        curNotSeeds.add(minseed.id)
        heapq.heappush(NotSeedsList, minseed)
        for node in graph.successors(minseed.id) + [minseed.id]:
            covered[node] -= 1
            if covered[node] == 1:
                # 只需要更新其他原种子节点的独立覆盖数量
                for fatherseed in set(graph.predecessors(node)).intersection(curSeeds):
                    uniqueCover[fatherseed] += 1
            elif covered[node] == 0:
                # 更新所有原非种子节点的覆盖数量
                for fathernotseed in set(graph.predecessors(node)).intersection(curNotSeeds):
                    NotSeedCover[fathernotseed] += 1
        # 对于新增的种子来说，其覆盖的节点需要covered++，还要改变其父种子节点的uniqueCovered或NotSeedCover
        curSeeds.add(maxnotseed.id)
        curNotSeeds.remove(maxnotseed.id)
        heapq.heappush(SeedsList, maxnotseed)
        for node in graph.successors(maxnotseed.id) + [maxnotseed.id]:
            covered[node] += 1
            if covered[node] == 1:
                for fatherseed in set(graph.predecessors(node)).intersection(curSeeds):
                    uniqueCover[fatherseed] += 1
                for pred in set(graph.predecessors(node)).intersection(curNotSeeds):
                    NotSeedCover[pred] -= 1
            # elif covered[node]==2:
            #     for fatherseed in set(graph.predecessors(node)+[node]).intersection(curSeeds):
            #         uniqueCover[fatherseed] -= 1
        count += 1
        if count == len(seeds):
            break
    return list(curSeeds)