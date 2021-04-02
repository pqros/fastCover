import time
import sys
from util import get_model, get_memory
from load_graph import load_train
from algo import greedy
import torch
from copy import deepcopy
from tqdm import tqdm
import logging
from pytorchtools import EarlyStopping
import numpy as np

# ==================== PARAMETERS ====================
PATH_TO_PARAMS = "params/"
USE_CUDA_TRAIN = True
# ==================== END OF PARAMETERS ====================


# ==================== HELPER FUNCTIONS ====================
def model_file(model_name, path_to_params=PATH_TO_PARAMS):
    time_str = time.strftime("%m-%d", time.localtime())
    return f"{path_to_params}{time_str}{model_name}.pkl"


def get_influence(graph, seeds):
    covered = set()
    for seed in seeds:
        covered.add(int(seed))
        for u in graph.successors(seed):  # Add all the covered seeds
            covered.add(int(u))
    return len(covered)


def get_k_top(net, dglgraph, test_k):
    """Get top k nodes based on the net
    """
    dglgraph_ = deepcopy(dglgraph)
    seeds = set()

    cum_n_covered = 0
    for i in range(test_k):
        # bar.set_description('GetKTop')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        _, top_node = torch.max(out, 0)  # index of the best-scored node
        seeds.add(top_node)
        srcs = []  # the src node of the i_th edge
        dsts = []  # the dst node of the i_th edge
        covered = dglgraph_.predecessors(top_node)
        parents = dglgraph_.successors(top_node)

        # add edges (top_node, succ)
        srcs.extend([top_node]*len(parents))
        dsts.extend(parents)

        # add edges (pred, top_node)
        srcs.extend(covered)
        dsts.extend([top_node]*len(covered))

        for node in covered:
            parents = dglgraph_.successors(node)
            srcs.extend([node]*len(parents))
            dsts.extend(parents)
        dglgraph_.remove_edges(dglgraph_.edge_ids(srcs, dsts))

        n_covered = len(covered)
        cum_n_covered += n_covered
        logging.info(f"Node {i+1}: {n_covered}/{cum_n_covered} points are covered.")

    return list(seeds)


def get_x_top(net, dglgraph, test_k, x):  # get top 1/x points.
    d = deepcopy(dglgraph)
    seeds = []

    for _ in range((test_k + x - 1) // x):
        # bar.set_description(' Get'+str(x)+'Top')
        out = net(d, d.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        _, topNodes = torch.topk(out, x)
        seeds.extend(topNodes.tolist())
        srcs = []
        dsts = []
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(d.predecessors(topNode))
        nodes = list(set(nodes))
        for node in nodes:
            for succ in d.successors(node):
                srcs.append(node)
                dsts.append(succ)
        d.remove_edges(d.edge_ids(srcs, dsts))

    seeds = list(set(seeds))
    return seeds


def get_x_top_plus(net, graph, dglgraph, test_k, x):
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = []
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
        # bar = list(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
        # bar = list(range(test_k//x+1))
    for i in bar:
        bar.set_description(' Get'+str(x)+'TopPlus')

        out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)

        y = x if test_k >= x else test_k
        test_k -= x

        _, topNodes = torch.topk(out, x)

        # 删边可以有重复，但不能删除不存在的边
        nodes = topNodes.tolist()
        topNodesNum = len(nodes)
        seeds.extend(nodes)
        for i in range(topNodesNum):
            nodes.extend(graph_.successors(nodes[i]))
        nodes = list(set(nodes))

        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))

        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph.remove_edges(dglgraph.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float()
    seeds = list(set(seeds))
    return seeds


def get_x_top_plus_plus(net, graph, dglgraph, test_k, x):
    dglgraph_ = deepcopy(dglgraph)
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = set()
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
    for i in bar:
        bar.set_description(' GetXTopPlusPlus')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        r = 10  # 查找范围
        _, indices = torch.topk(out, r*x)
        count = 0
        i = 0
        topNodes = []
        while count < x and i < r*x:
            idx = int(indices[i])
            if idx in seeds:
                pass
            else:
                seeds.add(idx)
                topNodes.append(idx)
                count += 1
            i += 1
        # 删边可以有重复，但不能删除不存在的边
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(graph_.successors(topNode))
        nodes = list(set(nodes))
        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))
        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph_.remove_edges(dglgraph_.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
    seeds = list(seeds)
    return seeds
# ==================== END OF HELPER FUNCTIONS ====================


def train_single(net, optimizer, n_epoch, loss_function, adj_mat, graph, dglgraph, greedy_perf, k, closed_graph=None):
    """helper function of train
    """
    loss_list = []
    for _ in range(n_epoch):
        out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
        loss = loss_function(out, adj_mat, k)
        loss_list.append(float(loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logits = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(
        1)
        
    _, indices = torch.topk(logits, k)

    if closed_graph is None:
        train_perf = get_influence(graph, indices)
    else:
        train_perf = get_influence(closed_graph, indices)
    
    perf_ratio = train_perf/greedy_perf
    logging.info(f"Train influence: {train_perf}/{greedy_perf}={perf_ratio:.2f}")
    return loss_list, perf_ratio


def train(net, adj_matrices, graphs, dglgraphs, loss_function, greedy_perfs, closed_graphs=None, k_train=1, n_epoch=5, n_batch=10, lr=0.1, save_filename=None, patience=5):
    if USE_CUDA_TRAIN:
        net.cuda()
    logging.info(f"Learning rate: {lr:.2}")

    # TODO: why do we use SGD?  -> generalization 
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # EarlyStopping Module
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 

    # cover performence
    train_perfs = []
    avg_train_perfs = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if closed_graphs is None:
        closed_graphs = [None] * len(graphs)
    
    for epoch in range(n_epoch):
        timer = time.time()
        for adj_mat, graph, dglgraph, greedy_perf, closed_graph in zip(adj_matrices, graphs, dglgraphs, greedy_perfs, closed_graphs):
            # train the i_th graph
            loss_list, perf = train_single(
                net, optimizer, n_batch, loss_function, adj_mat, graph, dglgraph, greedy_perf, k_train, closed_graph
            )
            train_perfs.append(perf)
            train_losses.append(sum(loss_list)/n_batch)  # track losses
        
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        
        train_perf = np.average(train_perfs)
        avg_train_perfs.append(train_perf)

        logging.info(f"Train Epoch {epoch} | Loss: {train_loss:.2f} | Perf: {train_perf:.2f} | Elapsed Time: {time.time() - timer:.2f}")

        # if save_filename:
        #     torch.save(net.state_dict(), f=f"{save_filename}-epoch-{epoch}.pt")
        #     logging.debug(f"Epoch {epoch} saved.")
        
        # clear lists to track next epoch
        train_losses = []
        train_perfs = []
        # valid_losses = []

        # early_stopping(train_loss, net)
        early_stopping(-train_perf, net)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    # load the last checkpoint with the best model
    net.load_state_dict(torch.load('checkpoint.pt'))
    return avg_train_losses, avg_train_perfs, net


def test_single(net, graph_name, graph, dglgraph, ks):
    n_node = graph.vcount()
    n_edge = graph.ecount()
    logging.info(f"Testing on the graph {graph_name}.")
  
    greedy_ts = []  # timing
    greedy_perfs = []
    nn_t0s = []
    nn_t1s = []
    nn_ts = []
    nn_perfs = []
    memories = []

    for k in ks:
        # greedy
        if k <= n_node:
            t0 = time.time()
            _, greedy_perf = greedy(graph, k)
            t1 = time.time()  # t1 - t0: calculate greedy influence

            out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(
        1)
            t2 = time.time()  # t2 - t1: nn
            _, nn_seeds = torch.topk(out, k)
            t3 = time.time()  # t3 - t2: get top k seeds.

            nn_perf = get_influence(graph, nn_seeds)  # we move it out
            memory = get_memory()

            greedy_time = t1 - t0
            nn_time_0 = t2 - t1
            nn_time_1 = t3 - t2
            nn_time = nn_time_0 + nn_time_1

            logging.info(
                f"n_node:{n_node}; n_edge:{n_edge}; k:{k}; perf ratio:{nn_perf/greedy_perf:.2f}; time ratio: {nn_time/greedy_time:.2f}; topk ratio: {nn_time_1/nn_time:.2f}"
            )
        
            greedy_ts.append(greedy_time)
            greedy_perfs.append(greedy_perf)
            nn_t0s.append(nn_time_0)
            nn_t1s.append(nn_time_1)
            nn_ts.append(nn_time)
            nn_perfs.append(nn_perf)
            memories.append(memory)
        else:
            greedy_ts.append(-1)
            greedy_perfs.append(-1)
            nn_t0s.append(-1)
            nn_t1s.append(-1)
            nn_ts.append(-1)
            nn_perfs.append(-1)     
            memories.append(-1)       

    result = {
        "graph_name": graph_name,
        "n": n_node,
        "m": n_edge,
        "k": ks,
        "greedy_perf": greedy_perfs,
        "greedy_time": greedy_ts,
        "nn_perf": nn_perfs,
        "nn_time": nn_ts,
        "nn_time_0": nn_t0s,
        "nn_time_1": nn_t1s,
        "memory": memories,
    }

    return result
