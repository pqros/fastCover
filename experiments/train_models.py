import sys
sys.path.append("/home/featurize/work/newMCP/")  # Change the path

import torch
from graph_util import get_adj_mat
from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
from load_graph import load_train
from util import get_model
from train import train

import time
import logging
import pickle
import logging
from pathlib import Path
import pandas as pd
from itertools import product
from furl import furl
from baselines.heuristics import d_greedy, d_closure, greedy, bfs


# ==================== LOG SETTING ====================
DATE = time.strftime('%m-%d', time.localtime())
TIME = time.strftime('%H.%M.%S', time.localtime())
Path(f"log/{DATE}").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"log/{DATE}/debug-{DATE}_{TIME}.log"),
        logging.StreamHandler()
    ]
)
# ==================== END OF LOG SETTING ====================


# ==================== PARAMETERS ====================
TRAIN = True  # TRAIN MODE
HIDDEN_FEATS = [32, 32, 32, 32, 32, 32]
C_LOSS = 1
LOSS_FUNCTION = KSetMaxCoverAdjLoss(C_LOSS)
N_EPOCH = 50
N_BATCH = 2
ROUND = 2
LR = 5e-4
DS = [1]
K_TRAINS = [32]
SEEDS = [42]
MODEL_NAMES = [
    "GRAT3",
]


def train_models(debug=False):
    if debug:
        logging.info("=========New round=========")
# 
    # Prepare training instances
    _, train_graphs, train_dglgraphs = load_train(input_dim=HIDDEN_FEATS[0])
    if debug:
        logging.debug("Loading training graphs done.")
        logging.debug(f"{len(train_dglgraphs)} graphs are loaded.")

    # Train models
    for (k, d) in zip(K_TRAINS, DS):
        logging.debug(f"d: {d}, k: {k}")
        adj_matrices = []
        greedy_perfs = []
        train_closed_graphs = []
        for train_graph in train_graphs:
            # calculate graph closure
            train_closed_graph = bfs(train_graph, d=d)
            train_closed_graphs.append(train_closed_graph)

            # calculate the adj matrix of each graph
            adj_matrices.append(get_adj_mat(train_closed_graph))

            # solve by greedy as baseline
            _, n_covered = greedy(train_closed_graph, k=k)
            greedy_perfs.append(n_covered)
            
            logging.debug(f"Greedy influence: {n_covered}/{train_graph.vcount()}={n_covered/train_graph.vcount():.2f}.")

        for model_name, seed in product(MODEL_NAMES, SEEDS):
            logging.debug(f"Parameters: model: {model_name}, d: {d}, k: {k}, seed: {seed}")
            torch.manual_seed(seed)  # reproducibility
            param_file = furl("params").add({
                "model": model_name,
                "d": d,
                "seed": seed,
            })
            
            net = get_model(model_name, *HIDDEN_FEATS)
            if debug:
                logging.debug("Model loaded. Model info:")
                logging.debug(net)

            loss_list, _, net = train(
                net=net, adj_matrices=adj_matrices,
                graphs=train_graphs, dglgraphs=train_dglgraphs,
                loss_function=LOSS_FUNCTION, greedy_perfs=greedy_perfs,
                k_train=k, n_epoch=N_EPOCH, n_batch=N_BATCH, lr=LR,
                patience=5, closed_graphs=train_closed_graphs
            )  # simplified with pre-calculated adj matrices

            net.cpu()
            torch.save(net.state_dict(), f=f"/home/featurize/work/newMCP/output/params/{param_file}.pt")  # Change the path
            with open(f"/home/featurize/work/newMCP/output/losses-{param_file}.pkl", "wb") as f:
                pickle.dump(loss_list, f)    
    return


if __name__ == '__main__':
    train_models(debug=False)
