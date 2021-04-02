import sys
sys.path.append("/home/featurize/work/newMCP/")  # Change the path.

import time
from pathlib import Path
import logging
from util import get_filename, get_model, get_memory
from glob import glob
from furl import furl
import torch
import pandas as pd
from experiments.igraph_loader import next_graph, next_dglgraph
from baselines.heuristics import d_greedy, greedy, bfs, get_influence, get_influence_d
import numpy as np
from timeout_decorator import timeout


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
TIME_LIMIT = 300
HIDDEN_FEATS = [32] * 6
PATH_TO_MODELS = "/home/featurize/work/newMCP/output/params/"  # Change the path
PATH_TO_OUTPUT = "/home/featurize/work/newMCP/output/grat/4000/"  # Change the path
USE_CUDA = True
KS = [2 ** n for n in range(11)]
bfs = timeout(seconds=TIME_LIMIT)(bfs)


def get_modelnames(path_to_models=PATH_TO_MODELS):
    return glob(f"{path_to_models}*.pt")


def load_net(param_file, use_cuda=USE_CUDA, debug=False):
    f = furl(param_file)
    d = int(f.args["d"])
    model_name = f.args["model"]
    net = get_model(model_name, *HIDDEN_FEATS)
    net.load_state_dict(torch.load(param_file))
    if use_cuda:
        net.cuda()
    return net


def evaluate_model(net, d, model_name, seed, graph_name, graph, dglgraph, ks, repeat=1, closed_graph=None, debug=False):
    n, m = graph.vcount(), graph.ecount()

    records = []
    try:
        for k in ks:
            ts = np.zeros(repeat)
            for i in range(repeat):
                # Select seeds
                t_start = time.time()
                out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(1)
                _, nn_seeds = torch.topk(out, k)
                ts[i] = (time.time() - t_start)
            
            # Evaluate time
            t_mean = ts.mean() 
            t_std = ts.std() / np.sqrt(repeat)

            # Evaluate memory
            memory = get_memory()

            # Evaluate covereage
            # if closed_graph is None:
            #     n_covered = get_influence(graph, nn_seeds)
            # else:
            #     n_covered = get_influence(closed_graph, nn_seeds)
            n_covered = get_influence_d(graph, nn_seeds, d)
            logging.info(f"k: {k}. Coverage: {n_covered}/{n}={n_covered/n:.2f}. Time: {t_mean:.2f} ({t_std:.2f})")

            # Write to records
            records.append({
                "graph": graph_name,
                "model": model_name,
                "seed": seed,
                "n": n,
                "m": m,
                "d": d,
                "k": k,
                "n_covered": n_covered,
                "coverage": n_covered/n,
                "t_mean": t_mean,
                "t_std": t_std,
                "memory": memory,
                "gpu": USE_CUDA,
            })
    except:
        logging.info(f"Failed to evaluate on {graph_name}!")
        records.append({
            "graph": graph_name,
            "model": model_name,
            "seed": seed,
            "n": n,
            "m": m,
            "d": d,
            "k": np.nan,
            "n_covered": np.nan,
            "coverage": np.nan,
            "t_mean": np.nan,
            "t_std": np.nan,
            "memory": np.nan,
            "gpu": USE_CUDA
        })
    return records


def evaluate_models(debug=False):
    model_param_names = get_modelnames()

    for x in next_dglgraph(input_dim=HIDDEN_FEATS[0], n_limit=None, m_limit=None, use_cuda=USE_CUDA):
        name, graph, dglgraph, is_directed = x
        graph_name_body = get_filename(name)

        records = []
        for d in [1, 2, 3]:
            # closed_graph = bfs(graph, d=d)
            closed_graph = None

            for param_name in model_param_names:
                param_name_body = get_filename(param_name)
                net_d = int(furl(param_name_body).args["d"])
                if net_d != d:
                    continue
                
                logging.info(f"Graph: {name}. Model: {param_name_body}. d: {d}.")
                net = load_net(param_name)
                model_name = furl(param_name_body).args["model"]
                seed = furl(param_name_body).args["seed"]
                records.extend(
                    evaluate_model(net, d, model_name, seed, name, graph, dglgraph, ks=KS, repeat=1, closed_graph=closed_graph)
                )  # name => graph name
        df_result = pd.DataFrame(records)
        df_result.to_csv(f"{PATH_TO_OUTPUT}{graph_name_body}.csv")
        logging.info(f"Saving results of {graph_name_body} to {PATH_TO_OUTPUT}{graph_name_body}.csv!")
    return


if __name__ == '__main__':
    evaluate_models(debug=False)
