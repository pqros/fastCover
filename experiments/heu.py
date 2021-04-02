import sys
sys.path.append("/home/featurize/work/newMCP/")  # Change the path

import numpy as np
import pandas as pd
from baselines.heuristics import d_greedy, greedy, bfs, get_influence, heu1
from experiments.igraph_loader import next_graph
from pathlib import Path
import time
import logging
import signal
from timeout_decorator import timeout
import os
import multiprocessing
from itertools import product
from util import get_memory


# Global Vars
TIME_LIMIT = 900
PATH_TO_OUTPUT = "/home/featurize/work/newMCP/output/heu/"  # Change the path

FILE_OUTPUT = "heu_all.csv"
DS = [1, 2, 3]
KS = [2 ** n for n in range(10)]
repeat = 5
bfs = timeout(seconds=TIME_LIMIT)(bfs)
greedy = timeout(seconds=TIME_LIMIT)(greedy)
heu1 = timeout(seconds=TIME_LIMIT)(heu1)

def integrated_heu(x):
    graph_name, graph, is_directed = x
    graph_name = os.path.splitext(graph_name)[0]
    
    # Config Loggers
    DATE = time.strftime('%m-%d', time.localtime())
    TIME = time.strftime('%H.%M.%S', time.localtime())
    Path(f"log/{DATE}").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"log/{DATE}/debug-{DATE}-{TIME}-{graph_name}.log"),
            logging.StreamHandler()
        ]
    )

    m, n = graph.ecount(), graph.vcount()
    print(f"Graph: {graph_name}, n: {n}, m: {m}")
    logging.info(f"Graph: {graph_name}, n: {n}, m: {m}")

    # Final results
    records = []

    # Prepare case d=1
    dict_seeds_1 = {}
    dict_t_mean_1 = {}
    dict_t_std_1 = {}

    for d, k in product(DS, KS):
        try:
            logging.info(f"Graph: {graph_name}, d: {d}, k: {k}. Using HEU1")
            ts = np.zeros(repeat)
            mems = np.zeros(repeat)
            for i in range(repeat):
                t_start = time.time()
                _, n_covered = heu1(graph, k, d)
                ts[i] = (time.time() - t_start)
                mems[i] = get_memory()
                
            t0_mean = ts.mean()
            t0_std = ts.std() / np.sqrt(repeat)
            mem_mean = mems.mean()
            mem_std = mems.std() / np.sqrt(repeat)

            # dict_t_mean_1[k] = t0_mean
            # dict_t_std_1[k] = t0_std

            logging.info(
                f"d: {d}, k: {k}. Time: {t0_mean:.2f} ({t0_std:.2f}). Coverage: {n_covered}/{n}={n_covered/n:.2f}"
            )
            records.append({
                "graph": graph_name,
                "n": n,
                "m": m,
                "d": d,
                "k": k,
                "n_covered": n_covered,
                "coverage": n_covered/n,
                "t_mean": t0_mean,
                "t_std": t0_std,
                "memory": mem_mean,
            })

        except:
            logging.info(
                f"d: {d}, k: {k}. Cannot calculate 1-coverage!"
            )
            records.append({
                "graph": graph_name,
                "n": n,
                "m": m,
                "d": d,
                "k": k,
                "n_covered": np.nan,
                "coverage": np.nan,
                "t_mean": np.nan,
                "t_std": np.nan,
            })
            break

    df_result = pd.DataFrame(records)
    df_result.to_csv(f"{PATH_TO_OUTPUT}{graph_name}.csv")
    logging.info(f"Saving results of {graph_name} to {PATH_TO_OUTPUT}{graph_name}.csv !")
    print(f"Graph: {graph_name}, n: {n}, m: {m} Done!")
    logging.info(f"Done!")
    return


def multiprocessing_heu():
    # pool = multiprocessing.Pool(4)
    # list(pool.imap(integrated_heu, next_graph(m_limit=None)))
    for x in next_graph():
        integrated_heu(x)
    return


if __name__ == '__main__':
    results = multiprocessing_heu()
