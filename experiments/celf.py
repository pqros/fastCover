import sys
sys.path.append("/home/featurize/work/newMCP/")  # Change the path

import numpy as np
import pandas as pd
from baselines.heuristics import d_greedy, greedy, bfs, get_influence
from experiments.igraph_loader import next_graph
from pathlib import Path
import time
import logging
import signal
from timeout_decorator import timeout
import os
import multiprocessing
from util import get_memory


# Global Vars
TIME_LIMIT = 600
PATH_TO_OUTPUT = "/home/featurize/work/newMCP/output/greedy/8000/"

FILE_OUTPUT = "celf_all.csv"
DS = [2, 3]
KS = [2 ** n for n in range(10)]
repeat = 1
bfs = timeout(seconds=TIME_LIMIT)(bfs)
greedy = timeout(seconds=TIME_LIMIT)(greedy)


def integrated_celf(x):
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
#             logging.StreamHandler()
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
    FAIL = False
    d = 1
    for k in KS:
        try:
            logging.info(f"Graph: {graph_name}, d: {d}, k: {k}. Using CELF")
            ts = np.zeros(repeat)
            for i in range(repeat):
                t_start = time.time()
                seeds_1, n_covered = greedy(graph, k)
                ts[i] = (time.time() - t_start)
            t0_mean = ts.mean()
            t0_std = ts.std() / np.sqrt(repeat)

            dict_seeds_1[k] = seeds_1
            dict_t_mean_1[k] = t0_mean
            dict_t_std_1[k] = t0_std

            logging.info(
                f"d: {d}, k: {k}. Calculated 1-coverage. Time: {t0_mean:.2f} ({t0_std:.2f}). Coverage: {n_covered}/{n}={n_covered/n:.2f}"
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
                "CELF": True,
            })

        except:
            logging.info(
                f"d: {d}, k: {k}. Cannot calculate 1-coverage!"
            )
            FAIL = True
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
                "CELF": True,
            })
            break

    if not FAIL:
        CELF = True
        for d in DS:
            logging.info(f"d: {d}")
            try:
                # Calculate d-closure
                k = 0
                ts = np.zeros(repeat)
                for i in range(repeat):
                    t_start = time.time()
                    closed_graph = bfs(graph, d)
                    ts[i] = (time.time() - t_start)
                t0_mean = ts.mean()
                t0_std = ts.std() / np.sqrt(repeat)
                logging.info(
                    f"Calculated {d}-closure. Time: {t0_mean:.2f} ({t0_std:.2f})"
                )
                records.append({
                        "graph": graph_name,
                        "n": n,
                        "m": m,
                        "d": d,
                        "k": k,
                        "n_covered": np.nan,
                        "coverage": np.nan,
                        "t_mean": t0_mean,
                        "t_std": t0_std,
                        "CELF": True,
                })

                # Calculate coverage with 1-coverage seeds
                CELF = False
                for k in KS:
                    logging.info(f"Graph: {graph_name}, d: {d}, k: {k}. Using seeds in 1-coverage.")
                    seeds = dict_seeds_1[k]
                    t1_mean = dict_t_mean_1[k]
                    t1_std = dict_t_std_1[k]

                    ts = np.zeros(repeat)
                    for i in range(repeat):
                        t_start = time.time()
                        n_covered = get_influence(closed_graph, seeds)
                        ts[i] = (time.time() - t_start)

                    t_mean = ts.mean() + t1_mean
                    t_std = ts.std() / np.sqrt(repeat) + t1_std
                    logging.info(
                        f"Coverage: {n_covered}/{n}={n_covered/n:.4f}. Time: {t_mean:.2f} ({t_std:.2f})")

                    records.append({
                        "graph": graph_name,
                        "n": n,
                        "m": m,
                        "d": d,
                        "k": k,
                        "n_covered": n_covered,
                        "coverage": n_covered/n,
                        "t_mean": t_mean,
                        "t_std": t_std,
                        "CELF": CELF,
                    })

                # Recalculate seeds
                CELF = True
                for k in KS:
                    logging.info(f"Graph: {graph_name}, d: {d}, k: {k}. Using CELF")

                    ts = np.zeros(repeat)
                    for i in range(repeat):
                        t_start = time.time()
                        _, n_covered = greedy(closed_graph, k=k)
                        ts[i] = (time.time() - t_start)

                    t_mean = ts.mean() + t0_mean
                    t_std = ts.std() / np.sqrt(repeat) + t0_std
                    logging.info(
                        f"Coverage: {n_covered}/{n}={n_covered/n:.4f}. Time: {t_mean:.2f} ({t_std:.2f})")

                    records.append({
                        "graph": graph_name,
                        "n": n,
                        "m": m,
                        "d": d,
                        "k": k,
                        "n_covered": n_covered,
                        "coverage": n_covered/n,
                        "t_mean": t_mean,
                        "t_std": t_std,
                        "CELF": CELF,
                    })

            except:
                logging.info(f"Cannot solve the instance!")
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
                    "CELF": CELF,
                })

    df_result = pd.DataFrame(records)
    df_result.to_csv(f"{PATH_TO_OUTPUT}{graph_name}.csv")
    logging.info(f"Saving results of {graph_name} to {PATH_TO_OUTPUT}{graph_name}.csv !")
    print(f"Graph: {graph_name}, n: {n}, m: {m} Done!")
    logging.info(f"Done!")
    return


def multiprocessing_celf():
    pool = multiprocessing.Pool(4)
    list(pool.imap(integrated_celf, next_graph(m_limit=None)))
    return


if __name__ == '__main__':
    results = multiprocessing_celf()
