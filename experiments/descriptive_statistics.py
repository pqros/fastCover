import pandas as pd
import numpy as np
import igraph
import os
import time

PATH_TO_DIRECTED_GRAPHS = "/home/featurize/data/data/graphs/directed/"
PATH_TO_UNDIRECTED_GRAPHS = "/home/featurize/data/data/graphs/undirected/"
PATH_TO_OUTPUT = "/home/featurize/work/newMCP/output/"
FILE_OUTPUT = "descriptive_statistics.csv"
BLACK_LIST = [
    "LiveJournal.txt"
]


def get_graph_names(path_to_test):
    graph_names = []
    for rt, _, files in os.walk(path_to_test):
        if rt == path_to_test:
            for file in files:
                if file.endswith('.txt'):
                    graph_names.append(file)
    return graph_names


def load_igraphs(graph_names, path, is_directed, black_list=BLACK_LIST):
    records = []
    for name in graph_names:
        if name in black_list:
            continue
        print(f"Current graph: {name}")
        m = n = -1
        t_start = time.time()
        graph = None
        try:
            graph = igraph.Graph().Read_Edgelist(
                f"{path}{name}", directed=is_directed)
            m, n = graph.ecount(), graph.vcount()
        except:
            del graph
            print(f"Loading {name} failed.")
        t = time.time() - t_start
        records.append({
            "graph": name,
            "n": n,
            "m": m,
            "t": t,
            "directed": is_directed,
        })
    return records


if __name__ == '__main__':
    records = []
    for path, is_directed in [(PATH_TO_DIRECTED_GRAPHS, True), (PATH_TO_UNDIRECTED_GRAPHS, False)]:
        graph_names = get_graph_names(path)
        records.extend(load_igraphs(graph_names, path, is_directed))
    df_result = pd.DataFrame(records)
    df_result.to_csv(f"{PATH_TO_OUTPUT}{FILE_OUTPUT}")
