import os
import igraph
import pandas as pd
from graph_util import get_rev_dgl


PATH_TO_DIRECTED_GRAPHS = "/home/featurize/data/data/graphs/directed/"  # Change the path
PATH_TO_UNDIRECTED_GRAPHS = "/home/featurize/data/data/graphs/undirected/"  # Change the path

BLACK_LIST = [
    "LiveJournal.txt"
]
STAT_CSV = "/home/featurize/work/newMCP/output/descriptive_statistics.csv"


# dglgraph parameters
INPUT_DIM = 32
FEATURE_TYPE="1"

DIRECTED_GRAPHS = None  # all
UNDIRECTED_GRAPHS = []  # empty

# DIRECTED_GRAPHS = [  # small directed graphs
#     "soc-sign-bitcoinotc.txt",
#     "soc-advogato.txt",
#     "soc-wiki-elec.txt",
#     "soc-anybeat.txt",
#     "soc-gplus.txt",
#     "soc-epinions.txt"
# ]


# DIRECTED_GRAPHS = [
#     "soc-brightkite.txt",
#     "soc-slashdot.txt",
#     "soc-douban.txt",
#     "soc-themarker.txt",
#     "soc-sign-epinions.txt",
#     "soc-gowalla.txt",
#     "soc-twitter-follows.txt",
#     "soc-twitter-follows-mun.txt",
#     "soc-delicious.txt",
# ]


def get_graph_names(path_to_test):
    graph_names = []
    for rt, _, files in os.walk(path_to_test):
        if rt == path_to_test:
            for file in files:
                if file.endswith('.txt'):
                    graph_names.append(file)
    return graph_names


def load_igraphs(graph_names, path, is_directed, black_list=BLACK_LIST, n_limit=None, m_limit=None):
    df_stat = pd.read_csv(STAT_CSV, index_col="graph")
    for name in graph_names:
        if name in black_list:
            continue
        print(f"Current graph: {name}", flush=True)
        if n_limit is not None:
            n = df_stat.loc[name].n
            if n > n_limit:
                print(f"Skipped {name} with n={int(n)}>{n_limit}", flush=True)
                continue
        if m_limit is not None:
            m = df_stat.loc[name].m
            if m > m_limit:
                print(f"Skipped {name} with m={int(m)}>{m_limit}", flush=True)
                continue

        graph = igraph.Graph().Read_Edgelist(
            f"{path}{name}", directed=is_directed)
        yield name, graph, is_directed


def next_graph(n_limit=None, m_limit=None):
    for path, is_directed, graph_names in [(PATH_TO_DIRECTED_GRAPHS, True, DIRECTED_GRAPHS), (PATH_TO_UNDIRECTED_GRAPHS, False, UNDIRECTED_GRAPHS)]:
        if graph_names is None:
            graph_names = get_graph_names(path)
        for x in load_igraphs(graph_names, path, is_directed, n_limit=n_limit, m_limit=m_limit):
            yield x


def load_dglgraphs(graph_names, path, is_directed, black_list=BLACK_LIST, n_limit=None, m_limit=None, input_dim=INPUT_DIM, feature_type=FEATURE_TYPE, use_cuda=False):
    df_stat = pd.read_csv(STAT_CSV, index_col="graph")
    for name in graph_names:
        if name in black_list:
            continue
        print(f"Current graph: {name}", flush=True)
        if n_limit is not None:
            n = df_stat.loc[name].n
            if n > n_limit:
                print(f"Skipped {name} with n={int(n)}>{n_limit}", flush=True)
                continue

        if m_limit is not None:
            m = df_stat.loc[name].m
            if m > m_limit:
                print(f"Skipped {name} with m={int(m)}>{m_limit}", flush=True)
                continue
        
        # Load igraph
        graph = igraph.Graph().Read_Edgelist(
            f"{path}{name}", directed=is_directed)
        
        # Load dglgraph
        dglgraph = get_rev_dgl(graph, feature_type,
                               input_dim, is_directed, use_cuda)
        yield name, graph, dglgraph, is_directed


def next_dglgraph(input_dim=32, n_limit=None, m_limit=None, use_cuda=False):
    for path, is_directed, graph_names in [(PATH_TO_DIRECTED_GRAPHS, True, DIRECTED_GRAPHS), (PATH_TO_UNDIRECTED_GRAPHS, False, UNDIRECTED_GRAPHS)]:
        if graph_names is None:
            graph_names = get_graph_names(path)
        for x in load_dglgraphs(graph_names, path, is_directed, input_dim=input_dim, n_limit=n_limit, m_limit=m_limit, use_cuda=use_cuda):
            yield x
