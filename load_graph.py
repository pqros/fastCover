import igraph
from graph_util import get_rev_dgl, gen_erdos_graphs
import os


# ==================== PARAMETERS ====================
USE_CUDA_TRAIN = True
USE_CUDA_TEST = True

# undirected
# DIRECTED_TRAIN = False
# PATH_TO_TRAIN = "data/graphs/undirected/"

# DIRECTED_TEST = True
# PATH_TO_TEST = "/home/featurize/data/data/graphs/directed/"
# TRAIN_LIST = [
#     "fb-pages-tvshow.txt",
#     "fb-pages-politician.txt",
#     "fb-pages-government.txt",
#     "fb-pages-public-figure.txt",
#     # "HepPh.txt",
# ]
# VAL_LIST = []

# directed
DIRECTED_TRAIN = True
DIRECTED_TEST = True
# PATH_TO_TRAIN = "/home/featurize/data/data/graphs/directed/"
# PATH_TO_TEST = "/home/featurize/data/data/graphs/directed/"
PATH_TO_TRAIN = "/home/featurize/work/newMCP/data/graphs/directed/"
PATH_TO_TEST = "/home/featurize/work/newMCP/data/graphs/directed/"
TRAIN_LIST = [
    "soc-wiki-elec.txt",
    "soc-anybeat.txt",
    "soc-advogato.txt",
    "soc-sign-bitcoinotc.txt"
]

VAL_LIST = [
    "soc-gplus.txt",
    "soc-epinions",
    "soc-brightkite.txt",
    "soc-themarker.txt",
]


FEATURE_TYPE = "1"
N_TEST_GRAPH = 1
N_TEST_NODE = 3000
# ==================== END OF PARAMETERS ====================


def load_train(input_dim, train_list=TRAIN_LIST, directed_train=DIRECTED_TRAIN, feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TRAIN, path_to_train=PATH_TO_TRAIN):
    graphs = []
    dglgraphs = []
    for file in train_list:
        g = igraph.Graph().Read_Edgelist(
            f"{path_to_train}{file}", directed=directed_train)
        dg = get_rev_dgl(g, feature_type, input_dim, directed_train, use_cuda)
        graphs.append(g)
        dglgraphs.append(dg)

    return train_list, graphs, dglgraphs


def get_test_names(path_to_test=PATH_TO_TEST):
    test_names = []
    for rt, _, files in os.walk(path_to_test):
        if rt == path_to_test:
            for file in files:
                if file.endswith('.txt') and 'friend' not in file:
                    test_names.append(file)
    return test_names


def gen_random_test(input_dim, directed_test=DIRECTED_TEST, feature_type=FEATURE_TYPE, n_test_graph=N_TEST_GRAPH, n_test_node=N_TEST_NODE, p=1e-2):
    graphs = gen_erdos_graphs(n_test_graph, n_test_node, p, directed_test)

    dglgraphs = [get_rev_dgl(
        graph, feature_type, input_dim, directed_test, False) for graph in graphs]
    return graphs, dglgraphs


def load_test(input_dim, directed_test=DIRECTED_TEST, feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TEST, path_to_test=PATH_TO_TEST):

    test_list = get_test_names(path_to_test)
    for graph_name in test_list:
        graph = igraph.Graph().Read_Edgelist(
            f"{path_to_test}{graph_name}", directed=directed_test)
        dglgraph = get_rev_dgl(graph, feature_type,
                               input_dim, directed_test, use_cuda)
        yield graph_name, graph, dglgraph


def load_igraph(filename, path=PATH_TO_TEST, is_directed=DIRECTED_TEST):
    graph = igraph.Graph().Read_Edgelist(
        f"{path}{filename}", directed=is_directed)
    return graph
