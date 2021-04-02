import time
import sys
sys.path.append("/home/featurize/work/newMCP/")

from baselines.heuristics import d_greedy
from load_graph import load_train
import pandas as pd

def test_d_greedy(debug=False):
    graph_names, graphs, _ = load_train(input_dim=32)
    ds = [1, 2, 3]
    ks = [10, 20, 30, 40, 50]
    results = []
    for graph_name, graph in zip(graph_names, graphs):
        for d in ds:
            for k in ks:
                t0 = time.time()
                _, n_covered = d_greedy(graph, k, d)
                t = time.time() - t0
                results.append({"graph": graph_name, "d": d, "k": k, "n_covered": n_covered, "n": graph.vcount(),"coverage_rate": n_covered/graph.vcount()})
                print(
                    f"Graph {graph_name}: d={d}, k={k}, {n_covered}/{graph.vcount()}={n_covered/graph.vcount():.2f} ({t:.2f}s)")

    df_result = pd.DataFrame(results)
    df_result.to_csv("../output/d_hop_greedy_train.csv", index=False)
    return


if __name__ == '__main__':
    test_d_greedy(debug=False)
