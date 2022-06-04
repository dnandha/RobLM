import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout, read_dot


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('infile')
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    G = read_dot(args.infile)
    pred = nx.all_simple_paths(G, source=args.source, target=args.target, cutoff=2)

    # drop target and source node
    print([x for x in pred])
    pred = ','.join([x for x in pred])

    #print(f"{args.source}-[{pred}]={args.target}")
