import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout, read_dot


def read_graph(file):
    return read_dot(file)


def prune_graph(G, source, target):
    pred = nx.all_simple_paths(G, source=source, target=target, cutoff=2)

    # drop target and source node
    pred = ','.join([x[1] for x in pred])
    return f"{source}-[{pred}]={target}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('infile')
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    print(prune_graph(read_graph(args.infile), args.source, args.target))
