import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout, read_dot


def read_graph(file):
    return read_dot(file)


def prune_graph(G, source, target):
    pred = nx.all_simple_paths(G, source=source, target=target, cutoff=2)

    # drop target and source node
    pred = [x[1] for x in pred]
    if source in pred:
        pred.remove(source)
    if target in pred:
        pred.remove(target)
    return source, target, pred


def describe_relation(a, node):
    rel = []

    if a['dist'] > 5.:
        rel += ["distant"]
    elif a['dist'] > 4.:
        rel += ["far"]
    elif a['dist'] > 3.:
        rel += ["close"]
    elif a['dist'] > 2.:
        rel += ["closer"]
    elif a['dist'] > 1.:
        rel += ["near"]
    elif a['dist'] < 0.1:
        rel += ["next"]
    elif a['dist'] < 0.5:
        rel += ["on"]
    elif a['dist'] < 1.:
        rel += ["is"]

    rel += [node.split('|')[0]]
    rel += ["on the"]

    if a['pitch'] > 0.:
        rel += ["top"]
    else:
        rel += ["bottom"]

    if abs(a['yaw']) < 0.1 or abs(a['yaw']) > 3.:
        rel += ["front"]
    elif abs(a['yaw']) > 1.:
        rel += ["back"]
    elif a['yaw'] < 0.:
        rel += ["left"]
    else:
        rel += ["right"]

    return " ".join(rel)


def describe_graph(G):
    words = []
    edges = sorted(G.edges(data=True), key=lambda e: e[-1]['dist'])
    for edge in edges:  # G.edges
        a = edge[-1]
        #a = G.edges[edge]
        a = {k: float(v[1:-1]) if type(v) is str else v
             for k, v in a.items()}  # TODO: only needed for dot file
        rel = describe_relation(a, edge[1])
        words += [rel]
    res = " ".join(words).lower()

    return res


def describe_graph_st(G, source, target, omit_target=False):
    res = []
    paths = nx.all_simple_edge_paths(G, source, target, cutoff=2)
    for p in paths:
        words = []
        for edge in p:
            a = G.edges[edge]
            a = {k: float(v[1:-1]) if type(v) is str else v
                 for k, v in a.items()}  # TODO: only needed for dot file

            rel = describe_relation(a, edge[1])
            words += [rel]
            #if not omit_target or (omit_target and edge[1] != target):
            #    words += [edge[1]]
        res += ["".join(words).lower()]

    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('infile')
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    G = read_graph(args.infile)

    #res = describe_graph(G)
    res = describe_graph_st(G, args.source, args.target)
    print(res)
