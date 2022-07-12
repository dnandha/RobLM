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


def describe_graph(G):
    words = []
    edges = sorted(G.edges(data=True), key=lambda e: e[-1]['dist'])
    for edge in edges:  # G.edges
        a = edge[-1]
        #a = G.edges[edge]
        a = {k: float(v[1:-1]) if type(v) is str else v
             for k, v in a.items()}  # TODO: only needed for dot file

        rel = []
        if a['dist'] > 5.:
            rel += ["A"]
        elif a['dist'] > 4.:
            rel += ["B"]
        elif a['dist'] > 3.:
            rel += ["C"]
        elif a['dist'] > 2.:
            rel += ["D"]
        elif a['dist'] > 1.:
            rel += ["E"]
        elif a['dist'] < 0.1:
            rel += ["F"]
        elif a['dist'] < 0.5:
            rel += ["G"]
        elif a['dist'] < 1.:
            rel += ["H"]

        if a['pitch'] > 0.:
            rel += ["I"]
        else:
            rel += ["J"]

        if abs(a['yaw']) < 0.1 or abs(a['yaw']) > 3.:
            rel += ["K"]
        elif abs(a['yaw']) > 1.:
            rel += ["L"]
        elif a['yaw'] < 0.:
            rel += ["M"]
        else:
            rel += ["N"]

        words += [":" + "".join(rel) + ":" + edge[1]]
    res = " ".join(words).lower()

    return res


def describe_graph_st(G, source, target, septoken="<SEP>", omit_target=False):
    res = ""
    paths = nx.all_simple_edge_paths(G, source, target, cutoff=2)
    for p in paths:
        words = []
        for edge in p:
            a = G.edges[edge]
            a = {k: float(v[1:-1]) if type(v) is str else v
                 for k, v in a.items()}  # TODO: only needed for dot file

            rel = []
            if a['dist'] > 5.:
                rel += ["A"]
            elif a['dist'] > 4.:
                rel += ["B"]
            elif a['dist'] > 3.:
                rel += ["C"]
            elif a['dist'] > 2.:
                rel += ["D"]
            elif a['dist'] > 1.:
                rel += ["E"]
            elif a['dist'] < 0.1:
                rel += ["F"]
            elif a['dist'] < 0.5:
                rel += ["G"]
            elif a['dist'] < 1.:
                rel += ["H"]

            if a['pitch'] > 0.:
                rel += ["I"]
            else:
                rel += ["J"]

            if abs(a['yaw']) < 0.1 or abs(a['yaw']) > 3.:
                rel += ["K"]
            elif abs(a['yaw']) > 1.:
                rel += ["L"]
            elif a['yaw'] < 0.:
                rel += ["M"]
            else:
                rel += ["N"]

            words += [":" + "".join(rel) + ":" + edge[1]]
            #if not omit_target or (omit_target and edge[1] != target):
            #    words += [edge[1]]
        res += "".join(words).lower()
        res += septoken

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
