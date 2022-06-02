import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    G = nx.Graph()

    df = pd.read_csv(args.infile)
    print(df['Scenes'])

    rooms = []
    for index, row in df.iterrows():
        obj = row['ObjectType']
        if obj == 'TargetCircle' or obj.endswith('*'):
            continue
        G.add_node(obj)

        scenes = row['Scenes']
        scenes = [s.split('(')[0].strip().replace(' ', '')[:-1] for s in scenes.split('\n')]
        for s in scenes:
            G.add_node(s)
            G.add_edge(obj, s)

        receps = row['Default Compatible Receptacles']
        if type(receps) is str:
            receps = [r.strip() for r in receps.split(',')]
            for r in receps:
                G.add_node(r)
                G.add_edge(obj, r)
                for s in scenes:
                    G.add_edge(s, r)

    nx.draw(G, graphviz_layout(G))
    write_dot(G, args.outfile)
