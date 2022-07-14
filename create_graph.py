import json
from os import path as osp
from glob import glob
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout
from networkx.readwrite.gpickle import write_gpickle


def create_knowledge_G(df):
    G = nx.Graph()

    for index, row in df.iterrows():
        obj = row['ObjectType'].lower()
        if obj == 'targetcircle' or obj.endswith('*'):
            continue

        props = row['Actionable Properties']
        if type(props) is str and 'Pickupable' not in props.split(','):
            t = 'recep'
        else:
            t = 'object'
        G.add_node(obj, t=t)

        scenes = row['Scenes']
        scenes = [s.split('(')[0].strip().replace(' ', '')[:-1].lower() for s in scenes.split('\n')]
        for s in scenes:
            G.add_node(s, t='room')
            G.add_edge(obj, s)

        receps = row['Default Compatible Receptacles']
        if type(receps) is str:
            receps = [r.strip().lower() for r in receps.split(',')]
            for r in receps:
                G.add_node(r, t='recep')
                G.add_edge(obj, r)
                for s in scenes:
                    G.add_edge(s, r)
    return G


def metrics(pos, pos_ref):
    dpos = pos - pos_ref
    r = np.linalg.norm(dpos)
    pitch = np.arctan2(dpos[1], dpos[2])
    yaw = np.arctan2(dpos[2], dpos[0])
    return {'dist': r, 'pitch': pitch, 'yaw': yaw}


def create_scene_G_floor(data, KG=None, full_names=False):
    G = nx.Graph()

    pos_ref = np.array([0.0, 1.0, 0.0])
    G.add_node('floor', x=pos_ref[0], y=pos_ref[1], z=pos_ref[2])

    for entry in data:
        if not full_names:
            ent = entry['id'].split('|')[0].lower()
        else:
            ent = entry['id'].lower()
        pos = np.array([
            entry['position']['x'],
            entry['position']['y'],
            entry['position']['z']
        ], dtype=float)
        G.add_node(ent, x=pos[0], y=pos[1], z=pos[2])

        # edges describe relation in polar coordinates
        if KG is not None:
            if ent not in KG.nodes:
                continue

            # connect receptacles with objects
            for n in KG.neighbors(ent):
                if KG.nodes[n]['t'] != 'room' and G.has_node(n):
                    pos_n = np.array([
                        G.nodes[n]['x'],
                        G.nodes[n]['y'],
                        G.nodes[n]['z']
                    ], dtype=float)
                    G.add_edge(ent, n, **metrics(pos, pos_n))

            # connect agent with receptacles
            if KG.nodes[ent]['t'] == 'recep':
                G.add_edge('floor', ent, **metrics(pos, pos_ref))

        else:
            G.add_edge('floor', ent, **metrics(pos, pos_ref))

    # connected isolated nodes with floor
    for n in nx.isolates(G):
        pos_n = np.array([
            G.nodes[n]['x'],
            G.nodes[n]['y'],
            G.nodes[n]['z']
        ], dtype=float)
        G.add_edge('floor', n, **metrics(pos, pos_ref))

    return G


def create_scene_G(data, KG=None):
    G = nx.Graph()

    pos_ref = np.zeros(3)
    #pos_ref = np.array([0.0, 1.0, 0.0])
    #G.add_node('floor', x=pos_ref[0], y=pos_ref[1], z=pos_ref[2])

    for entry in data:
        if entry['id'] == 'agent':
            pos_ref = np.array([
                entry['position']['x'],
                entry['position']['y'],
                entry['position']['z']
            ], dtype=float)
            G.add_node('agent', x=pos_ref[0], y=pos_ref[1], z=pos_ref[2])
        else:
            ent = entry['id'].split('|')[0].lower()
            pos = np.array([
                entry['position']['x'],
                entry['position']['y'],
                entry['position']['z']
            ], dtype=float)
            G.add_node(ent, x=pos[0], y=pos[1], z=pos[2])

            # edges describe relation in polar coordinates
            if KG is not None:
                if ent not in KG.nodes:
                    continue

                # connect receptacles with objects
                for n in KG.neighbors(ent):
                    if KG.nodes[n]['t'] != 'room' and G.has_node(n):
                        pos_n = np.array([
                            G.nodes[n]['x'],
                            G.nodes[n]['y'],
                            G.nodes[n]['z']
                        ], dtype=float)
                        G.add_edge(ent, n, **metrics(pos, pos_n))

                # connect agent with receptacles
                if KG.nodes[ent]['t'] == 'recep':
                    G.add_edge('agent', ent, **metrics(pos, pos_ref))

            else:
                G.add_edge('agent', ent, **metrics(pos, pos_ref))

    # connected isolated nodes with agent
    for n in nx.isolates(G):
        pos_n = np.array([
            G.nodes[n]['x'],
            G.nodes[n]['y'],
            G.nodes[n]['z']
        ], dtype=float)
        G.add_edge('agent', n, **metrics(pos, pos_ref))

    return G


def test_metrics():
    m = [
        metrics(np.array([0, 0, 0]), np.array([0, 0, 0])),  # --> 0 on
        metrics(np.array([0, 1, 0]), np.array([0, 0, 0])),  # --> + roof
        metrics(np.array([0, 0, 0]), np.array([0, 1, 0])),  # --> - floor
        metrics(np.array([1, 0, 0]), np.array([0, 0, 0])),  # --> 0 front
        metrics(np.array([0, 0, 0]), np.array([1, 0, 0])),  # --> -0 front
        metrics(np.array([0, 0, 1]), np.array([0, 0, 0])),  # --> + right
        metrics(np.array([0, 0, 0]), np.array([0, 0, 1])),  # --> - left
    ]
    for x in m:
        print(x)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--kg', help='path to domain knowledge')
    args = parser.parse_args()

    test_metrics()

    KG = None
    if args.kg:
        df = pd.read_csv(args.kg)
        KG = create_knowledge_G(df)
        #nx.draw(KG, graphviz_layout(G))
        #write_dot(KG, args.outfile)

    if osp.isdir(args.infile):
        for path in glob(osp.join(args.infile, "*/*/graphs/000000000.json")):
            parent_dir = osp.dirname(osp.dirname(path))
            with open(path) as f:
                G = create_scene_G(json.load(f), KG=KG)
                #nx.draw(G, graphviz_layout(G))
                #write_dot(G, osp.join(parent_dir, args.outfile))
            write_gpickle(G, osp.join(parent_dir, args.outfile))

        nx.draw(G, graphviz_layout(G))
        write_dot(G, "scene.dot")
    else:
        with open(args.infile) as f:
            G = create_scene_G(json.load(f), KG=KG)
            nx.draw(G, graphviz_layout(G))
            write_dot(G, args.outfile)
