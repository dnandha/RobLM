import pandas as pd
import argparse
import json
from glob import glob
from os import path as osp

from create_graph import create_knowledge_G, create_scene_G, create_scene_G_floor
from describe_graph import read_graph, describe_graph, describe_graph_st


class Result(object):
    def __init__(self, goal, task, scene, target, parent, instructions, actions):
        self.goal = goal.strip()
        self.task = task.strip()
        self.scene = int(scene)
        self.target = target
        self.parent = parent
        self.instructions = [i.strip() for i in instructions]
        self.actions = [a.strip() for a in actions]

    def __len__(self):
        if self.actions is None:
            return 0
        return len(self.actions)

    def gen_action_sequence(self, imax=0):
        subgoals = []

        act_seq = ""
        for i, (instr, act) in enumerate(zip(self.instructions, self.actions)):
            if imax != 0 and i == imax:
                break
            subgoals += [{
                'instruction': instr,
                'action': act}]
            act_tok = act.replace("(", "<").replace(")", ">")
            act_seq += f"{i}.{act_tok}\n"
        return subgoals, act_seq

    def to_json(self, condition=None, imax=0):
        task = {
            'goal': self.goal,
            'type': self.task,
            'scene': self.scene,
            'target': self.target,
            'parent': self.parent,
            'subgoals': [],
            'instructions': "",  # TODO: rename
        }

        if self.actions is not None:
            task['subgoals'], act_seq = self.gen_action_sequence(imax=imax)
            if condition is not None:
                #cond_seq = condition.get_knowledge(self) + "\n"
                target = self.target.lower()
                cond_seq = Domain.get_room(self.scene).lower() + "<SEP>"
                cond_seq += target + "<SEP>"
                #cond_seq += describe_graph(condition)
                #cond_seq += task['subgoals'][0]['instruction']
                cond_seq += describe_graph_st(condition, 'floor', target)
                task['instructions'] = self.goal + "<SEP>" + cond_seq + "<BOS>" + act_seq + "<EOS>"
            else:
                task['instructions'] = act_seq
        else:
            task['subgoals'] = self.instructions

        return task


    def to_txt(self, condition=None, imax=0):
        res = ""
        if self.actions is not None:
            _, act_seq = self.gen_action_sequence(imax=imax)
            if condition is not None:
                #cond_seq = condition.get_knowledge(self) + "\n"
                target = self.target.lower()
                cond_seq = Domain.get_room(self.scene).lower() + "<SEP>"
                cond_seq += target + "<SEP>"
                #cond_seq += describe_graph(condition)
                #cond_seq += describe_graph(condition, 'agent', target)
                cond_seq += describe_graph(condition, 'floor', target)
                #cond_seq += describe_graph(condition, 'floor', 'agent')
                res = self.goal + "<SEP>" + cond_seq + "<BOS>" + act_seq + "<EOS>"
            else:
                res = act_seq

        return res


def parse_file(file_):
    data = json.load(file_)

    plan = None
    if 'plan' in data:
        plan = data['plan']['high_pddl']
    #acts = data['plan']['low_actions']
    anns = data['turk_annotations']['anns']

    ttype = None
    if 'task_type' in data:
        ttype = data['task_type']

    scene = data['scene']['scene_num']
    target = data['pddl_params']['object_target']
    parent = data['pddl_params']['parent_target']

    acts = None
    if plan:
        acts = []
        for p in plan:
            act = p['discrete_action']
            action = "{}({})".format(act['action'], ",".join(act['args']))
            if action != "NoOp()":
                #print(action)
                acts += [action]
    #for act in acts:
    #    print(act['api_action'])
    for ann in anns:
        goal = ann['task_desc']
        instrs = ann['high_descs']

        if acts is not None and len(instrs) != len(acts):
            continue

        #for act, instr in zip(acts, instrs):
            #print(act, instr)
        yield Result(goal, ttype, scene, target, parent, instrs, acts)


class Domain(object):
    """
    alfred/gen/constants.py
    """
    def __init__(self, file):
        self.G = read_graph(file)
        self.floor_objects = {}

    def get_room(id_):
        if id_ in range(1, 31):
            return "Kitchen"
        elif id_ in range(201, 231):
            return "LivingRoom"
        elif id_ in range(301, 331):
            return "Bedroom"
        elif id_ in range(401, 431):
            return "Bathroom"

    def feed_floor_plans(self, floor_plans):
        for i in range(500):
            filename = osp.join(floor_plans, f"FloorPlan{i}-objects.json")
            if osp.isfile(filename):
                with open(filename) as f:
                    self.floor_objects[i] = json.load(f)
            # TODO: if we later want receptacles
            #filename = osp.join(args.floor_plans, f"FloorPlan{i}-openable.json")
            #if osp.isfile(filename):
            #    with open(filename) as f:
            #        floor_recepts[i] = json.load(f)

    def get_knowledge(self, res):
        # source (room) and target (object) can later be infered from image and goal or so
        source = Domain.get_room(res.scene)
        target = res.target
        source, target, nodes = prune_graph(self.G, source, target)
        #print("0. Source node / target node", source, target)
        #print("1. Nodes on path between source and target:", nodes)
        if res.scene is not None:
            #print("2. Provided scene number:", res.scene)
            nodes = [n for n in nodes if n in self.floor_objects[res.scene]]
        #print("3. Remaining nodes after scene specific pruning:", nodes)
        nodes = ",".join(nodes)

        res = f"{source}-{target}=[{nodes}]".lower()

        #print("3. Result:", res)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("outfile")
    parser.add_argument("--split_task", action='store_true')
    parser.add_argument("--txt", action='store_true')
    parser.add_argument("--cond", help="path to graph")
    parser.add_argument("--floor_plans", help="plans for conditioning")
    parser.add_argument("--kg", help="path to knowledge data")
    #parser.add_argument("--aug", type=int, help="no. of augmentation steps", default=0)
    args = parser.parse_args()
    print(args)

    domain = None
    if args.cond:
        domain = Domain(args.cond)
        if args.floor_plans:
            domain.feed_floor_plans(args.floor_plans)

    dpath = osp.join(args.dataset_path, "*/*/traj_data.json")

    KG = None
    if args.kg:
        df = pd.read_csv(args.kg)
        KG = create_knowledge_G(df)

    tasks = {}
    for path in glob(dpath):
        G = None
        path_graph = osp.join(osp.dirname(path), "graphs/000000000.json")
        if osp.isfile(path_graph):
            with open(path_graph) as f:
                #G = create_scene_G(json.load(f), KG=KG)
                G = create_scene_G_floor(json.load(f), KG=KG)

        with open(path) as f:
            for res in parse_file(f):
                #for aug in range(len(res)):
                if res.task not in tasks:
                    tasks[res.task] = []

                if args.txt:
                    tasks[res.task] += [res.to_txt(condition=G)]
                else:
                    tasks[res.task] += [res.to_json(condition=G)]

    if args.split_task:
        for t in tasks:
            with open(f"{t}_{args.outfile}", 'w') as outfile:
                json.dump(tasks[t], outfile)
    else:
        with open(args.outfile, 'w') as outfile:
            json.dump(sum(list(tasks.values()), []), outfile)
