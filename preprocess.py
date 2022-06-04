import argparse
import json
from glob import glob
from os import path as osp

from prune_graph import read_graph, prune_graph


class Result(object):
    def __init__(self, goal, task, scene, target, instructions, actions):
        self.goal = goal.strip()
        self.task = task.strip()
        self.scene = int(scene)
        self.target = target
        self.instructions = [i.strip() for i in instructions]
        self.actions = [a.strip() for a in actions]

    def __len__(self):
        if self.actions is None:
            return 0
        return len(self.actions)

    def gen_action_sequence(self, imax=0):
        subgoals = []

        if not self.goal.endswith("."):
            act_seq = self.goal + ":\n"
        else:
            act_seq = self.goal.replace(".", ":\n")

        for i, (instr, act) in enumerate(zip(self.instructions, self.actions)):
            if imax != 0 and i == imax:
                break
            subgoals += [{
                'instruction': instr,
                'action': act}]
            act_seq += f"{i}.{act}\n"
        act_seq += '|'
        return subgoals, act_seq

    def to_json(self, condition=None, imax=0):
        task = {
            'goal': self.goal,
            'type': self.task,
            'scene': self.scene,
            'target': self.target,
            'subgoals': [],
            'instructions': ""  # TODO: rename
        }

        if self.actions is not None:
            task['subgoals'], act_seq = self.gen_action_sequence(imax=imax)
            if condition is not None:
                task['instructions'] = condition.get_knowledge(self).lower() + "\n" + act_seq
            else:
                task['instructions'] = act_seq
        else:
            task['subgoals'] = self.instructions

        return task


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
        yield Result(goal, ttype, scene, target, instrs, acts)


class Domain(object):
    """
    alfred/gen/constants.py
    """
    def __init__(self, file):
        self.G = read_graph(file)

    @staticmethod
    def get_room(id_):
        if id_ in range(1, 31):
            return "Kitchen"
        elif id_ in range(201, 231):
            return "LivingRoom"
        elif id_ in range(301, 331):
            return "Bedroom"
        elif id_ in range(401, 431):
            return "Bathroom"

    def get_knowledge(self, res):
        # source (room) and target (object) can later be infered from image and goal
        source = Domain.get_room(res.scene)
        target = res.target
        return prune_graph(self.G, source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("outfile")
    parser.add_argument("--cond", help="path to graph")
    #parser.add_argument("--aug", type=int, help="no. of augmentation steps", default=0)
    args = parser.parse_args()
    print(args)

    domain = None
    if args.cond:
        domain = Domain(args.cond)

    dpath = osp.join(args.dataset_path, "*/*/traj_data.json")

    tasks = []
    for path in glob(dpath):
        with open(path) as f:
            for res in parse_file(f):
                #for aug in range(len(res)):
                tasks += [res.to_json(condition=domain)]

    with open(args.outfile, 'w') as outfile:
        json.dump(tasks, outfile)
