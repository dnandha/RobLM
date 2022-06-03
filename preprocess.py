import argparse
import json
from glob import glob
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument("outfile")
args = parser.parse_args()


class Result(object):
    def __init__(self, goal, task, scene, instructions, actions):
        self.goal = goal
        self.task = task
        self.scene = scene
        self.instructions = instructions
        self.actions = actions

    def to_json(self, condition=False):
        task = {
            'goal': self.goal,
            'type': self.task,
            'scene': self.scene,
            'subgoals': [],
            'instructions': ""
        }

        if self.actions is not None:
            task['instructions'] = ""
            if condition:
                task['instructions'] += f"{self.scene}>"
            task['instructions'] += self.goal.replace(".", ":\n")
            for i, (instr, act) in enumerate(zip(self.instructions, self.actions)):
                task['subgoals'] += [{
                    'instruction': instr,
                    'action': act}]
                task['instructions'] += f"{i}.{act}\n"
            task['instructions'] += '|'
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
        yield Result(goal, ttype, scene, instrs, acts)


max_samples = 100000
i = 0

dpath = osp.join(args.dataset_path, "*/*/traj_data.json")

tasks = []
for path in glob(dpath):
    i += 1
    if i >= max_samples:
        break
    with open(path) as f:
        for res in parse_file(f):
            tasks += [res.to_json(condition=True)]

with open(args.outfile, 'w') as outfile:
    json.dump(tasks, outfile)
