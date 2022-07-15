import numpy as np
import json
import os.path as osp
from pddlstream.language.constants import print_solution
from pddlstream.algorithms.search import solve_from_pddl

from evaluator import Eval


def clean_pddl_acts(txt):
    print(txt)
    tasks = [
                "gotolocation",
                "putobject",
                "pickupobject",
                "sliceobject",
                "heatobject",
                "coolobject",
                "cleanobject",
                "toggleobject",
            ]
    for t in tasks:
        if txt.startswith(t):
            return t

def clean_pddl_args(txt):
    replace = {
            "_minus_": "-",
            "_plus_": "+",
            "_bar_": "|",
            "_comma_": ",",
            "_dot_": ".",
    }
    for r in replace:
        txt = txt.replace(r, replace[r])
        txt = txt.replace("basin", "")

    return txt


def get_pddl_args(act, args):
    if act == "gotolocation":
        return [args[-1]]
    elif act == "pickupobject":
        return [args[2]]
    elif act == "putobject":
        return [args[-2], args[-1]]
    elif act == "sliceobject":
        return [args[-2]]
    elif act == "heatobject":
        return [args[-2]]
    elif act == "coolobject":
        return [args[-2]]
    elif act == "cleanobject":
        return [args[-2]]
    elif act == "toggleobject":
        return [args[-1]]


def split_pddl_args(args):
    r_args = []
    r_coords = []
    for arg in args:
        a, c = arg.split("_")
        c = c.split(",")
        c = list(map(lambda x: np.round(float(x)/4, 2), c))
        c = np.array([c[0], c[4], c[2]])
        r_args += [a]
        r_coords += [c]
    return r_args, r_coords


def split_expert_args(args):
    r_args = []
    r_coords = []
    for arg in args:
        s = arg.split("|")
        r_args += [s[0]]
        r_coords += [np.array(s[1:4], dtype=float)]
    return r_args, r_coords

def load_domain_pddl(path):
    with open(path) as f:
        domain_pddl = f.read()
        domain_pddl = domain_pddl.replace("totalCost", "total-cost")
    return domain_pddl


def load_problem_pddl(path):
    with open(path) as f:
        problem_pddl = f.read()
        problem_pddl = problem_pddl.replace("(:metric minimize (totalCost))\n", "")
        problem_pddl = problem_pddl.replace("totalCost", "total-cost")

    return problem_pddl


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("domain_pddl")
    parser.add_argument("json_file")
    pargs = parser.parse_args()

    domain = load_domain_pddl(pargs.domain_pddl)

    eval_ = Eval()

    with open(pargs.json_file) as f:
        for sample in json.load(f):
            path = sample['path']
            path = osp.join(osp.dirname(path), "problem_0.pddl")
            print("Loading PDDL problem:", path)
    
            problem = load_problem_pddl(path)
            plan, cost = solve_from_pddl(domain, problem)

            for step, sg in zip(plan, sample['subgoals']):
                act = clean_pddl_acts(step.name)
                args = tuple(map(clean_pddl_args, step.args))
                print(act, args)
                args = get_pddl_args(act, args)
                print(act, args)
                coords = None
                if act != "gotolocation":
                    args, coords = split_pddl_args(args)

                pred = {
                        'action': act,
                        'args': args,
                        'coords': coords,
                        }

                expert = sg['p_action']
                label = list(Eval.proc_instructions(expert, pattern=r"([A-z]+)\(([^\)]*)\)"))

                if len(label) == 0:
                    raise Exception(expert)

                label = label[0]
                label['action'] = label['action'].lower()
                label['coords'] = None

                if label['action'] != "gotolocation":
                    label['args'], label['coords'] = split_expert_args(label['args'])
                
                print(label)
                print(pred)
                eval_.eval(0, [label], [pred])
                eval_.print_stats(0)

    eval_.print_stats(0, savefile=pargs.json_file+"results.txt")
