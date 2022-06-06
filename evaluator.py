import torch
import re


class Eval(object):
    pattern_old = r"[^A-z]*([0-9])\.([A-z]+)\((.*)\)"
    pattern = r"[^A-z]*([0-9])\.([A-z]+)\<(.*)\>"

    def __init__(self):
        self.successes = [{}, {}, {}]
        self.failures = [{}, {}, {}]
        self.accuracies = [{}, {}, {}]

    @staticmethod
    def create_aux_mask(input_ids):
        inputs = torch.tensor(input_ids)
        mask_start = inputs == 27  # <
        mask_end = inputs == 29  # >

        mask = []
        toggle = 0
        for x in mask_start | mask_end:
            mask += [1 if toggle else 0]
            if x:
                toggle = not toggle
        return mask

    @staticmethod
    def proc_instructions(instructions):
        for instr in instructions.split('\n'):
            res = re.match(Eval.pattern, instr)
            if not res:
                res = re.match(Eval.pattern_old, instr)
                if not res:
                    continue
            #i = int(res.group(1))
            cmd = res.group(2)
            args = res.group(3).split(",")
            yield {'action': cmd, 'args': args}

    @staticmethod
    def aux_loss(labels, preds):
        success_acts = 0
        failed_acts = 0
        success_args = 0
        failed_args = 0

        for label, pred in zip(list(labels), list(preds)):
            act_ = pred['action']
            args_ = pred['args']
            expert_act = label['action']
            expert_args = label['args']

            if act_ == expert_act:
                success_acts += 1
            else:
                failed_acts += 1

            if args_ == expert_args:
                success_args += 1
            else:
                failed_args += 1

        acts_loss = failed_acts / (success_acts + failed_acts)
        args_loss = failed_acts / (success_acts + failed_acts)

        return (acts_loss + args_loss) / 2.

    def eval(self, i, labels, preds):
        failed_acts = 0
        failed_args = 0
        labels = list(labels)
        preds = list(preds)

        for label, pred in zip(labels, preds):
            act_ = pred['action']
            args_ = pred['args']
            expert_act = label['action']
            expert_args = label['args']

            if act_ == expert_act:
                act_entry = {expert_act: self.successes[i].get(expert_act, 0) + 1}
                self.successes[i].update(act_entry)
            else:
                failed_acts += 1
                act_entry = {expert_act: self.failures[i].get(expert_act, 0) + 1}
                self.failures[i].update(act_entry)

            if args_ == expert_args:
                args_entry = {expert_act+"_args": self.successes[i].get(expert_act+"_args", 0) + 1}
                self.successes[i].update(args_entry)
            else:
                failed_args += 1
                args_entry = {expert_act+"_args": self.failures[i].get(expert_act+"_args", 0) + 1}
                self.failures[i].update(args_entry)

        if not failed_acts:
            plan_entry = {'0STEPS': self.successes[i].get('0STEPS', 0) + 1}
            self.successes[i].update(plan_entry)
        else:
            plan_entry = {'0STEPS': self.failures[i].get('0STEPS', 0) + 1}
            self.failures[i].update(plan_entry)

        if not failed_args:
            args_entry = {'1ARGS': self.successes[i].get('1ARGS', 0) + 1}
            self.successes[i].update(args_entry)
        else:
            args_entry = {'1ARGS': self.failures[i].get('1ARGS', 0) + 1}
            self.failures[i].update(args_entry)

        if not failed_acts and not failed_args:
            full_entry = {'2PLAN': self.successes[i].get('2PLAN', 0) + 1}
            self.successes[i].update(full_entry)
        else:
            full_entry = {'2PLAN': self.failures[i].get('2PLAN', 0) + 1}
            self.failures[i].update(full_entry)

        for k in self.successes[i]:
            if k not in self.failures[i]:
                self.failures[i][k] = 0
            self.accuracies[i][k] = self.successes[i][k] / (self.successes[i][k] + self.failures[i][k])
        for k in self.failures[i]:
            if k not in self.successes[i]:
                self.successes[i][k] = 0
            self.accuracies[i][k] = self.successes[i][k] / (self.successes[i][k] + self.failures[i][k])

    def print_stats(self, i, savefile=None):
        header = "KEY ACCURACY SUCCESSES FAILURES\n"
        if savefile is None:
            print(header)
            for k in sorted(self.accuracies[i]):
                print(k, self.accuracies[i][k], self.successes[i][k], self.failures[i][k])
        else:
            with open(savefile, 'w') as f:
                f.write(header)
                for k in sorted(self.accuracies[i]):
                    f.write(f"{k} {self.accuracies[i][k]} {self.successes[i][k]} {self.failures[i][k]}\n")
