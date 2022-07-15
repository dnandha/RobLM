import numpy as np
import torch
import re


class Eval(object):
    pattern = r"([A-z]+)<([^>]*)>"

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
    def proc_instructions(instructions, pattern=None):
        if pattern is None:
            pattern = Eval.pattern
        for res in re.finditer(pattern, instructions):
            #i = int(res.group(1))
            cmd = res.group(1)
            args = res.group(2).split(",")
            yield {'action': cmd, 'args': args}

    @staticmethod
    def aux_loss(labels, preds):
        success_acts = 0
        failed_acts = 0
        success_args = 0
        failed_args = 0

        for label, pred in zip(list(labels), list(preds)):
            act_ = pred['action'].lower()
            args_ = pred['args'].lower()
            expert_act = label['action'].lower()
            expert_args = label['args'].lower()

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
    
    def _add_result(self, i, key, check):
        if check:
            self._add_success(i, key)
        else:
            self._add_failure(i, key)
        return check

    def _add_success(self, i, key):
        entry = {key: self.successes[i].get(key, 0) + 1}
        self.successes[i].update(entry)

    def _add_failure(self, i, key):
        entry = {key: self.failures[i].get(key, 0) + 1}
        self.failures[i].update(entry)

    def eval(self, i, labels, preds):
        labels = list(labels)
        preds = list(preds)

        check_coords = False

        for label, pred in zip(labels, preds):
            failed_acts = 0
            failed_args = 0
            failed_args_obj = 0
            failed_coords = 0

            act_ = pred['action']
            args_ = pred['args']
            expert_act = label['action']
            expert_args = label['args']

            if 'coords' in pred:
                check_coords = True

            res = self._add_result(i, expert_act, act_ == expert_act)
            self._add_result(i, '0STEPS', res)
            if not res:
                failed_acts = 1
                continue

            for arg, expert_arg in zip(args_, expert_args):
                if not check_coords:
                    arg_obj = arg.split(" ")[1]
                    expert_arg_obj = expert_arg.split(" ")[1]

                    res = self._add_result(i, expert_act+"_args_obj", arg_obj == expert_arg_obj)
                    self._add_result(i, '1ARGS_OBJ', res)
                    if not res:
                        failed_args_obj = 1

                res = self._add_result(i, expert_act+"_args", arg == expert_arg)
                self._add_result(i, '1ARGS', res)
                if not res:
                    print("FAIL:", arg, expert_arg)
                    failed_args = 1

            if check_coords:
                coords, expert_coords = pred['coords'], label['coords']
                if coords is None and expert_coords is None:
                    continue

                for c, expert_c in zip(coords, expert_coords):
                    res = self._add_result(i, expert_act+"_coords", np.linalg.norm(c-expert_c) < 0.1)
                    self._add_result(i, '2COORDS', res)
                    if not res:
                        failed_coords = 1

            self._add_result(
                    i,
                    '3PLAN',
                    not (failed_acts or failed_args or failed_args_obj or failed_coords)
                    )

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
