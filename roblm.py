import os
import re
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_scheduler
from tensorboardX import SummaryWriter


pattern_old = r"[^A-z]*([0-9])\.([A-z]+)\((.*)\)"
pattern = r"[^A-z]*([0-9])\.([A-z]+)\<(.*)\>"

successes = [{}, {}, {}]
failures = [{}, {}, {}]
accuracies = [{}, {}, {}]


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


def proc_instructions(instructions):
    for instr in instructions.split('\n'):
        res = re.match(pattern, instr)
        if not res:
            res = re.match(pattern_old, instr)
            if not res:
                continue
        #i = int(res.group(1))
        cmd = res.group(2)
        args = res.group(3).split(",")
        yield {'action': cmd, 'args': args}


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


def proc_eval(i, labels, preds):
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
            act_entry = {expert_act: successes[i].get(expert_act, 0) + 1}
            successes[i].update(act_entry)
        else:
            failed_acts += 1
            act_entry = {expert_act: failures[i].get(expert_act, 0) + 1}
            failures[i].update(act_entry)

        if args_ == expert_args:
            args_entry = {expert_act+"_args": successes[i].get(expert_act+"_args", 0) + 1}
            successes[i].update(args_entry)
        else:
            failed_args += 1
            args_entry = {expert_act+"_args": failures[i].get(expert_act+"_args", 0) + 1}
            failures[i].update(args_entry)

    if not failed_acts:
        plan_entry = {'0STEPS': successes[i].get('0STEPS', 0) + 1}
        successes[i].update(plan_entry)
    else:
        plan_entry = {'0STEPS': failures[i].get('0STEPS', 0) + 1}
        failures[i].update(plan_entry)

    if not failed_args:
        args_entry = {'1ARGS': successes[i].get('1ARGS', 0) + 1}
        successes[i].update(args_entry)
    else:
        args_entry = {'1ARGS': failures[i].get('1ARGS', 0) + 1}
        failures[i].update(args_entry)

    if not failed_acts and not failed_args:
        full_entry = {'2PLAN': successes[i].get('2PLAN', 0) + 1}
        successes[i].update(full_entry)
    else:
        full_entry = {'2PLAN': failures[i].get('2PLAN', 0) + 1}
        failures[i].update(full_entry)

    for k in successes[i]:
        if k not in failures[i]:
            failures[i][k] = 0
        accuracies[i][k] = successes[i][k] / (successes[i][k] + failures[i][k])
    for k in failures[i]:
        if k not in successes[i]:
            successes[i][k] = 0
        accuracies[i][k] = successes[i][k] / (successes[i][k] + failures[i][k])


def print_stats(i, savefile=None):
    header = "KEY ACCURACY SUCCESSES FAILURES\n"
    if savefile is None:
        print(header)
        for k in sorted(accuracies[i]):
            print(k, accuracies[i][k], successes[i][k], failures[i][k])
    else:
        with open(savefile, 'w') as f:
            f.write(header)
            for k in sorted(accuracies[i]):
                f.write(f"{k} {accuracies[i][k]} {successes[i][k]} {failures[i][k]}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='specify train dataset json')
    parser.add_argument('--log', help='logfile for tensorboard')
    parser.add_argument('--eval', help='specify valid dataset json')
    parser.add_argument('--eval_topk', type=int, help='use topk sampling')
    parser.add_argument('--gen', help='specify test dataset json')
    parser.add_argument('--chkpt_path', help='model to load', default="checkpoints/model.pt")
    parser.add_argument('--model_path', help='save path for model', default="checkpoints/model.pt")
    parser.add_argument('--prompt')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.eos_token = '|'
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.additional_special_tokens = ["<", ">"]

    #model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    named_layers = dict(model.named_modules())

    if args.train:
        def tokenize(e):
            #e = tokenizer(e['goal'], truncation=True, padding='max_length')#, return_tensors='pt')
            f = tokenizer(e['instructions'], truncation=True, padding='max_length')#, return_tensors='pt')
            #f['attention_mask_aux'] = create_aux_mask(f['input_ids'])
            f['labels'] = f['input_ids']
            return f

        model.to(device)
        if os.path.isfile(args.chkpt_path):
            print("Restoring checkpoint:", args.chkpt_path)
            model.load_state_dict(torch.load(args.chkpt_path))

        df = pd.read_json(args.train)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=8)
        #ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'attention_mask_aux', 'labels'])
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dl = DataLoader(ds, shuffle=True, batch_size=2)

        num_epochs = 2
        num_training_steps = num_epochs * len(dl)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        model.train()

        writer = SummaryWriter(args.log)
        progress = tqdm(range(num_training_steps))
        for epoch in range(num_epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs_lm = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

                # simply modify the attention mask such that it focuses on the arguments -> most important for successful plan
                #outputs_aux = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask_aux'], labels=batch['labels'])

                loss = outputs_lm.loss # + outputs_aux.loss

                #import pdb; pdb.set_trace()
                #preds = torch.argmax(outputs.logits, dim=-1)
                #res = tokenizer.batch_decode(preds, skip_special_tokens=True)[0]

                #if args.auxloss:
                #    outputs = model.generate(batch['input_ids'], do_sample=False, max_length=200)
                #    labels_text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                #    preds_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                #    for label, pred in zip(labels_text, preds_text):
                #        aux_loss(proc_instructions(label), proc_instructions(pred))

                loss.backward()
                writer.add_scalar('loss/train', loss.item(), progress.n)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress.update(1)

            torch.save(model.state_dict(), args.model_path)
    elif args.eval:
        def tokenize(e):
            #f = tokenizer(e['goal'].replace(".",":"))
            f = tokenizer(e['instructions'].split(":")[0]+":")  # TODO
            f['labels'] = tokenizer(e['instructions'])['input_ids']
            return f

        model.to(device)
        model.load_state_dict(torch.load(args.chkpt_path))
        model.eval()

        df = pd.read_json(args.eval)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=1)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        #ds.set_format(type='torch', columns=['input_ids'], output_all_columns=True)
        print(ds)

        dl = DataLoader(ds, batch_size=1)

        #metric = load_metric('glue', 'mrpc')
        metric = load_metric('accuracy')
        progress = tqdm(range(len(dl)))
        for batch in dl:
            #batch = {k: v.to(device) for k, v in batch.items()}

            if args.eval_topk:
                outputs = model.generate(batch['input_ids'].to(device), do_sample=True, top_k=10, top_p=0.92, num_return_sequences=3, max_length=200)
            else:
                outputs = model.generate(batch['input_ids'].to(device), do_sample=False, max_length=200)

            label_text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            preds_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = batch['labels'].squeeze()
            label_text = label_text[0]

            for i, (pred, pred_text) in enumerate(zip(outputs, preds_text)):
                proc_eval(i, proc_instructions(label_text), proc_instructions(pred_text))

                #score = metric.add_batch(predictions=pred[:len(labels)], references=labels)
                if progress.n % 20 == 0:
                    print_stats(i)

            progress.update(1)

        for i in range(3):
            print_stats(i, savefile=f"{args.eval}{i}.results.txt")

        #score = metric.compute()
        #print(score)
    elif args.gen:
        def tokenize(e):
            e['input_ids'] = tokenizer(e['goal'].replace(".",":"))['input_ids']
            return e

        model.to(device)
        model.load_state_dict(torch.load(args.chkpt_path))
        model.eval()

        df = pd.read_json(args.gen)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=8)
        ds.set_format(type='torch', columns=['input_ids'])
        print(ds)

        outfile = open(args.gen.replace(".json", "_out.txt"), 'w')
        dl = DataLoader(ds, batch_size=1)
        progress = tqdm(range(len(dl)))
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'])
            outputs = model.generate(batch['input_ids'], do_sample=False, max_length=200)
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            outfile.write(res)
            progress.update(1)
        outfile.close()
    else:
        model.load_state_dict(torch.load(args.chkpt_path))
        model.eval()

        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

        outputs = model.generate(input_ids, do_sample=False, max_length=128)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
