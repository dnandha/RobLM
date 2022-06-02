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


pattern = r"[^A-z]*([0-9])\.([A-z]+)\((.*)\)"

successes = {}
failures = {}
accuracies = {}


def proc_instructions(instructions):
    for instr in instructions.split('\n'):
        res = re.match(pattern, instr)
        if not res:
            continue
        #i = int(res.group(1))
        cmd = res.group(2)
        args = res.group(3).split(",")
        yield {'action': cmd, 'args': args}
    return ()


def proc_eval(labels, preds):
    failed_acts = 0
    failed_args = 0
    labels = list(labels)
    preds = list(preds)

    for label, pred in zip(labels, preds):
        act_ = pred['action']
        args_ = pred['args']
        expert_act = label['action']
        expert_args = label['args']

        act_entry = {expert_act: successes.get(expert_act, 0) + 1}
        args_entry = {expert_act+"_args": successes.get(expert_act+"_args", 0) + 1}

        if act_ == expert_act:
            successes.update(act_entry)
        else:
            failed_acts += 1
            failures.update(act_entry)

        if args_ == expert_args:
            successes.update(args_entry)
        else:
            failed_args += 1
            failures.update(args_entry)

    plan_entry = {'PLAN': successes.get('PLAN', 0) + 1}
    args_entry = {'ARGS': successes.get('ARGS', 0) + 1}
    full_entry = {'FULL': successes.get('FULL', 0) + 1}

    if not failed_acts:
        successes.update(plan_entry)
    else:
        failures.update(plan_entry)

    if not failed_args:
        successes.update(args_entry)
    else:
        failures.update(args_entry)

    if not failed_acts and not failed_args:
        successes.update(full_entry)
    else:
        failures.update(full_entry)

    for k in failures:
        if k not in successes:
            continue
        accuracies[k] = successes[k] / (successes[k] + failures[k])


def print_stats(savefile=None):
    header = "KEY ACCURACY SUCCESSES FAILURES"
    if savefile is None:
        print(header)
        for k in sorted(accuracies):
            print(k, accuracies[k], successes[k], failures[k])
    else:
        with open(savefile, 'w') as f:
            f.write(header)
            for k in sorted(accuracies):
                f.write(f"{k} {accuracies[k]} {successes[k]} {failures[k]}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='specify train dataset json')
    parser.add_argument('--eval', help='specify valid dataset json')
    parser.add_argument('--gen', help='specify test dataset json')
    parser.add_argument('--chkpt_path', default="checkpoints/model.pt")
    parser.add_argument('--prompt')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.eos_token = '|'
    tokenizer.pad_token = tokenizer.eos_token

    #model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    named_layers = dict(model.named_modules())

    if args.train:
        def tokenize(e):
            #e = tokenizer(e['goal'], truncation=True, padding='max_length')#, return_tensors='pt')
            e = tokenizer(e['instructions'], truncation=True, padding='max_length')#, return_tensors='pt')
            #e['labels'] = tokenizer(e['instructions'], truncation=True, padding='max_length', return_tensors='pt')['input_ids']
            e['labels'] = e['input_ids']
            return e

        model.to(device)

        df = pd.read_json(args.train)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=8)
        #ds = ds.map(tokenize)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dl = DataLoader(ds, shuffle=True, batch_size=2)

        num_epochs = 3
        numing_steps = num_epochs * len(dl)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, numing_steps=numing_steps
        )

        torch.save(model.state_dict(), args.chkpt_path)
        model.train()

        writer = SummaryWriter('runs/test1')
        progress = tqdm(range(numing_steps))
        for epoch in range(num_epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                writer.add_scalar('loss/train', loss.item(), progress.n)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress.update(1)

            torch.save(model.state_dict(), args.chkpt_path)
    elif args.eval:
        def tokenize(e):
            f = tokenizer(e['goal'].replace(".",":"))
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

            outputs = model.generate(batch['input_ids'].to(device), do_sample=False, max_length=200)
            #preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            #outputs = model(batch['input_ids'].to(device))
            #preds = torch.argmax(outputs.logits, dim=-1).squeeze()

            labels = batch['labels'].squeeze()
            preds = outputs.squeeze()#[:len(labels)]
            #labels = batch['instructions'][0][:-1]

            label_text = tokenizer.decode(labels, skip_special_tokens=True)
            pred_text = tokenizer.decode(preds, skip_special_tokens=True)

            proc_eval(proc_instructions(label_text), proc_instructions(pred_text))

            #import pdb; pdb.set_trace()

            score = metric.add_batch(predictions=preds[:len(labels)], references=labels)
            #for l, p in zip(batch['labels'], preds):
            #    metric.add_batch(predictions=p, references=l)
            progress.update(1)

            if progress.n % 20 == 0:
                print_stats()
        print_stats(savefile=args.eval+".results.txt")

        score = metric.compute()
        print(score)
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
            #outputs = model(batch['input_ids'])
            #preds = torch.argmax(outputs.logits, dim=-1)
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
