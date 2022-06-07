import os
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

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer

from evaluator import Eval


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='specify train dataset json')
    parser.add_argument('--train_rl', help='tune model in RL setting')
    parser.add_argument('--train_epochs', type=int, default=2)
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

        df = pd.read_json(args.train)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=8)
        #ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'attention_mask_aux', 'labels'])
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dl = DataLoader(ds, shuffle=True, batch_size=2)
        num_training_steps = args.train_epochs * len(dl)

        model.to(device)
        if os.path.isfile(args.chkpt_path):
            print("Restoring checkpoint:", args.chkpt_path)
            model.load_state_dict(torch.load(args.chkpt_path))
        model.train()

        optimizer = AdamW(model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        progress = tqdm(range(num_training_steps))
        writer = SummaryWriter(args.log)
        for epoch in range(args.train_epochs):
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
    elif args.train_rl:
        def tokenize(e):
            # queries
            f = tokenizer(e['instructions'].split(":")[0]+":")  # TODO
            # responses
            f['labels'] = tokenizer(e['instructions'].split(":")[1])['input_ids']
            return f

        df = pd.read_json(args.train_rl)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=8)
        ds.set_format(type='torch', columns=['input_ids', 'labels'])

        dl = DataLoader(ds, shuffle=True, batch_size=1)
        num_training_steps = args.train_epochs * len(dl)

        model = GPT2HeadWithValueModel.from_pretrained('gpt2')
        model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')

        model.to(device)
        model_ref.to(device)

        model.load_state_dict(torch.load(args.chkpt_path), strict=False)
        model_ref.load_state_dict(torch.load(args.chkpt_path), strict=False)

        ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
        ppo_trainer = PPOTrainer(model, model_ref, tokenizer, **ppo_config)
        
        progress = tqdm(range(num_training_steps))
        writer = SummaryWriter(args.log)
        for epoch in range(args.train_epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = batch['input_ids']
                outputs = model.generate(input_ids=inputs, do_sample=False, max_length=200)
                #outputs = respond_to_batch(model, batch['input_ids'], txt_len=200)
                preds = outputs[:, inputs.shape[1]:]  # TODO: .generate always includes query.. has to be removed manuall -> use BOS token?
                preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)

                labels = batch['labels']
                labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

                loss = torch.sum(torch.tensor([Eval.aux_loss(Eval.proc_instructions(label), Eval.proc_instructions(pred))
                                        for label, pred in zip(preds_text, labels_text)]))
                rewards = [1. - loss]  # TODO: support other batch size?

                train_stats = ppo_trainer.step(labels, preds, rewards)
                writer.add_scalar('ppo/model_reward', rewards[0])
                writer.add_scalar('ppo/return/mean', train_stats['ppo/returns/mean'][0])
                writer.add_scalar('ppo/return/var', train_stats['ppo/returns/var'][0])

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
        ev = Eval()

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
                ev.eval(i, Eval.proc_instructions(label_text), Eval.proc_instructions(pred_text))

                #score = metric.add_batch(predictions=pred[:len(labels)], references=labels)
                if progress.n % 20 == 0:
                    ev.print_stats(i)

            progress.update(1)

        for i in range(3):
            ev.print_stats(i, savefile=f"{args.eval}{i}.results.txt")

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
