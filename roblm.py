import os
import torch
import pandas as pd
import torch.nn.functional as F
from datasets import Dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.distributions import Categorical
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers import get_scheduler
from tensorboardX import SummaryWriter

#from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
#from trl.ppo import PPOTrainer

from evaluator import Eval
from env import OfflineEnv


def train_tokenizer(tok, df, size=10000):
    def get_corpus(df):
        for name, value in df['instructions'].iteritems():
            yield value
    ds = get_corpus(df)

    tokenizer = tok.train_new_from_iterator(
        ds,
        vocab_size=size,
        new_special_tokens=[
            "<BOS>",
            "<EOS>",
            "<SEP>",
        ]
    )
    return tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='specify train dataset json')
    parser.add_argument('--train_tok', help='tune tokenizer')
    parser.add_argument('--tokdir', help='tokenizer dir', default="")
    parser.add_argument('--train_rl', help='tune model in RL setting')
    parser.add_argument('--train_epochs', help='total epochs to train', type=int, default=2)
    parser.add_argument('--warmstart', help='pre-train LM model without RL', type=int, default=0)
    parser.add_argument('--log', help='logfile for tensorboard')
    parser.add_argument('--eval', help='specify valid dataset json')
    parser.add_argument('--eval_topk', type=int, help='use topk sampling')
    parser.add_argument('--gen', help='specify test dataset json')
    parser.add_argument('--chkpt_path', help='model to load', default="checkpoints/model.pt")
    parser.add_argument('--model_path', help='save path for model', default="checkpoints/model.pt")
    parser.add_argument('--prompt')
    parser.add_argument('--forward')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    named_layers = dict(model.named_modules())

    if not args.train_tok and os.path.isdir(args.tokdir):
        print("Loading custom tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.tokdir)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    #tokenizer.add_special_tokens({'additional_special_tokens': ['<SEP>', '<BOS>', '<EOS>']})
    #tokenizer.sep_token = '<SEP>'
    #tokenizer.bos_token = '<BOS>'
    #tokenizer.eos_token = '<EOS>'
    tokenizer.pad_token = tokenizer.eos_token
    #print(tokenizer.vocab_size)
    #print(tokenizer.tokenize("cil:lightswitch cjl:garbagecan<BOS>0.GotoLocation<countertop>\n1.PickupObject<butterknife>\n2.GotoLocation<apple>\n"))

    if args.train:
        def tokenize(e):
            s = e['instructions'].split('<BOS>')
            f = tokenizer(s[0])  # , truncation=True, padding='max_length')
            f['labels'] = tokenizer('<BOS>' + s[1])['input_ids']
            return f

        df = pd.read_json(args.train)
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, num_proc=1)
        #ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'attention_mask_aux', 'labels'])
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dl = DataLoader(ds, shuffle=True, batch_size=1)
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

        env = OfflineEnv()

        progress = tqdm(range(num_training_steps))
        writer = SummaryWriter(args.log)
        for epoch in range(args.train_epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = batch['input_ids']
                att_mask = batch['attention_mask']
                labels = batch['labels']

                losses = []

                # 0. train LM normally
                # (inputs == labels for LM)
                #in_ = inputs
                #in_ = torch.cat((inputs, labels), dim=1)
                #att_mask = torch.ones_like(in_)
                #outputs_lm = model(input_ids=in_, attention_mask=att_mask, labels=in_)
                #loss = outputs_lm.loss
                #losses += [loss]
                #writer.add_scalar(f'train/lm_loss', loss.item(), progress.n)

                # REINFORCE
                if progress.n >= args.warmstart:
                    # 1. collect trajectory
                    env.reset(labels)
                    S, A, R = [], [], []

                    # instead of state -> next_state we follow expert trajectory
                    # 1. BOS --> ??
                    # 2. BOS GotoLocation -> ??
                    # 3. BOS GotoLocation countertop --> ??
                    # t. BOS GotoLocation countertop ... EOS
                    for i in range(labels.shape[1]):
                        # new state: old state + next expert action
                        state = torch.cat((inputs, labels[:, :i+1]), dim=1)
                        att_mask = torch.ones_like(state)

                        # run inference only
                        with torch.no_grad():
                            outputs_lm = model(input_ids=state, attention_mask=att_mask, labels=state)

                        # instead of argmax we do softmax
                        next_token_probs = F.softmax(outputs_lm.logits[:, -1, :], dim=1)  # dim := [bs, vocab_size]
                        action_probs = next_token_probs
                        # torch equivalent of np.random.choice(x, p)
                        action = Categorical(action_probs).sample()

                        # this doesn't give next state, because we follow expert trajectory
                        done, reward = env.step(action)

                        S += [state]
                        A += [action]
                        R += [reward]

                        # keep going in any case to collect more samples for LM training
                        #if done:
                        #    writer.add_scalar(f'train/eps_len', i, progress.n)
                        #    break

                    # 2. sum discounted future rewards
                    G = torch.tensor(R).cumsum(0).flip(0)

                    # 3. rerun policy with optimization
                    L1 = torch.zeros_like(G, dtype=float)
                    L2 = torch.zeros_like(G, dtype=float)
                    #if torch.sum(G) > 0:
                    #    import pdb; pdb.set_trace()
                    for i, (s, a, g) in enumerate(zip(S, A, G)):
                        outputs_lm = model(input_ids=s, attention_mask=torch.ones_like(s), labels=s)
                        L1[i] = outputs_lm.loss

                        # pick previously chosen action from logits
                        log_prob = outputs_lm.logits[:, -1, a]
                        # and multiply with expected return
                        L2[i] = -torch.mean(log_prob * g)

                    # mean losses
                    L1 = L1.mean()
                    L2 = L2.mean()
                    losses += [L1]
                    losses += [L2]
                    writer.add_scalar('train/lm_loss', L1.item(), progress.n)
                    writer.add_scalar('train/policy_loss', L2.item(), progress.n)

                # LM loss + policy loss
                loss = sum(losses)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress.update(1)

                if progress.n % 5000 == 0:
                    torch.save(model.state_dict(), args.model_path)

            torch.save(model.state_dict(), args.model_path)
    elif args.train_tok:
        df = pd.read_json(args.train_tok)
        tokenizer = train_tokenizer(tokenizer, df)
        tokenizer.save_pretrained(args.tokdir)
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
            #instr = e['goal'].replace(".",":")
            #if not instr.endswith(":"):
            #    instr += ":"
            #instr = e['instructions'].split("<BOS>")[0] + "<BOS>"
            
            # starting with second step
            instr = e['instructions'].split("1.")[0] + "1."

            #####
            #context, goal = instr.split('\n')
            #context = context.split('-')
            #context[1] = context[1].split('=')[0]
            #arg = ','.join(list(Eval.proc_instructions(e['instructions']))[0]['args'])
            #instr = f"{context[0]}-{context[1]}=[{arg}]\n{goal}"
            #####
            f = tokenizer(instr)  # TODO
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
                outputs = model.generate(batch['input_ids'].to(device), do_sample=True, max_length=200)

            label_text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            preds_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            labels = batch['labels'].squeeze()
            label_text = label_text[0]
            label_text = "0." + label_text.split("0.")[1]
            #label_text = label_text.split("<EOS")[0].split("<BOS>")[1]

            for i, (pred, pred_text) in enumerate(zip(outputs, preds_text)):
                pred_text = "0." + pred_text.split("0.")[1]
                print("LBL:", label_text)
                print("PRD:", pred_text)
                ev.eval(i, Eval.proc_instructions(label_text), Eval.proc_instructions(pred_text))

                #score = metric.add_batch(predictions=pred[:len(labels)], references=labels)
                #if progress.n % 20 == 0:
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
    elif args.prompt:
        if os.path.isfile(args.chkpt_path):
            model.load_state_dict(torch.load(args.chkpt_path))
            model.eval()

        print(tokenizer.tokenize(args.prompt))
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        print(input_ids)

        outputs = model.generate(input_ids, do_sample=False, max_length=128)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    elif args.forward:
        if os.path.isfile(args.chkpt_path):
            model.load_state_dict(torch.load(args.chkpt_path))
            model.eval()

        print(tokenizer.tokenize(args.forward))
        input_ids = tokenizer(args.forward, return_tensors="pt").input_ids
        print(input_ids)

        bos_token = 1
        eos_token = 2
        inputs = input_ids
        action = 0
        while not action == eos_token:
            att_mask = torch.ones_like(inputs)

            with torch.no_grad():
                outputs_lm = model(input_ids=inputs, attention_mask=att_mask)

            # instead of argmax we do softmax
            next_token_probs = F.softmax(outputs_lm.logits[:, -1, :], dim=1)  # dim := [bs, vocab_size]
            action_probs = next_token_probs
            # torch equivalent of np.random.choice(x, p)
            action = Categorical(action_probs).sample()
            inputs = torch.cat((inputs, torch.tensor([[action]])), dim=1)
            print(tokenizer.decode(action))
