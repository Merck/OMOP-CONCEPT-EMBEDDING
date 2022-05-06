#!/usr/bin/env python3
#coding=utf-8
import sys
import os
from collections import defaultdict
import numpy as np
from datetime import datetime
import importlib
from glob import glob
from tqdm import tqdm
import pickle
import sklearn.metrics
import torch
from omop_embed import conf

class Metrics:
    def __init__ (self, prefix, save_batches = False):
        self.prefix = prefix
        self.metrics = defaultdict(lambda:[])
        self.labels = []
        self.predicts = []
        self.best_auc = 0
        self.best_epoch = 0
        self.last_auc = 0
        self.epoch = 0
        self.trace = []
        self.save_batches = save_batches
        self.batches = defaultdict(lambda: [])
        pass

    def reset (self, epoch):
        self.epoch = epoch
        self.metrics = defaultdict(lambda:[])
        self.labels = []
        self.predicts = []
        self.batches = defaultdict(lambda: [])
        pass

    def update (self, metrics, label, predict, batch=None):
        for k, v in metrics.items():
            self.metrics[k].append(v)
        if not label is None:
            self.labels.extend(list(label))
        if not predict is None:
            self.predicts.extend(list(predict[:, 1]))
        if self.save_batches:
            for k, v in batch.items():
                self.batches[k].append(v)
        pass

    def snapshot (self, path):
        out = {k: np.concatenate(v) for k, v in self.batches.items()}
        out['predict'] = np.array(self.predicts)
        with open(path, 'wb') as f:
            pickle.dump(out, f)

    def report (self):
        names = []
        values = []
        for k, v in self.metrics.items():
            names.append(k)
            values.append(np.mean(np.array(v), axis=0))
        auc = 0
        if len(self.labels) > 0:
            assert len(self.labels) == len(self.predicts)
            auc = sklearn.metrics.roc_auc_score(self.labels, self.predicts)
        self.trace.append([self.epoch, auc] + values)

        self.last_auc = auc
        if (auc > self.best_auc):
            self.best_auc = auc
            self.best_epoch = self.epoch

        metrics = ' '.join(['%s: %.4f' % (n, x) for n, x in zip(names, values)])
        print(f"{self.prefix} {metrics} auc: {auc:.4f} best {self.best_epoch}: {self.best_auc:.4f}")
        pass

    pass

SKIP = set(['patients', 'source', 'raw'])

def make_torch_batch (batch, device):
    #batch['mask'] = (1-batch['mask']).astype(np.bool)
    return {k: torch.tensor(v, device=device) for k, v in batch.items() if not k in SKIP}

def run_epoch (epoch, model, opt, stream, metrics, device):
    stream.reset() 
    metrics.reset(epoch)
    for _ in tqdm(list(range(stream.__len__()))):
        batch_cpu = stream.__next__()
        batch = make_torch_batch(batch_cpu, device)
        batch['epoch'] = epoch
        loss, mm, labels, pred = model.iterate(batch)
        if not opt is None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics.update(mm, labels, pred, batch_cpu)
    metrics.report()

def train (train_stream, val_stream, args): 
    device = torch.device("cpu") # if cuda_condition else "cpu")
    if args.gpu:
        device = torch.device("cuda:0") # if cuda_condition else "cpu")

    # try to find nets
    search_dirs = [('.', 'nets'), (conf.HOME, 'omop_embed.nets')]
    module = None
    for search_dir, module_prefix in search_dirs:
        path = os.path.join(search_dir, 'nets', args.net + '.py')
        if os.path.exists(path):
            module = importlib.import_module('.'.join([module_prefix, args.net]))
            break
    assert not module is None, "net not found"
    model = getattr(module, 'Model')(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TOTAL PARAMETERS: {total_params}")
    print(f"UPPER PARAMETERS: {total_params - args.vocab_size * args.dim}")

    finetune = []

    if hasattr(model, 'finetune'):
        for module in model.finetune:
            finetune.extend(module.parameters())

    if len(finetune) > 0 and args.tune is None:
        print("%d fine-tune layers found, set args.tune to enable fine-tune" % len(finetune))
    
    if False:
        if len(finetune) > 0 and not args.tune is None:
            total = 0
            others = []
            for p in model.parameters():
                total += 1
                fine = False
                for t in finetune:
                    if t is p:
                        fine = True
                        break
                if not fine:
                    others.append(p)
            print("fine tuning %d parameters" % len(finetune))
            assert len(finetune) + len(others) == total

            params = [ {'params': others},
                    {'params': finetune, 'lr': args.learning_rate * args.tune,  'weight_decay': args.weight_decay * args.tune} ]

            optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_metrics = Metrics("train")
    val_metrics = Metrics("val") #, args.snapshot)
    outdir = '%s/%s' % (args.output, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    degrade = 0
    last_auc = 0

    for epoch in range(args.epochs):
        print("epoch %d:" % epoch)
        if epoch == args.freeze and not args.tune is None:
            print("REDUCING LEARNING RATE")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * args.tune, weight_decay=args.weight_decay)
        model.train()
        run_epoch(epoch, model, optimizer, train_stream, train_metrics, device)
        if (not args.save is None) and ((epoch + 1) % args.save == 0):
            output_path = os.path.join(outdir, "model.ep%d" % epoch)
            torch.save(model.cpu(), output_path)
            model.to(device)
            print("epoch %d model Saved on:" % epoch, output_path)

        if not val_stream is None:
            model.eval()
            with torch.no_grad():
                run_epoch(epoch, model, None, val_stream, val_metrics, device)
            #if args.snapshot:
            #    snapshot_path = os.path.join(outdir, 'snapshot.ep%d' % epoch)
            #    print("Saving snapshot to %s" % snapshot_path)
            #    val_metrics.snapshot(snapshot_path)
        
            if val_metrics.last_auc > last_auc:
                degrade = 0
            else:
                degrade = degrade + 1

            if (degrade >= 10 or val_metrics.best_auc > val_metrics.last_auc + 0.05):
                if not getattr(args, "no_early_stop", False):
                    print("Early stop!")
                    break
            last_auc = val_metrics.best_auc
        with open(os.path.join(outdir, 'metrics.pkl'), 'wb') as f:
            pickle.dump((train_metrics.trace, val_metrics.trace), f)
        pass
    return val_metrics.best_auc


'''
Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
