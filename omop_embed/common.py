import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier (nn.Module):

    def __init__ (self):
        super().__init__()
        self.finetune = []
        pass

    def finetune (self):
        return []

    def iterate (self, batch):
        old_batch = batch
        pred = self(batch)
        labels = batch['labels']
        nll = F.nll_loss(pred, labels)
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        l2 = None
        for params in self.parameters():
            if l2 is None:
                l2 = torch.norm(params)
            else:
                l2 += torch.norm(params)
        metrics = {
                'loss': nll.item(),
                'l2': l2.item()
            }
        return nll, metrics, labels, pred

class Pretrain (nn.Module):

    def __init__ (self):
        super().__init__()
        self.finetune = []
        self.cos = nn.CosineSimilarity(2)
        pass

    def iterate (self, batch):
        pred = self(batch)
        labels = batch['pretrain_labels']
        control = batch['pretrain_control']
        mask = batch['pretrain_mask']
        gaps = batch['pretrain_gaps']
        labels = self.embed(labels)
        control = self.embed(control)

        batch_size, seq_len = mask.shape

        # pred      batch x seq_len x D
        # mask      batch x seq_len
        # gaps      batch x seq_len
        # labels    batch x seq_len x dim
        # control   batch x seq_len x dim

        # labels    batch x seq_len x dim
        #pred = F.normalize(pred, dim=2)
        #labels = F.normalize(labels, dim=2)
        #control = F.normalize(control, dim=2)

        # min |pred - labels|
        # max |pred + control|

        # max pred . label
        # min pred

        if False:
            A = self.cos(pred, labels)       # batch x seq_len
            B = self.cos(pred, control)

            loss = torch.sum((B - A) * mask)

            # mask: batch x seq_len
            
            met = loss / (torch.sum(mask) + 1)
            name = 'cos'
            labels = None
            pred = None
        else:
            # pred, labe, contro:   batch x seq x dim
            A = torch.sum(pred * labels, dim=2, keepdim=True)
            B = torch.sum(pred * control, dim=2, keepdim=True)
            # A, B: batch x seq x 1
            nl_soft = -F.log_softmax(torch.cat([A, B], dim=2), dim=2)
            loss = torch.sum(nl_soft[:, :, 0] * mask)
            name = 'nll'
            met = loss / (torch.sum(mask) + 1)

            # - nl_soft  = log_softmax( ..)
            # exp(-nl_soft) = softmax(...)

            pred = torch.exp(-torch.masked_select(nl_soft[:, :, 0], mask > 0))
            pred = pred.unsqueeze(1)
            pred = torch.cat([pred, 1.0-pred], dim=1).detach().cpu().numpy()
            labels = np.zeros_like(pred[:, 0], dtype=np.int)
            # we cannot feed all label 1 to AUC
            # so we switch the first example
            if pred.shape[0] > 0:
                pred[0, 0], pred[0, 1] = pred[0, 1], pred[0, 0]
                labels[0] = 1 - labels[0]


        l2 = None
        for params in self.parameters():
            if l2 is None:
                l2 = torch.norm(params)
            else:
                l2 += torch.norm(params)
        metrics = {
                'loss': loss.item(),
                name: met.item(),
                'l2': l2.item()
            }
        return loss, metrics, labels, pred


class PretrainCOS (nn.Module):

    def __init__ (self):
        super().__init__()
        self.finetune = []
        self.cos = nn.CosineSimilarity(2)
        pass

    def iterate (self, batch):
        pred = self(batch)
        labels = batch['pretrain_labels']
        control = batch['pretrain_control']
        mask = batch['pretrain_mask']
        gaps = batch['pretrain_gaps']
        labels = self.embed(labels)
        control = self.embed(control)

        batch_size, seq_len = mask.shape

        A = self.cos(pred, labels)       # batch x seq_len
        B = self.cos(pred, control)

        loss = torch.sum((B - A) * mask)

        
        met = loss / (torch.sum(mask) + 1)
        name = 'cos'
        labels = None
        pred = None

        l2 = None
        for params in self.parameters():
            if l2 is None:
                l2 = torch.norm(params)
            else:
                l2 += torch.norm(params)
        metrics = {
                'loss': loss.item(),
                name: met.item(),
                'l2': l2.item()
            }
        return loss, metrics, labels, pred

class PretrainMLM (nn.Module):

    def __init__ (self):
        super().__init__()
        self.finetune = []
        pass

    def iterate (self, batch):
        pred = self(batch)                  # batch x seqlen x vocab_size
        batch_size, seq_len, vocab_size = pred.shape
        pred = torch.reshape(pred, (batch_size * seq_len, vocab_size))
        
        pred = F.log_softmax(pred, dim=-1)
        labels = batch['pretrain_labels']   # batch x seqlen
        labels = torch.reshape(labels, (batch_size * seq_len,))
        mask = batch['pretrain_mask']       # batch x seqlen
        mask = torch.reshape(mask, (batch_size * seq_len,))

        nll = F.nll_loss(pred, labels, reduction="none")
        loss = torch.sum(nll * mask) / (torch.sum(mask) + 1)

        l2 = None
        for params in self.parameters():
            if l2 is None:
                l2 = torch.norm(params)
            else:
                l2 += torch.norm(params)
        metrics = {
                'loss': loss.item(),
                'l2': l2.item()
            }
        return loss, metrics, None, None


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
