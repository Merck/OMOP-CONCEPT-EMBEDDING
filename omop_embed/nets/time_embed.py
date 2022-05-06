import sys
import math
import numpy as np
from omop_embed.common import *

class TimeEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 1 x 1 x d_model/2
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        self.div_term = nn.Parameter(data=div_term, requires_grad=False)

    def forward(self, batch):
        d_model = self.d_model
        time = batch['time'].unsqueeze(2)
        # batch x seq_len x 1
        phase = time * self.div_term
        sin = torch.sin(phase)
        cos = torch.cos(phase)
        return torch.cat([sin, cos], dim=2)

class Model (Classifier):

    def __init__ (self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.time = TimeEmbedding(args.dim)
        self.tlinear = nn.Linear(args.dim, args.hidden)
        #self.linear1 = nn.Linear(args.dim + args.heads, args.hidden)
        self.linear = nn.Linear(args.dim + args.hidden, 2)
        pass

    def forward (self, batch):
        tokens = batch['tokens']
        batch_size, seq_len = tokens.shape
        embed = self.embed(tokens)  # batch x seq_len x dim
        v = embed
        tv = self.time(batch)   # batch x seq_len x dim
        tv = self.tlinear(tv)   # batch x seq_len x heads
        #tv = F.relu(tv)

        #v = torch.unsqueeze(v, 3) # batch x seq_len x dim x 1
        #tv = torch.unsqueeze(tv, 2)
                                # batch x seq_len x 1 x heads
        #v = v * tv
                                # batch x seq_len x dim x heads
        #v = torch.reshape(v, [batch_size, seq_len, -1])
        #tv *= 0
        v = torch.cat([v, tv], dim=2)
        #v = self.linear(v)
        #v = torch.cat([embed, v], dim=2)
        #v = F.relu(v)
        mask = batch['mask'].unsqueeze(dim=2)
                                # batch x seq_len x 1
        v *= mask
        v, _ = torch.max(v, dim=1)
        v = self.linear(v)
        v = F.log_softmax(v, dim=1)
        return v


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
