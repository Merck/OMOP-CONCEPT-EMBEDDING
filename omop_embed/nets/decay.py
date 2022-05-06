import sys
import numpy as np
from omop_embed.common import *

NNN = 10

class Model (Classifier):

    def __init__ (self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        #scales = [0, 1.0/30, 1.0/180, 1.0/360]
        self.linear = nn.Linear(args.dim * 3, 2)
        self.scale = nn.Parameter(data=torch.tensor([[[0.0, 0.01, 0.02]]], dtype=torch.float), requires_grad=False)
        #self.scale = nn.Parameter(data=torch.Tensor(1,1,scales-1), requires_grad=True)
        pass

    def forward (self, batch):
        #ll = batch['labels'].detach().cpu().numpy()
        #xx = batch['time'].detach().cpu().numpy()
        #for i in range(ll.shape[0]):
        #    print('xxx', ll[i], ','.join([str(x) for x in xx[i]]))
        #sys.exit(0)
        tokens = batch['tokens']
        B = tokens.shape[0]
        v = self.embed(tokens).unsqueeze(dim=2) # batch x seq x 1 x dim
        time = batch['time'].unsqueeze(dim=2)   # batch x seq x 1
                                        # scale : 1       1     n
        #xscale = torch.cat([self.fixed, self.scale], dim=2)
        decay = torch.exp(time * self.scale)    # batch x seq x n
        mask = batch['mask'].unsqueeze(dim=2)   # batch x seq x 1
        weight = (decay * mask).unsqueeze(3)    # batch x seq x n x 1
        v = v * weight              # batch x seq x n x dim
        v, _ = torch.max(v, dim=1)  # batch x n x dim
        #v = torch.sum(v, dim=1)     # batch x dim
        v = torch.reshape(v, [B, -1])
        v = self.linear(v)
        v = F.log_softmax(v, dim=1)
        return v

    def iterateX (self, batch):
        nll, metrics, labels, pred = super().iterate(batch)
        scale = self.scale.detach().cpu().numpy()[0, 0]
        for i in range(scale.shape[0]):
            metrics['s%d' %i] = scale[i]
        return nll, metrics, labels, pred
        


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
