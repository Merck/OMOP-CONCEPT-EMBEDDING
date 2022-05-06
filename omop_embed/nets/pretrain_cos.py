from omop_embed.common import *

class Model (PretrainCOS):

    def __init__ (self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        #self.linear = nn.Linear(args.dim, args.vocab_size)
        self.linear = nn.Linear(args.dim, args.dim)
        pass

    def forward (self, batch):
        tokens = batch['tokens']
        mask = batch['mask']
        mask = torch.unsqueeze(mask, dim=2)
        v = self.embed(tokens)
        v = v * mask
        v, _ = torch.cummax(v, dim=1)
        v = self.linear(v)
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
