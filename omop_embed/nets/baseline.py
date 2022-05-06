from omop_embed.common import *

class Model (Classifier):

    def __init__ (self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.linear = nn.Linear(args.dim, 2)
        pass

    def forward (self, batch):
        tokens = batch['tokens']
        v = self.embed(tokens)
        mask = batch['mask'].unsqueeze(dim=2)
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
