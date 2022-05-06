from omop_embed.common import *

class Model (Classifier):

    def __init__ (self, args):
        super().__init__()
        #pre = args.pre
        if args.pre is None:
            print("Creating new embedding layer")
            self.embed = nn.Embedding(args.vocab_size, args.dim)
            self.embed.weight.data.uniform_(-1, 1)
            self.embed.requires_grad_(False)
            self.dim = args.dim
            last_dim = self.dim
        else:
            pre = torch.load(args.pre)
            self.embed = nn.Embedding.from_pretrained(pre.embed.weight.data, freeze=True)
            last_dim = self.embed.embedding_dim
            self.dim = last_dim
            self.finetune.append(self.embed)
            print("Loaded embedding from %s of dim %d." % (args.pre, last_dim))

        if not args.hidden is None:
            print("Creating hidden layer.")
            self.linear1 = nn.Linear(last_dim, args.hidden)
            last_dim = args.hidden
        else:
            self.linear1 = nn.Identity()


        #self.linear = nn.Linear(D, 2)
        self.linear2 = nn.Linear(last_dim, 2)
        pass

    def forward (self, batch):
        tokens = batch['tokens']
        v = self.embed(tokens)
        v = self.linear1(v)
        mask = batch['mask'].unsqueeze(dim=2)
        v *= mask
        v, _ = torch.max(v, dim=1)
        v = self.linear2(v)
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
