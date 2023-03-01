"""
Copyright 2020 Sirong Huang

Modifications copyright (C) 2023 Andrea Gabrieli

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

def squeeze(arr, axes_to_keep):
    result = []
    for i,sh in enumerate(arr.shape):
        if i in axes_to_keep or sh != 1:
            result.append(sh)

    return arr.reshape(result)

# A simple hook class that returns the input and output of a layer during forward/backward pass
# Reference: https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.features = []
    def hook_fn(self, module, input, output):
        #self.input = input
        self.output = nn.AdaptiveAvgPool2d(1)(output)
        #self.features.append(self.output.cpu().detach().numpy().squeeze())
        self.features.append(squeeze(self.output.cpu().detach().numpy(), axes_to_keep=[0]))
    def close(self):
        self.hook.remove()