import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config,GPT2Model

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
class MAE_GPT2_Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(999999)
        self.gpt2_config = GPT2Config.from_pretrained("gpt2-medium")
        # self.gpt2 = GPT2Model.from_pretrained("gpt2-medium", config=self.gpt2_config)
        self.patch_embed = PatchEmbed(img_size=args.input_size, patch_size=16, in_chans=3, embed_dim=1024)
        # ########## Initialize GPT-2 with random weights ##########
        self.gpt2 = GPT2Model(config=self.gpt2_config)
        
        # Add a new classifier layer
        self.classifier = nn.Linear(self.gpt2_config.n_embd, args.nb_classes)
        self.initialize_weights()
    def initialize_weights(self):
        
        # Initialize patch_embed weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Initialize random GPT-2 weights
        self.gpt2.init_weights()
    def forward(self, x):
        x = self.patch_embed(x)
        gpt2_output = self.gpt2(inputs_embeds=x).last_hidden_state
        logits = self.classifier(gpt2_output[:, -1, :])
        return logits
