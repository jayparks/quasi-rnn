import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import model

class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, tgt_vocab_size):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(emb_size, tgt_vocab_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers-1 else False
            
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, use_attn=use_attn))
                                          
    def forward(self, dec_input, memory_list):
        assert len(memory) == len(self.layers)

        output = self.embedding(dec_input)

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory_list[idx])
        return output