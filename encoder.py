import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import model

class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, src_vocab_size):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(emb_size, src_vocab_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, use_attn=False))
                                          
    def forward(self, enc_input):
        output = self.embedding(enc_input)

        for layer in self.layers:
            output = layer(output)
        return output