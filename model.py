import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class QRNNLayer(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, is_decoder,):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        # quasi_conv_layer
        self.conv1d = nn.Conv1d(input_size, 3*hidden_size, kernel_size)
        if is_decoder:
            self.linear = nn.Linear(hidden_size, 3*hidden_size)

    def conv_step(inputs, is_decoder=True, memory=None,)
        # inputs: [Batch_size x Depth x Length]
        # memory: [Batch_size x Depth x Length]
        gates = self.conv1d(inputs)
        if is_decoder:
            final_memory = memory.split(split_size=1, dim=2)[-1].squeeze(2)
            gates = gates + \
                    self.linear(final_memory).unsqueeze(-1).expand_as(gates)
        
        z, f, o = gates.split(split_size=self.hidden_size, dim=1)
        return z.tanh_(), f.sigmoid_(), o.sigmoid_()

    def rnn_step(z, f, o, c, is_decoder, memory):
        # z, f, o, c: [Batch_size x Depth]

        c_ = torch.mul(c, f) + torch.mul(z, 1-f)
        h_ = torch.mul(o, c_)

        if is_decoder:
            

    def forward(self, input, query=None):
        # input: [Batch_size x Channel_dim x Length]

        return output


class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers=1, kernel_size,
                 hidden_size, emb_size, src_vocab_size):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(emb_size, src_vocab_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, is_decoder=False))
                                          
    def forward(self, enc_input):
        output = self.embedding(enc_input)

        for layer in self.layers:
            output = layer(output)
        return output
