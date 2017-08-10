import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, use_memory, use_attn):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_attn = use_attn
        
        # quasi_conv_layer
        self.conv1d = nn.Conv1d(input_size, 3*hidden_size, kernel_size)
        if use_memory:
            self.conv_linear = nn.Linear(hidden_size, 3*hidden_size)
            self.rnn_linear = nn.Linear(2*hidden_size, hidden_size)

    def conv_step(inputs, memory=None)
        # inputs: [Batch_size x Depth x Length]
        # memory: [Batch_size x Depth x Length]
        gates = self.conv1d(inputs) # gates: [Batch_size x 3*Depth x Length]
        if memory:
            final_memory = memory.split(split_size=1, dim=2)[-1].squeeze(2)
            gates = gates + \
                    self.conv_linear(final_memory).unsqueeze(-1).expand_as(gates)
 
        z, f, o = gates.split(split_size=self.hidden_size, dim=1)
        return z.tanh_(), f.sigmoid_(), o.sigmoid_()

    def rnn_step(z, f, o, c, memory=None, use_attn=False):
        # uses 'fo pooling'
        # z, f, o, c: [Batch_size x Depth x 1]
        # memory: [Batch_size x Depth X Length]
        c_ = torch.mul(c, f) + torch.mul(z, 1-f)
        
        if not use_attn:
            return c_, torch.mul(c_, o)	# return c_t and h_t

        alpha = torch.bmm(c_.transpose(1, 2), memory)	# alpha: [Batch_size x 1 x Length] # TODO: softmax
        context = torch.sum(alpha * memory, 2)		# context: [Batch_size x Depth]
        h_ = torch.mul(o, self.rnn_linear(torch.concat(c_, context)))
            
        return c_, h_

    def forward(self, inputs, memory=None):
        # input: [Batch_size x Depth x Length] 
        Z, F, O = self.conv_step(inputs, memory)
        
        # initialize c to zero
        c = torch.zeros(inputs.size()[:2])
        hidden_states = []
        for z, f, o in zip(Z.split(1, 2), F.split(1, 2), O.split(1, 2)):
            c, h = self.rnn_step(z, f, o, c, memory, self.use_attn):
            hidden_states.append(h)

        # return a list of hidden states
        return hidden_states


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
