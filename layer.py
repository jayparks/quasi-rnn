import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.autograd import Variable

class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, use_attn=False):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_attn = use_attn
        
        # quasi_conv_layer
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=3*hidden_size, 
                                kernel_size=kernel_size)
        self.conv_linear = nn.Linear(hidden_size, 3*hidden_size)
        self.rnn_linear = nn.Linear(2*hidden_size, hidden_size)

    def _conv_step(self, inputs, keep_len, memory=None):
        # inputs: [batch_size, input_size, length]
        # memory: [batch_size, memory_size]
        if keep_len:
            # TODO: FF.pad(inputs, (self.kernel_size-1, 0,))
            padded = FF.pad(inputs.unsqueeze(2), (self.kernel_size-1, 0, 0, 0))
            inputs = padded.squeeze(2)
        
        gates = self.conv1d(inputs) # gates: [batch_size, 3*hidden_size, length]
        if memory is not None:
            gates = gates + self.conv_linear(memory).unsqueeze(-1) # broadcast memory

        # Z, F, O: [batch_size, hidden_size, length]
        Z, F, O = gates.split(split_size=self.hidden_size, dim=1)
        return Z.tanh(), F.sigmoid(), O.sigmoid()

    def _rnn_step(self, z, f, o, c, attn_memory=None):
        # uses 'fo pooling' at each time step
        # z, f, o, c: [batch_size, hidden_size, 1]
        # attn_memory: [batch_size, memory_size, length']
        c_ = (1 - f) * z if c is None else f * c + (1 - f) * z
        if not self.use_attn: 
            return c_, (o * c_)	# return c_t and h_t

        alpha = FF.softmax(torch.bmm(c_.transpose(1, 2), attn_memory).squeeze(1))	# alpha: [batch_size, length']
        context = torch.sum(alpha.unsqueeze(1) * attn_memory, dim=2)			# context: [batch_size, memory_size]
        h_ = self.rnn_linear(torch.cat([c_.squeeze(-1), context], dim=1)).unsqueeze(-1)
        
        # c_, h_: [batch_size, hidden_size, 1]
        return c_, (o * h_)

    def forward(self, inputs, state=None, memory=None, keep_len=True):
        # inputs: [batch_size, input_size, length]
        # state: [batch_size, hidden_size]
        c = None if state is None else state.unsqueeze(-1) # unsqueeze dim to feed in _rnn_step
        conv_memory, attn_memory =(None, None) if memory is None else memory

        # Z, F, O: [batch_size, hidden_size, length]
        Z, F, O = self._conv_step(inputs, keep_len, conv_memory)
        c_time, h_time = [], []
        for time, (z, f, o) in enumerate(zip(Z.split(1, 2), F.split(1, 2), O.split(1, 2))):
            c, h = self._rnn_step(z, f, o, c, attn_memory)
            c_time.append(c); h_time.append(h)
        
        # return concatenated cell & hidden states: [batch_size, hidden_size, length]
        return torch.cat(c_time, dim=2), torch.cat(h_time, dim=2)
