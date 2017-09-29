import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, src_vocab_size):
        super(Encoder, self).__init__()
        # Initialize source embedding
        self.embedding = nn.Embedding(src_vocab_size, emb_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, use_attn=False))
                                          
    def forward(self, inputs, input_len):
        # output: [batch_size, emb_size, length]
        output = self.embedding(inputs).transpose_(1, 2)

        h_list = []
        for layer in self.layers:
            _, output = layer(output, input_len)  # output: [batch_size, hidden_size, length]
            h_list.append(output)

        # return a list of hidden states of each layer
        return h_list


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Initialize target embedding
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers - 1 else False
            
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, use_attn=use_attn))
                                          
    def forward(self, inputs, input_len, init_states, memory_list):
        assert len(self.layers) == len(init_states)
        assert len(self.layers) == len(memory_list) 

        c_list, h_list = [], []

        # output: [batch_size, emb_size, length]
        output = self.embedding(inputs).transpose_(1, 2)
        
        for layer_idx, layer in enumerate(self.layers):
            state, output = \
                layer(output, input_len, init_states[layer_idx], memory_list[layer_idx])
            c_list.append(state); h_list.append(output)

        # The shape of the each state: [batch_size, hidden_size, length]
        # return lists of cell states and hidden_states
        return c_list, h_list


class QRRNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, 
                 hidden_size, emb_size, src_vocab_size, tgt_vocab_size):
        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, src_vocab_size)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, tgt_vocab_size)
        self.proj_linear = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, inputs):
        return self.encoder(inputs)


    def decode(self, inputs, init_states, memory_list):
        return self.decoder(inputs, init_states, memory_list)


    def forward(self, enc_inputs, dec_init, dec_inputs):
        # Encode source inputs
        memory_list = self.encode(enc_inputs)

        # The shape of the each state: [Batch_size, Depth, Length]
        c_list, h_list = self.decode(dec_inputs, dec_init, memory_list)

        # return:
        # i) cell_state list
        #     it will be used as initial states for decoding
        
        # ii) projected hidden_state of the last layer
        #     first reshape it to [Batch_size x Length, Depth]
        #     after projection: [Batch_size x Length, Target_vocab_size]
        h_last = h_list[-1]

        return c_list, self.proj_linear(h_last.view(-1, h_last.size(1)))
