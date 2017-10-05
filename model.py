import torch
import torch.nn as nn
from torch.autograd import Variable

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
        # input_len: [batch_size] torch.LongTensor
        # output: [batch_size, emb_size, length]
        output = self.embedding(inputs).transpose(1, 2)

        hidden_states = []
        for layer in self.layers:
            _, h = layer(output, input_len)  # h: [batch_size, hidden_size, length]
            # h_last: [batch_size, hidden_size]
            h_last = h.transpose(1, 2)[torch.arange(0, len(input_len)).long(), input_len]
            hidden_states.append((h_last, h))
        # return a list of the last state and hidden states of each layer
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Initialize target embedding
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)

        self.layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers-1 else False
            
            self.layers.append(qrnn_layer(
                input_size, hidden_size, kernel_size, use_attn=use_attn))
                                          
    def forward(self, inputs, input_len, states, memory):
        assert len(self.layers) == len(init_states)
        assert len(self.layers) == len(memory_list) 

        cell_states, hidden_states = [], []

        # output: [batch_size, emb_size, length]
        output = self.embedding(inputs).transpose(1, 2)
        for layer_idx, layer in enumerate(self.layers):
            c, h = layer(output, input_len, states[layer_idx], memory[layer_idx])
            cell_states.append(c); hidden_states.append(h)

        # The shape of the each state: [batch_size, hidden_size, length]
        # return lists of cell states and hidden_states
        return cell_states, hidden_states


class QRRNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, 
                 hidden_size, emb_size, src_vocab_size, tgt_vocab_size):
        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, src_vocab_size)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, tgt_vocab_size)
        self.proj_linear = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, inputs, input_len):
        return self.encoder(inputs, input_len)

    def decode(self, inputs, input_len, init_states, memory):
        return self.decoder(inputs, input_len, init_states, memory)

    # TODO: fix
    def forward(self, enc_inputs, enc_len, dec_inputs, dec_len):
        # Encode source inputs
        memory = self.encode(enc_inputs, enc_len)

        # The shape of the each state: [batch_size, hidden_size, length]
        _, hidden_states = self.decode(dec_inputs, dec_len, memory=memory)

        # return:
        # projected hidden_state of the last layer: logit
        #   first reshape it to [batch_size x length, hidden_size]
        #   after projection: [batch_size x length, tgt_vocab_size]
        h_last = hidden_states[-1]

        return self.proj_linear(h_last.view(-1, h_last.size(1)))
