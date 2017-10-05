import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, src_vocab_size):
        super(Encoder, self).__init__()
        # Initialize source embedding
        self.embedding = nn.Embedding(src_vocab_size, emb_size)
        layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, False))
        self.layers = nn.Sequential(*layers)
                                          
    def forward(self, inputs, input_len):
        # input_len: [batch_size] torch.LongTensor
        # h: [batch_size, emb_size, length]
        h = self.embedding(inputs).transpose(1, 2)

        hidden_states = []
        for layer in self.layers:
            _, h = layer(h, input_len)  # h: [batch_size, hidden_size, length]
            # h_last: [batch_size, hidden_size]
            h_last = h.transpose(1, 2)[range(len(inputs)), (input_len-1).data,:]
            hidden_states.append((h_last, h))

        # return a list of the last state and hidden states of each layer
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Initialize target embedding
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
        layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers-1 else False
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, use_attn))
        self.layers = nn.Sequential(*layers)
                                          
    def forward(self, inputs, input_len, states, memory_tuples):
        if states:
            assert len(self.layers) == len(states)
        if memory_tuples:
            assert len(self.layers) == len(memory_tuples)

        cell_states, hidden_states = [], []

        # h: [batch_size, emb_size, length]
        h = self.embedding(inputs).transpose(1, 2)
        for layer_idx, layer in enumerate(self.layers):
            state = states[layer_idx] if states else None
            memory_tuple = memory_tuples[layer_idx] if memory_tuples else None

            c, h = layer(h, input_len, state, memory_tuple)
            cell_states.append(c); hidden_states.append(h)

        # The shape of the each state: [batch_size, hidden_size, length]
        # return lists of cell states and hidden_states
        return cell_states, hidden_states


class QRNNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, hidden_size,
                 emb_size, src_vocab_size, tgt_vocab_size):
        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               emb_size, src_vocab_size)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               emb_size, tgt_vocab_size)
        self.proj_linear = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, inputs, input_len):
        return self.encoder(inputs, input_len)

    def decode(self, inputs, input_len, init_states, memory_tuples):
        cell_states, hidden_states = self.decoder(inputs, input_len, 
                                                  init_states, memory_tuples)
        # return:
        # projected hidden_state of the last layer: logit
        #   first reshape it to [batch_size x length, hidden_size]
        #   after projection: [batch_size x length, tgt_vocab_size]
        h_last = hidden_states[-1]

        return cell_states, self.proj_linear(h_last.view(-1, h_last.size(1)))

    def forward(self, enc_inputs, enc_len, dec_inputs, dec_len):
        # Encode source inputs
        memory_tuples = self.encode(enc_inputs, enc_len)
        
        # logits: [batch_size x length, tgt_vocab_size]
        _, logits = self.decode(dec_inputs, dec_len, None, 
                                memory_tuples=memory_tuples)

        return logits
