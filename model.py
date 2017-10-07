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
        # input_len: [batch_size] Variable(torch.LongTensor)
        # h: [batch_size, emb_size, length]
        h = self.embedding(inputs).transpose(1, 2)

        last_states, hidden_states = [], []
        for layer in self.layers:
            c, h = layer(h, keep_len=True)  # c, h: [batch_size, hidden_size, length]
            
            time = torch.arange(0, h.size(2)).expand_as(h).long()
            # mask to support variable seq lengths
            mask = (input_len.unsqueeze(-1).unsqueeze(-1) > time).float()
            h = h * mask

            # c_last, h_last: [batch_size, hidden_size]           
            c_last = c.transpose(1, 2)[range(len(inputs)), (input_len-1).data,:]
            h_last = h.transpose(1, 2)[range(len(inputs)), (input_len-1).data,:]
            last_states(torch.cat([c_last, h_last], dim=0)) # [batch_size x 2, hidden_size]
            hidden_states.append(h)

        # return a list of the last state and hidden states of each layer
        return last_states, hidden_states


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
                                          
    def forward(self, inputs, init_states, memories, keep_len):
        assert len(self.layers) == len(init_states)
        assert len(self.layers) == len(memories)

        cell_states, hidden_states = [], []

        # h: [batch_size, emb_size, length]
        h = self.embedding(inputs).transpose(1, 2)
        for layer_idx, layer in enumerate(self.layers):
            state = init_states[layer_idx]
            memory = memories[layer_idx]

            c, h = layer(h, state, memory, keep_len)
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

    def decode(self, inputs, init_states, memories, keep_len):
        cell_states, hidden_states = self.decoder(inputs, init_states, 
                                                  memories, keep_len)
        # return:
        # projected hidden_state of the last layer: logit
        #   first reshape it to [batch_size x length, hidden_size]
        #   after projection: [batch_size x length, tgt_vocab_size]
        h_last = hidden_states[-1]

        return cell_states, self.proj_linear(h_last.view(-1, h_last.size(1)))

    def forward(self, enc_inputs, enc_len, dec_inputs):
        # Encode source inputs
        init_states, memories = self.encode(enc_inputs, enc_len)
        
        # logits: [batch_size x length, tgt_vocab_size]
        _, logits = self.decode(dec_inputs, init_states, memories, keep_len=True)

        return logits
