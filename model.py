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
                                          
    def forward(self, enc_input):

        memory_list = []
        # output: [Batch_size, Depth, Length]
        output = self.embedding(enc_input).transpose_(1, 2)

        for layer in self.layers:
            output = layer(output)  # output: [Batch_size, Depth, Length]
            memory_list.append(output)

        # return a list of memories
        return memory_list


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
                                          
    def forward(self, dec_input, memory_list):
        assert len(memory_list) == len(self.layers)

        # output: [Batch_size, Depth, Length]
        output = self.embedding(dec_input).transpose_(1, 2)

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory_list[idx])

        # output: [Batch_size, Depth, Length]
        return output


class QRRNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, 
                 hidden_size, emb_size, src_vocab_size, tgt_vocab_size):

        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, src_vocab_size)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size,
                               hidden_size, emb_size, tgt_vocab_size)
        self.proj_linear = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, src_input, tgt_input):
        memory_list = self.encoder(src_input)

        # hidden: [Batch_size, Depth, Length]
        hidden = self.decoder(tgt_input, memory_list)

        # reshaping to [Batch_size x Length, Depth]
        # return [Batch_size x Length, Target_vocab_size]
        return hidden, self.proj_linear(hidden.view(-1, hidden.size(1)))