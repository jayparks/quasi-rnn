import torch
import numpy as np

import gzip
from util import load_dict

# Extra vocabulary symbols
_GO = '_GO_'
EOS = '_EOS'
UNK = '_UNK'
PAD = '_PAD'

extra_tokens = [_GO, EOS, UNK, PAD]

start_token = extra_tokens.index(_GO)	# start_token = 0
end_token = extra_tokens.index(EOS)	    # end_token = 1
unk_token = extra_tokens.index(UNK)     # unk_token = 2
pad_token = extra_tokens.index(PAD)     # pad_token = 3


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def load_inverse_dict(dict_path):
    orig_dict = load_dict(dict_path)
    idict = {}
    for words, idx in orig_dict.iteritems():
        idict[idx] = words
    return idict


def seq2words(seq, inv_target_dict):
    words = []
    for w in seq:
        if w == end_token:
            break
        if w in inv_target_dict:
            words.append(inv_target_dict[w])
        else:
            words.append(UNK)
    return ' '.join(words)


# batch preparation of a given sequence
def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)
    
    x_lengths = torch.LongTensor(lengths_x)
    maxlen_x = torch.max(x_lengths)

    x = torch.ones(batch_size, maxlen_x) * pad_token
    
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths


# batch preparation of a given sequence pair for training
def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y + 1)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)
    
    x_lengths = torch.LongTensor(lengths_x)
    y_lengths = torch.LongTensor(lengths_y) + 1

    maxlen_x = torch.max(x_lengths)
    maxlen_y = torch.max(y_lengths)

    x = torch.ones(batch_size, maxlen_x) * pad_token
    # length + 1 for _GO or EOS token
    y_input = torch.ones(batch_size, maxlen_y) * pad_token
    y_target = torch.ones(batch_size, maxlen_y) * pad_token
    
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        
        # insert _GO token at the beginning of a sequence
        y_input[idx, 0] = start_token
        y_input[idx, 1:1+lengths_y[idx]] = s_y

        # insert EOS token at the end of a sequence
        y_target[idx, :lengths_y[idx]] = s_y
        y_target[idx, lengths_y[idx]] = end_token

    return x, x_lengths, y_input, y_target, y_lengths
