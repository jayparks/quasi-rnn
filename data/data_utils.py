import torch

import sys
import json
import cPickle as pkl
import numpy as np
import gzip

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


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_inv_dict(dict_path):
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

