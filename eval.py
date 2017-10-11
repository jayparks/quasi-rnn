#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import argparse
from layer import QRNNLayer
from model import QRNNModel

import data.data_utils as data_utils
from data.data_utils import fopen
from data.data_utils import load_inv_dict
from data.data_utils import seq2words

from data.data_iterator import TextIterator
from data.data_iterator import prepare_batch

use_cuda = torch.cuda.is_available()

def load_model(config):
    if os.path.exists(config.model_path):
        print 'Reloading model parameters..'
        checkpoint = torch.load(config.model_path)
        model = QRNNModel(QRNNLayer, checkpoint['num_layers'], checkpoint['kernel_size'],
                          checkpoint['hidden_size'], checkpoint['emb_size'], 
                          checkpoint['num_enc_symbols'], checkpoint['num_dec_symbols'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('No such file:[{}]'.format(config.model_path))

    for key in config.__dict__:
        checkpoint[key] = config.__dict__[key]

    return model, checkpoint


def decode(config):

    model, config = load_model(config)
    
    # Load source data to decode
    test_set = TextIterator(source=config['decode_input'],
                            source_dict=config['src_vocab'],
                            batch_size=config['batch_size'],
                            n_words_source=config['num_enc_symbols'],
                            maxlen=None)
    target_inv_dict = load_inv_dict(config['tgt_vocab'])

    if use_cuda:
        print 'Using gpu..'
        model = model.cuda()

    try:
        print 'Decoding starts..'
        fout = fopen(config['decode_output'], 'w')
        for idx, source_seq in enumerate(test_set):
            source, source_len = prepare_batch(source_seq)

            preds_prev = torch.zeros(len(source), config['max_decode_step']).long()
            preds_prev[:,0] += data_utils.start_token
            preds = torch.zeros(len(source), config['max_decode_step']).long()

            if use_cuda:
                source = Variable(source.cuda())
                source_len = Variable(source_len.cuda())
                preds_prev = Variable(preds_prev.cuda())
                preds = preds.cuda()
            else:
                source = Variable(source)
                source_len = Variable(source_len)
                preds_prev = Variable(preds_prev)

            states, memories = model.encode(source, source_len)
            
            for t in xrange(config['max_decode_step']):
                # logits: [batch_size x max_decode_step, tgt_vocab_size]
                _, logits = model.decode(preds_prev, states, memories, keep_len=True)
                # outputs: [batch_size, max_decode_step]
                outputs = torch.max(logits, dim=1)[1].view(config['batch_size'], -1)
                preds[:,t] = outputs[:,t].data
                if t < config['max_decode_step'] - 1:
                    preds_prev[:,t+1] = outputs[:,t]

            for i in xrange(len(preds)):
                fout.write(str(seq2words(preds[i], target_inv_dict)) + '\n')
                fout.flush()

            print '  {}th line decoded'.format(idx * config['batch_size'])
        print 'Decoding terminated'

    except IOError:
        pass
    finally:
        fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Decoding parameters
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--src_vocab', type=str, default=None)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--decode_input', type=str, default=None)
    parser.add_argument('--decode_output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_decode_step', type=int, default=100)
    
    config = parser.parse_args()
    print(config)
    decode(config)
    print('DONE')
