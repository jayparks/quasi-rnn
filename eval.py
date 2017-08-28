
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import math
import time
import argparse
import data_utils
import numpy as np
from layer import QRNNLayer
from model import QRNNModel

import data_utils
from data_utils import prepare_batch
from data_utils import prepare_train_batch
from data_iterator import BiTextIterator

use_cuda = torch.cuda.is_available()


def load_model(config):
    model = QRNNModel(QRNNLayer, config.num_layers, config.kernel_size,
    	              config.hidden_size, config.emb_size, 
    	              config.num_enc_symbols, config.num_dec_symbols)

    # Initialize a training state
    train_state = { 'epoch': 0, 'train_steps': 0, 'state_dict': None }

    model_path = os.path.join(config.model_dir, config.model_name)
    if os.path.exists(model_path):
        print 'Reloading model parameters..'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    else:
    	raise ValueError(
    		'No such file:[{}]'.format(config.model_path))
    return model


def decode(config):
    # Load source data to decode
    test_set = TextIterator(source=config['decode_input'],
                            batch_size=config['batch_size'],
                            source_dict=config['source_vocabulary'],
                            maxlen=None,
                            n_words_source=config['num_encoder_symbols'])    

    # Load inverse dictionary used in decoding
    target_inverse_dict = data_utils.load_inverse_dict(config['target_vocabulary'])

    model = load_model(config)
    if use_cuda:
        model.cuda()

    inputs = torch.ones(config.batch_size, 1) * data_utils.start_token
    try:
        fout = [data_utils.fopen(config.decode_output, 'w')]

        for idx, source_seq in enumerate(test_set):
            source, source_len = prepare_batch(source_seq)
            memory_list = model.encode(source)

            temp = []
            for t in xrange(config.max_decode_step):
                state, logit = model.decode(inputs, state, memory_list)
                inputs = torch.max(logit, dim=1)
                temp.append(inputs.unsqueeze(-1))

            # outputs: [batch_size, max_decode_step]
            outputs = torch.cat(temp, dim=1)
            for seq in outputs:
                fout.write(str(data_utils.seq2words(seq, target_inverse_dict)) + '\n')

            print '  {}th line decoded'.format(idx * config.batch_size)
        print 'Decoding terminated'

    except IOError:
        pass
    finally:
        fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Decoding parameters
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--decode_input', type=str, default=None)
    parser.add_argument('--decode_output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_decode_step', type=int, default=200)
    
    config = parser.parse_args()
    print(config)
    decode(config)
    print('DONE')