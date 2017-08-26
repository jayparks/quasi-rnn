
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


	zero_pad = torch.zeros(config.batch_size, config.kernel_size-1)
	start_token = torch.ones(config.batch_size, 1) * data_utils.start_token
	input = torch.cat(zero_pad, start_token, dim=1)

	for t in xrange(config.max_decode_step):
		







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