
import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import numpy as np
from layer import QRNNLayer
from model import QRNNModel

use_cuda = torch.cuda.is_available()

def main(config):

	# Load parallel data to train
    print 'Loading training data..'
    train_set = BiTextIterator(source=config.stc_train,
                               target=config.tgt_train,
                               source_dict=config.src_vocab,
                               target_dict=config.tgt_vocab,
                               batch_size=config.batch_size,
                               maxlen=config.max_seq_len,
                               n_words_source=config.num_enc_symbols,
                               n_words_target=config.num_dec_symbols,
                               shuffle_each_epoch=config.shuffle,
                               sort_by_length=config.sort_by_len,
                               maxibatch_size=config.maxi_batches)

    if config.src_valid and config.tgt_valid:
        print 'Loading validation data..'
        valid_set = BiTextIterator(source=config.src_valid,
                                   target=config.tgt_valid,
                                   source_dict=config.src_vocab,
                                   target_dict=config.tgt_vocab,
                                   batch_size=config.batch_size,
                                   maxlen=None,
                                   n_words_source=config.num_enc_symbols,
                                   n_words_target=config.num_dec_symbols)
    else:
        valid_set = None


    qrnn_model = QRNNModel(QRNNLayer, config.num_layers, config.kernel_size,
		                   config.hidden_size, config.emb_size, 
		                   config.num_enc_symbols, config.num_dec_symbols)
	qrnn_model.cuda()

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(qrnn_model.parameters(), lr=config.lr)


	step_time, loss = 0.0, 0.0
    words_seen, sents_seen = 0, 0
    start_time = time.time()

    # Training loop
    print 'Training..'
    for epoch_idx in xrange(config.max_epochs):
        if model.global_epoch_step.eval() >= config.max_epochs:
            print 'Training is already complete.', \
                  'current epoch:{}, max epoch:{}'.format()
            break



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# Data loading parameters
	parser.add_argument('--src_vocab', type=str, default=None)
	parser.add_argument('--tgt_vocab', type=str, default=None)
	parser.add_argument('--src_train', type=str, default=None)
	parser.add_argument('--tgt_train', type=str, default=None)
	parser.add_argument('--src_valid', type=str, default=None)
	parser.add_argument('--tgt_valid', type=str, default=None)

	# Network parameters
	parser.add_argument('--kernel_size', type=int, default=2)
	parser.add_argument('--hidden_size', type=int, default=1024)
	parser.add_argument('--num_layers', type=int, default=2)
	parser.add_argument('--emb_size', type=int, default=500)
	parser.add_argument('--num_enc_symbols', type=int, default=30000)
	parser.add_argument('--num_dec_symbols', type=int, default=30000)
	parser.add_argument('--dropout_rate', type=float, default=0.3)

	# Training parameters
	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--max_grad_norm', type=float, default=1.0)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--max_epochs', type=int, default=10)
	parser.add_argument('--maxi_batches', type=int, default=20)
	parser.add_argument('--max_seq_len', type=int, default=50)
	parser.add_argument('--display_freq', type=int, default=100)
	parser.add_argument('--save_freq', type=int, default=100)
	parser.add_argument('--valid_freq', type=int, default=100)
	parser.add_argument('--model_dir', type=str, default='model/')
	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--sort_by_len', type=bool, default=True)

	config = parser.parse_args()
	print(config)
	main(config)
	print('DONE')