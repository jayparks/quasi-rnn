#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import math
import time
import argparse
import numpy as np
from layer import QRNNLayer
from model import QRNNModel

import data.data_utils as data_utils
from data.data_utils import prepare_batch
from data.data_utils import prepare_train_batch
from data.data_iterator import BiTextIterator

use_cuda = torch.cuda.is_available()

def create_model(config):
    print 'Creating new model parameters..'
    model = QRNNModel(QRNNLayer, config.num_layers, config.kernel_size,
                      config.hidden_size, config.emb_size,
                      config.num_enc_symbols, config.num_dec_symbols)

    # Initialize a model state
    model_state = vars(config)
    model_state['epoch'], model_state['train_steps'] = 0, 0
    model_state['state_dict'] = None
    
    model_path = os.path.join(config.model_dir, config.model_name)
    if os.path.exists(model_path):
        print 'Reloading model parameters..'
        checkpoint = torch.load(model_path)

        model_state['epoch'] = checkpoint['epoch']
        model_state['train_steps'] = checkpoint['train_steps']
        model.load_state_dict(checkpoint['state_dict'])

    return model, model_state


def train(config):
    # Load parallel data to train
    # TODO: using PyTorch DataIterator
    print 'Loading training data..'
    train_set = BiTextIterator(source=config.src_train,
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
                                   n_words_target=config.num_dec_symbols,
                                   shuffle_each_epoch=False,
                                   sort_by_length=config.sort_by_len,
                                   maxibatch_size=config.maxi_batches)
    else:
        valid_set = None

    # Create a Quasi-RNN model
    model, model_state = create_model(config)
    if use_cuda:
        print 'Using gpu..'
        model = model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=data_utils.pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss = 0.0
    words_seen, sents_seen = 0, 0
    start_time = time.time()

    # Training loop
    print 'Training..'
    for epoch_idx in xrange(config.max_epochs):
        if model_state['epoch'] >= config.max_epochs:
            print 'Training is already complete.', \
                  'current epoch:{}, max epoch:{}'.format(model_state['epoch'], config.max_epochs)
            break

        for source_seq, target_seq in train_set:    
            # Get a batch from training parallel data
            enc_input, enc_len, dec_input, dec_target, dec_len = \
                prepare_train_batch(source_seq, target_seq, config.max_seq_len)
 
            if enc_input is None or dec_input is None or dec_target is None:
                print 'No samples under max_seq_length ', config.max_seq_len
                continue
           
            if use_cuda:
                enc_input = Variable(enc_input.cuda())
                enc_len = Variable(enc_len.cuda())
                dec_input = Variable(dec_input.cuda())
                dec_target = Variable(dec_target.cuda())
                dec_len = Variable(dec_len.cuda())
            else:
                enc_input = Variable(enc_input)
                enc_len = Variable(enc_len)
                dec_input = Variable(dec_input)
                dec_target = Variable(dec_target)
                dec_len = Variable(dec_len)

            # Execute a single training step
            optimizer.zero_grad()
            dec_logits = model(enc_input, enc_len, dec_input)
            step_loss = criterion(dec_logits, dec_target.view(-1))
            step_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss += float(step_loss.data[0]) / config.display_freq
            words_seen += torch.sum(enc_len + dec_len).data[0]
            sents_seen += enc_input.size(0)  # batch_size

            model_state['train_steps'] += 1

            # Display training status
            if model_state['train_steps'] % config.display_freq == 0:

                avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                time_elapsed = time.time() - start_time
                step_time = time_elapsed / config.display_freq

                words_per_sec = words_seen / time_elapsed
                sents_per_sec = sents_seen / time_elapsed

                print 'Epoch ', model_state['epoch'], 'Step ', model_state['train_steps'], \
                      'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time {0:.2f}'.format(step_time), \
                      '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec)

                loss = 0.0
                words_seen, sents_seen = 0, 0
                start_time = time.time()

            # Execute a validation process
            if valid_set and model_state['train_steps'] % config.valid_freq == 0:
                print 'Validation step'
                
                valid_steps = 0
                valid_loss = 0.0
                valid_sents_seen = 0
                for source_seq, target_seq in valid_set:
                    # Get a batch from validation parallel data
                    enc_input, enc_len, dec_input, dec_target, _ = \
                        prepare_train_batch(source_seq, target_seq)

                    if use_cuda:
                        enc_input = Variable(enc_input.cuda())
                        enc_len = Variable(enc_len.cuda())
                        dec_input = Variable(dec_input.cuda())
                        dec_target = Variable(dec_target.cuda())
                    else:
                        enc_input = Variable(enc_input)
                        enc_len = Variable(enc_len)
                        dec_input = Variable(dec_input)
                        dec_target = Variable(dec_target)

                    dec_logits = model(enc_input, enc_len, dec_input)
                    step_loss = criterion(dec_logits, dec_target.view(-1))
                    valid_steps += 1 
                    valid_loss += float(step_loss.data[0])
                    valid_sents_seen += enc_input.size(0)
                    print '  {} samples seen'.format(valid_sents_seen)

                print 'Valid perplexity: {0:.2f}'.format(math.exp(valid_loss / valid_steps))

            # Save the model checkpoint
            if model_state['train_steps'] % config.save_freq == 0:
                print 'Saving the model..'

                model_state['state_dict'] = model.state_dict()
#                state = dict(list(model_state.items()))
                model_path = os.path.join(config.model_dir, config.model_name)
                torch.save(model_state, model_path)

        # Increase the epoch index of the model
        model_state['epoch'] += 1
        print 'Epoch {0:} DONE'.format(model_state['epoch'])


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
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--valid_freq', type=int, default=200)
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--model_name', type=str, default='model.pkl')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--sort_by_len', type=bool, default=True)

    config = parser.parse_args()
    print(config)
    train(config)
    print('DONE')
