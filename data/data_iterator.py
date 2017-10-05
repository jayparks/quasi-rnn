import torch
import numpy as np
import shuffle
from util import load_dict

import data_utils
from data_utils import load_dict

'''
Majority of this code is borrowed from the data_iterator.py of
nematus project (https://github.com/rsennrich/nematus)
'''

class TextIterator:
    """Simple Text iterator."""
    def __init__(self, source, source_dict,
                 batch_size=128, maxlen=None,
                 n_words_source=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=20,):

        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')

        self.source_dict = load_dict(source_dict)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []

        # fill buffer, if it's empty
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if not ss:
                    break
                self.source_buffer.append(ss.strip().split())
    
            # sort by buffer
            if self.sort_by_length:
                slen = np.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
    
                _sbuf = [self.source_buffer[i] for i in sidx]
    
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()
    
        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
    
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                
                if self.maxlen and len(ss) > self.maxlen:
                    continue
                if self.skip_empty and (not ss):
                    continue

                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]
                
                source.append(ss)
    
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
    
        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0:
            source = self.next()
    
        return source

class BiTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')
            self.target = data_utils.fopen(target, 'r')

        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if not ss:
                    break
                tt = self.target.readline()
                if not tt:
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                # read from target file and map to word index
                tt = self.target_buffer.pop()

                if self.maxlen:
                    if len(ss) > self.maxlen and len(tt) > self.maxlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]

                tt = [self.target_dict[w] if w in self.target_dict 
                      else data_utils.unk_token for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target 
                          else data_utils.unk_token for w in tt]

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


    # batch preparation of a given sequence
def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if maxlen is None or l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)
    
    x_lengths = torch.LongTensor(lengths_x)
    maxlen_x = torch.max(x_lengths)

    x = torch.ones(batch_size, maxlen_x) * data_utils.pad_token
    
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = torch.LongTensor(s_x)
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
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)
    
    x_lengths = torch.LongTensor(lengths_x)
    y_lengths = torch.LongTensor(lengths_y)

    maxlen_x = torch.max(x_lengths)
    maxlen_y = torch.max(y_lengths)
    
    x = torch.ones(batch_size, maxlen_x).long() * data_utils.pad_token
    # length + 1 for _GO or EOS token 
    y_input = torch.ones(batch_size, maxlen_y+1).long() * data_utils.pad_token
    y_target = torch.ones(batch_size, maxlen_y+1).long() * data_utils.pad_token
   
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = torch.LongTensor(s_x)

        # insert _GO token at the beginning of a sequence
        y_input[idx, 0] = data_utils.start_token
        y_input[idx, 1:lengths_y[idx]+1] = torch.LongTensor(s_y)

        # insert EOS token at the end of a sequence
        y_target[idx, :lengths_y[idx]] = torch.LongTensor(s_y)
        y_target[idx, lengths_y[idx]] = data_utils.end_token

    return x, x_lengths, y_input, y_target, y_lengths
