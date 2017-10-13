'''
Utility functions
'''

''' 
This code is borrowed from the util.py of 
nematus project (https://github.com/rsennrich/nematus)
'''

import sys
import json
import cPickle as pkl
import gzip

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


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

