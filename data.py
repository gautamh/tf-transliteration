from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, io, re, random, functools
from functools import reduce

#Special Tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<GO>'
END_TOKEN = '<EOS>'
INS_TOKEN = '_'
UNKNOWN_TOKEN = ' '

SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, INS_TOKEN, UNKNOWN_TOKEN]

def split_text_file(data_file, model_dir, eval_fraction):
    """
    Split a Text dataset into train and evaluation
    """
    with io.open(data_file, 'r', encoding='utf-8') as fp:
        data = fp.readlines()

    random.shuffle(data)

    root, ext = os.path.splitext(data_file)
    train_file = os.path.join(model_dir, "{}-train{}".format(root, ext))
    eval_file = os.path.join(model_dir,"{}-eval{}".format(root, ext))
    train_offset = int(len(data)*(1-eval_fraction))

    if not os.path.exists(train_file) or not os.path.exists(eval_file):
        tf.compat.v1.logging.info('Splitting into train and test datasets..')
        with io.open(train_file, 'w', encoding='utf-8') as tfp,\
            io.open(eval_file, 'w', encoding='utf-8') as efp:

            for i, line in enumerate(data):
                if i < train_offset:
                    tfp.write(line)
                else:
                    efp.write(line)

    return train_file, eval_file

def create_vocab(data_files, vocab_fname):
    """
    Creates the character vocabulary file from a
    text dataset. Adds special tokens
    """
    chars = set()
    for data_fname in data_files:
        with io.open(data_fname, 'r', encoding='utf8') as fp:
            raw = fp.read().lower()
            chars.update(raw)

    vocab = list(chars - set(['\t', '\n'])) + SPECIALS
    tf.compat.v1.logging.info('Creating vocab file..')
    with io.open(vocab_fname, 'w', encoding='utf8') as fp:
        fp.write('\n'.join(vocab))

def load_vocab(vocab_fname):
    with io.open(vocab_fname, 'r', encoding='utf-8') as f:
        characters = f.read().splitlines()
        char_vocab = {c:i for i, c in enumerate(characters)}

    return char_vocab, characters

def make_dataset_fn(data_file, char_vocab_file, batch_size, is_train):
    char_vocab, characters = load_vocab(char_vocab_file)
    sources, targets = [], []
    with io.open(data_file, 'r', encoding='utf-8') as fp:
        for i, line in enumerate(fp):
            if i % 10000 == 0:
                tf.compat.v1.logging.info('Read %d lines', i)
            if '\t' in line:
                try:
                    s, t = line.strip().lower().split('\t')
                    sources.append(s)
                    targets.append(t)
                except ValueError as e:
                    tf.compat.v1.logging.warning('ValueError:')
                    tf.compat.v1.logging.warning(line)
            else:
                s = line.strip().lower()
                t = ''
                sources.append(s)
                targets.append(t)

    def direct_input_fn():        
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(1,), dtype=tf.int32)),
            (tf.SparseTensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(1,), dtype=tf.int32)))
        
        shapes = (([None], [None]), ((), ()))
        types = ((tf.int32, tf.int32), (tf.int32, tf.int32))
        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, sources, targets, char_vocab, characters),
            output_signature=output_signature)
        if is_train:
            dataset = dataset.shuffle(500).repeat()
        dataset = dataset.batch(batch_size).prefetch(2)
        return dataset

    # This takes in source and target items
    # from split lines. input_fn reads the input file,
    # get the sources/targets, and then feeds them to the 
    # generator_fn with functools.partial in the
    # tf.data.Dataset.from_generator.
    def generator_fn(sources, targets, char_vocab, characters):
        src, target = [], []
        src_lengths, target_lengths = [], []
        maxlen_src = 0
        maxlen_target = 0
        num_ep = 3

        ep = [INS_TOKEN]
        pad_id = char_vocab[PAD_TOKEN]
        start_id = char_vocab[START_TOKEN]
        end_id = char_vocab[END_TOKEN]
        unk_id = char_vocab[UNKNOWN_TOKEN]

        for s, t in zip(sources, targets):
            len_s = len(s)
                
            # Insert epsilons, basically spaces
            s_ex = list(reduce(lambda x,y: x + y, zip(list(s), *[ep*len_s for i in range(num_ep)])))
            
            maxlen = 500
            if len(s_ex) + 2 < maxlen:
                maxlen_src = max(maxlen_src, len(s_ex) + 2)
                maxlen_target = max(maxlen_target, len(t) + 2)
                
                src.append([start_id] + [char_vocab.get(c, unk_id) for c in s_ex] + [end_id])
                target.append([start_id] + [char_vocab.get(c, unk_id) for c in t] + [end_id])
                
                src_lengths.append(len(src[-1]))
                target_lengths.append(len(target[-1]))
        
        tf.compat.v1.logging.info('Total items %d', len(src))
        tf.compat.v1.logging.info('Max source length is %d', maxlen_src)

        src = [s + [pad_id]*(maxlen_src - len(s)) for s in src]

        for i, (s, t, l_s, l_t) in enumerate(zip(src, target, src_lengths, target_lengths)):
            input = tf.constant(s, dtype=tf.int32)
            target = tf.sparse.from_dense(tf.constant(t, dtype=tf.int32))
            input_length = tf.constant([l_s], dtype=tf.int32)
            target_length = tf.constant([l_t], dtype=tf.int32)
            yield ((input, input_length), (target, target_length))
    
    return direct_input_fn
