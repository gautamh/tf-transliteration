from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import os, sys, io, re
import six

from absl import flags
from functools import reduce
from data import create_vocab, load_vocab, make_dataset_fn
from data import split_text_file, SPECIALS
from tensorboard.plugins.hparams import api as hp

FLAGS = flags.FLAGS


flags.DEFINE_integer("train_steps", 0,
                     "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 100, "Number of steps in evaluation.")
flags.DEFINE_integer("min_eval_frequency", 101, "Minimum steps between evals")

flags.DEFINE_string("hparams", "", "Comma separated list of hyperparameters")
flags.DEFINE_string("model_name", "ei", "Name of model")                
flags.DEFINE_string("data_file", None, "TSV Data filename")    
flags.DEFINE_float("eval_fraction", 0.05, "Fraction dataset used for evaluation")
flags.DEFINE_string("decode_input_file", None, "File to decode")  

flags.DEFINE_string("vocab_file", "chars.vocab", "Character vocabulary file")


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def get_hparams(vocab_size, overrides=""):
    hparams = {
        "batch_size": 32,
        "embedding_size": 64,
        "char_vocab_size": vocab_size + 2, # Blank label for CTC loss
        "hidden_size": 128,
        "learn_rate": 0.0002
    }
    return hparams

def get_model_dir(model_name):
    model_dir = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir

def cer(labels, predictions):
    dist = tf.edit_distance(predictions, labels)
    return tf.compat.v1.metrics.mean(dist)

def create_model():
    """
    Actual model function. 
    Refer https://arxiv.org/abs/1610.09565
    """
    def model_fn(features, labels, mode, params):
        if mode != tf.estimator.ModeKeys.PREDICT:
            features_dict = {
                'input': features[0],
                'target': labels[0],
                'input_length': features[1],
                'target_length': labels[1]
            }
        else:
            features_dict = {
                'input': features[0],
                'target': None,
                'input_length': features[1],
                'target_length': None
            }
        hparams = params

        inputs = features_dict['input']
        input_lengths = features_dict['input_length']
        targets = features_dict['target']
        target_lengths = features_dict['target_length']

        # Flatten input lengths
        input_lengths = tf.reshape(input_lengths, [-1])
        
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                    tf.random.truncated_normal(
                        [hparams['char_vocab_size'], hparams['embedding_size']],
                        stddev=(1/np.sqrt(hparams['embedding_size']))),
                    name='embeddings')

            input_emb = tf.nn.embedding_lookup(params=embeddings, ids=inputs)

        cell_fw = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hparams['hidden_size'])
        cell_bw = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hparams['hidden_size'])


        with tf.compat.v1.variable_scope('encoder'):
            # BiLSTM
            enc_outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb, 
                                input_lengths, dtype=tf.float32)

            enc_outputs = tf.concat(enc_outputs, axis=-1)

        with tf.compat.v1.variable_scope('decoder'):
            # Project to vocab size
            logits = tf.compat.v1.layers.dense(enc_outputs, hparams['char_vocab_size'])
            # CTC loss and decoder requires Time major
            logits = tf.transpose(a=logits, perm=[1, 0, 2])

        loss = None
        eval_metric_ops = None
        train_op = None
        predictions = None

        def _add_empty_cols(t, max_shape, constant_values):
            s = tf.shape(input=t)
            shape_diff = tf.math.subtract(max_shape, s)
            return tf.cond(
                pred=tf.reduce_all(
                    input_tensor=tf.math.equal(s, max_shape)),
                    # if the tensor is the same shape as the largest tensor, don't do anything
                    true_fn=lambda: t,
                    # if it's not the same shape, we assume it has the same number of rows
                    # and we add as many 0-filled columns as needed to make it the same
                    # number of years.
                    false_fn=lambda: tf.concat([t, tf.fill((max_shape[0], shape_diff[1]), 0)], axis=1))

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.compat.v1.nn.ctc_loss(targets, logits, input_lengths)
            loss = tf.reduce_mean(input_tensor=loss)
            optimizer = tfa.optimizers.LazyAdam(learning_rate=hparams['learn_rate'])
            # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
            # to make tf.compat.v1.train.global_step increased correctly.
            # This assignment is a must for any `tf.train.SessionRunHook` specified in
            # estimator, as SessionRunHooks rely on global step.
            optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            # Compute the minimize_op.
            minimize_op = optimizer.get_updates(
                loss,
                tf.compat.v1.trainable_variables())[0]
            train_op = tf.group(minimize_op, *update_ops)

        elif mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.compat.v1.nn.ctc_loss(targets, logits, input_lengths,
                                 ignore_longer_outputs_than_inputs=True)
            loss = tf.reduce_mean(input_tensor=loss)
            eval_predictions, _ = tf.nn.ctc_greedy_decoder(logits, input_lengths)
            eval_metric_ops = {
                'CER': cer(targets, tf.cast(eval_predictions[0], tf.int32))
            }

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions, _ = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=input_lengths, beam_width=100, top_paths=4)
            predictions = [tf.sparse.to_dense(tf.cast(prediction, tf.int32)) for prediction in predictions]
            max_shape = reduce((lambda a, b: tf.math.maximum(a, b)), [tf.shape(input=prediction) for prediction in predictions])
            predictions = [_add_empty_cols(prediction, max_shape, 0) for prediction in predictions]
            predictions = {'decoded': tf.stack(predictions)}

        return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op,
                    eval_metric_ops=eval_metric_ops
                )

    return model_fn

def train():
    """
    Train the model:
    1. Create vocab file from dataset if not created
    2. Split dataset into test/eval if not available
    3. Create TFRecord files if not available
    4. Load TFRecord files using tf.data pipeline
    5. Train model using tf.Estimator
    """
    model_dir = get_model_dir(FLAGS.model_name)
    vocab_file = os.path.join(model_dir, FLAGS.vocab_file)

    if not os.path.exists(vocab_file):
        create_vocab([FLAGS.data_file], vocab_file)
    
    vocab, characters = load_vocab(vocab_file)
    
    train_file, eval_file = split_text_file(FLAGS.data_file, model_dir, FLAGS.eval_fraction)
    
    hparams = get_hparams(len(vocab), FLAGS.hparams)
    tf.compat.v1.logging.info('params: %s', str(hparams))

    train_input_fn = make_dataset_fn(train_file, vocab_file, hparams['batch_size'], True)
    eval_input_fn = make_dataset_fn(eval_file, vocab_file, hparams['batch_size'], False)

    estimator = tf.estimator.Estimator(
            model_fn=create_model(),
            model_dir=model_dir,
            params=hparams,
            config=tf.estimator.RunConfig()
    )

    hook = tf.estimator.experimental.make_stop_at_checkpoint_step_hook(estimator, FLAGS.train_steps)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120, steps=FLAGS.eval_steps)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def predict():
    """
    Perform transliteration using trained model. Input must be a text 
    file. Converts to a TFRecord first.
    """
    model_dir = get_model_dir(FLAGS.model_name)
    vocab_file = os.path.join(model_dir, FLAGS.vocab_file)
    
    if not os.path.exists(vocab_file):
        raise IOError("Could not find vocabulary file")
    
    vocab, rev_vocab = load_vocab(vocab_file)
    hparams = get_hparams(len(vocab), FLAGS.hparams)
    tf.compat.v1.logging.info('params: %s', str(hparams))

    if FLAGS.decode_input_file is None:
        raise ValueError("Must provide input field to decode")

    infer_input_fn = make_dataset_fn(
        FLAGS.decode_input_file, vocab_file, hparams['batch_size'], False)
    
    estimator = tf.estimator.Estimator(
            model_fn=create_model(),
            model_dir=model_dir,
            params=hparams,
            config=tf.estimator.RunConfig()
    )

    y = estimator.predict(input_fn=infer_input_fn, predict_keys=['decoded'])

    ignore_ids = set([vocab[c] for c in SPECIALS] + [0])
    
    decode_output_file = re.sub(r'\..+', '.out.txt', FLAGS.decode_input_file)

    count = 0
    with io.open(decode_output_file, 'w', encoding='utf-8') as fp:
        decodeds = [pred['decoded'] for pred in y]
        for translit_set in list(zip(*decodeds)):
            for translit in translit_set:
                fp.write(_convert_pred_to_str(translit, rev_vocab, ignore_ids) + ',')
                count += 1
                if count % 10000 == 0:
                    tf.compat.v1.logging.info('Decoded %d lines', count)
            fp.write('\n')

def _convert_pred_to_str(pred, rev_vocab, ignore_ids):
    return ''.join([rev_vocab[i] for i in pred if i not in ignore_ids])

def main(unused_argv):
    if FLAGS.decode_input_file:
        predict()
    elif FLAGS.train_steps > 0:
        train()

tf.compat.v1.app.run()