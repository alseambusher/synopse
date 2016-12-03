from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import re
import numpy as np
from six.moves import xrange
import tensorflow as tf
import data_initializer as di
import seq2seq_model

tf.app.flags.DEFINE_boolean("decode", False, "Train vs test")

tf.app.flags.DEFINE_string("data_dir", "~/.translate", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "~/.translate", "Training directory.")

tf.app.flags.DEFINE_integer("x_vocab_size", 40000, "Vocabulary size of text to be summarized.")
tf.app.flags.DEFINE_integer("y_vocab_size", 40000, "vocabulary size of summaries.")


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50, "Steps per checkpoint")

tf.app.flags.DEFINE_string("vocab_path", "data/vocab.txt", "Vocabulary path.")
tf.app.flags.DEFINE_string("checkpoint_dir", "~/.translate", "Training directory.")
tf.app.flags.DEFINE_string("review_text_train", "data/train_x.txt", "Review Text train file path.")
tf.app.flags.DEFINE_string("summary_train", "data/train_y.txt", "Summary train file path.")
tf.app.flags.DEFINE_string("review_text_validation", "data/test_x.txt", "Review Text validation file path.")
tf.app.flags.DEFINE_string("summary_validation", "data/test_y.txt", "Summary validation file path.")

FLAGS = tf.app.flags.FLAGS

buckets = [(10, 5), (20, 10), (30, 20), (50, 30), (75, 50), (100, 60)]


def create_model(session):
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.x_vocab_size, FLAGS.y_vocab_size,
        buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=True, dtype=tf.float32)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and os.path.exists(ckpt.model_checkpoint_path):
        print("Reading model %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("New model")
        session.run(tf.initialize_all_variables())
    return model


def train():
    with tf.Session() as sess:
        model = create_model(sess)
        dev_set = di.initialize_data(FLAGS.review_text_validation, FLAGS.summary_validation, buckets)[0]
        train_set = di.initialize_data(FLAGS.review_text_train, FLAGS.summary_train, buckets)[0]

        train_bucket_sizes = map(lambda x: len(train_set[x]), range(len(buckets)))
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = map(lambda x: sum(train_bucket_sizes[:x + 1]) / train_total_size, range(len(train_bucket_sizes)))

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        # stop this at the right perplexity
        while True:
            split = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > split])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 200 else float("inf")
                print ("Step %d : learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(FLAGS.train_dir, "summarize.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                for bucket_id in xrange(len(buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        model = create_model(sess)
        model.batch_size = 1
        review_text_vocab_path = os.path.join(FLAGS.vocab_path)
        _, review_text_vocab, reversed_summary_vocab = di.initialize_vocabulary(None, None, review_text_vocab_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            words = seq2seq_model.basic_tokenizer(sentence)
            token_ids = [tf.compat.as_bytes(sentence).get(w, 3) for w in words]
            bucket_id = len(buckets) - 1
            for i, bucket in enumerate(buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Bucketing failed for: %s", sentence)

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if 2 in outputs:
                outputs = outputs[:outputs.index(2)]
            print(" ".join([tf.compat.as_str(reversed_summary_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
