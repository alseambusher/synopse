from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import logging
import data_initializer as di
import numpy as np
import tensorflow as tf
import seq2seq_model


tf.app.flags.DEFINE_string("vocab_path", "data/vocab.txt", "Vocabulary path.")
tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("review_text_train", "data/train_x.txt", "Review Text train file path.")
tf.app.flags.DEFINE_string("summary_train", "data/train_y.txt", "Summary train file path.")
tf.app.flags.DEFINE_string("review_text_validation", "data/test_x.txt", "Review Text validation file path.")
tf.app.flags.DEFINE_string("summary_validation", "data/test_y.txt", "Summary validation file path.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50, "training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,"Set to True for decoding(while testing)")
tf.app.flags.DEFINE_boolean("use_fp16", False,"Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

buckets = [(10, 5), (20, 10), (30, 20), (50, 30), (75, 50), (100, 60)]
EOS_VAL_IN_VOCAB = 2


def create_model(session, forward_only):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size,
            FLAGS.vocab_size,
                buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            forward_only=forward_only,
            dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and os.path.exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    with tf.Session() as sess:
        model = create_model(sess, False)

        dev_set, _, _ = di.initialize_data(FLAGS.review_text_validation, FLAGS.summary_validation, buckets)
        train_set, _, _ = di.initialize_data(FLAGS.review_text_train, FLAGS.summary_train, buckets)
        train_bucket_sizes = []
        for b in range(len(buckets)):
                train_bucket_sizes.append(len(train_set[b]))
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = []
        for i in range(len(train_bucket_sizes)):
                train_bucket_sizes.append(sum(train_bucket_sizes[:i + 1]) / train_total_size)
                
        #training.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            bucket_id = min([i for i in range(len(train_buckets_scale))
                                             if train_buckets_scale[i] > np.random.random_sample()])

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
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                             "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                 step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "summarize.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                for bucket_id in range(len(buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("    eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                                             target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 200 else float(
                            "inf")
                    print("    eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1
        review_text_vocab_path = os.path.join(FLAGS.vocab_path)
        _, review_text_vocab, reversed_summary_vocab = di.initialize_vocabulary(None, None, review_text_vocab_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            token_ids = []
            for word in sentence.split():
                    if word in review_text_vocab:
                        token_ids.append(review_text_vocab[word])
            bucket_id = len(buckets) - 1
            for i, bucket in enumerate(buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
                else:
                        logging.warning("Sentence couldn't be put into any bucket: %s", sentence)

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                                             target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if EOS_VAL_IN_VOCAB in outputs:
                outputs = outputs[:outputs.index(EOS_VAL_IN_VOCAB)]
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