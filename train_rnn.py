#!/usr/bin/env python2

from argparse import ArgumentParser
from reuters_article import ReutersArticle
from rnn_model import RNNModel
from batch_generator import BatchGenerator
from utils import idx_arr_to_str
from definitions import MODEL_SAVE_DIR, DATASET_DIR, LOG_DIR, MODEL_PREFIX
import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

DISPLAY_INTERVAL = 20
TEXT_PREDICTION_LOG_INTERVAL = 200
MODEL_SAVE_INTERVAL = 3000
VALIDATION_INTERVAL = 5000


def get_model_file():
    if os.path.exists(os.path.join(MODEL_SAVE_DIR, 'checkpoint')):
        return tf.train.latest_checkpoint(MODEL_SAVE_DIR)
    else:
        return ''


def perform_validation_run(rnn_model, validation_batch_generator):
    current_validation_epochs = validation_batch_generator.n_epochs
    batch_size = validation_batch_generator.batch_size
    istate = np.zeros(shape=(rnn_model.n_layers, 2, batch_size, rnn_model.cell_size))

    total_accuracy = 0.
    total_loss = 0.
    n_batches = 0

    while validation_batch_generator.n_epochs == current_validation_epochs:
        batch, labels, seq_length_arr, istate = validation_batch_generator.get_batch(istate)
        prediction, istate, accuracy, loss = rnn_model.process_validation_batch(batch, labels, seq_length_arr, istate)
        total_accuracy += accuracy
        total_loss += loss
        n_batches += 1

        if n_batches % DISPLAY_INTERVAL == 0:
            print "[ INFO] Processed %d validation batches." % n_batches

    average_accuracy = total_accuracy / n_batches
    average_loss = total_loss / n_batches

    print "---------------------------------------------"
    print "[ INFO] Validation run results:"
    print "[ INFO] Average Accuracy: %.3f" % average_accuracy
    print "[ INFO] Average Loss: %.3f" % average_loss
    print "---------------------------------------------"

    return average_loss, average_accuracy


def train_rnn(training_articles, testing_articles, n_epochs, batch_size, seq_length, char_skip, dropout_pkeep):
    print "[ INFO] Parsing training articles..."
    training_batch_generator = BatchGenerator(training_articles, batch_size, seq_length, char_skip)

    print "[ INFO] Parsing validation articles..."
    validation_batch_generator = BatchGenerator(testing_articles, batch_size, seq_length, char_skip)

    model_file = get_model_file()
    if model_file:
        rnn_model = RNNModel.load_from_model_file(model_file)
        state_file = os.path.join(MODEL_SAVE_DIR, 'saved-vars.npz')
        if not os.path.exists(state_file):
            raise IOError("Numpy state file does not exist")
        saved_vars = np.load(state_file)
        istate = saved_vars['cell-state']
        training_batch_generator.restore_state_dict(**saved_vars)
        print "[ INFO] Resuming training from epoch %d, global step %d" % (training_batch_generator.n_epochs,
                                                                           rnn_model.training_step_num)
    else:
        print "[ INFO] Initializing RNN"
        rnn_model = RNNModel(max_seq_length=seq_length)
        rnn_model.init_network()
        istate = np.zeros(shape=(rnn_model.n_layers, 2, batch_size, rnn_model.cell_size))

    log_dir = os.path.join(LOG_DIR, 'training_%s' % datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(log_dir)
    log_file = open(os.path.join(log_dir, 'log.txt'), 'w')

    validation_accuracies = list()
    validation_losses = list()
    validation_steps = list()

    while training_batch_generator.n_epochs < n_epochs:
        batch, labels, seq_length_arr, istate = training_batch_generator.get_batch(istate)
        pred, ostate, acc = rnn_model.process_training_batch(batch, labels, seq_length_arr, istate, dropout_pkeep)

        if rnn_model.training_step_num % DISPLAY_INTERVAL == 0:
            print "[ INFO] Accuracy at step %d (epoch %d): %.3f" % (rnn_model.training_step_num,
                                                                    training_batch_generator.n_epochs + 1, acc)
            print "[ INFO] Prediction of first sample in minibatch: %s" % idx_arr_to_str(pred[0])

        if rnn_model.training_step_num % TEXT_PREDICTION_LOG_INTERVAL == 0:
            log_file.write("Text prediction at step %d:\n" % rnn_model.training_step_num)
            for i in range(batch_size):
                log_file.write(idx_arr_to_str(pred[i]) + '\n')
            log_file.write("-----------------------------------------------------\n")

        if rnn_model.training_step_num % MODEL_SAVE_INTERVAL == 0:
            print "[ INFO] Saving model..."
            rnn_model.tf_saver.save(rnn_model.session, os.path.join(MODEL_SAVE_DIR, MODEL_PREFIX),
                                    global_step=rnn_model.training_step_num)

            # also save the cell state and counters of the BatchGenerator
            vars_to_store = training_batch_generator.get_state_dict()
            vars_to_store.update({'cell-state': ostate})
            np.savez(os.path.join(MODEL_SAVE_DIR, 'saved-vars.npz'), **vars_to_store)

        if rnn_model.training_step_num % VALIDATION_INTERVAL == 0:
            print "[ INFO] Starting validation run"
            avg_loss, avg_accuracy = perform_validation_run(rnn_model, validation_batch_generator)
            validation_steps.append(rnn_model.training_step_num)
            validation_accuracies.append(avg_accuracy)
            validation_losses.append(avg_loss)

            plt.plot(validation_steps, validation_accuracies, label='accuracy')
            plt.plot(validation_steps, validation_losses, label='loss')

            plt.xlabel('Training Step')
            plt.yticks(np.arange(0., 1.05, 0.05))
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'validation_loss-accuracy-plot.png'))
            plt.close()

        istate = ostate

    log_file.close()


def main(args):
    if not os.path.exists(DATASET_DIR):
        raise EnvironmentError(
            "The 'reuters21578' dataset directory was not found at the pre-set path. Consider reviewing the paths in "
            "'definitions.py' for correctness.")

    pickled_training_articles = os.path.join(DATASET_DIR, 'processing/articles_training.pkl')
    pickled_validation_articles = os.path.join(DATASET_DIR, 'processing/articles_testing.pkl')

    assert os.path.exists(pickled_training_articles)
    assert os.path.exists(pickled_validation_articles)

    training_articles = ReutersArticle.unpickle_from_file(pickled_training_articles)
    validation_articles = ReutersArticle.unpickle_from_file(pickled_validation_articles)

    # The entire dataset is too big (~8000 articles. For our case it is sufficient to train on a small fraction of this.
    training_articles = training_articles[:1000]
    validation_articles = validation_articles[:200]

    print "[ INFO] Loaded %d training and %d testing articles" % (len(training_articles), len(validation_articles))

    train_rnn(training_articles, validation_articles, args.epochs, args.batch_size, args.seq_length, args.char_skip,
              args.dropout_pkeep)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training RNN model for text prediction using the Reuters21578 '
                                        'dataset.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--seq-length', type=int, default=70)
    parser.add_argument('--dropout-pkeep', type=float, default=0.5)
    parser.add_argument('--char-skip', type=int, default=4)

    main(parser.parse_args())
