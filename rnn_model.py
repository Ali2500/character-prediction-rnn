#!/usr/bin/env python2

import tensorflow as tf
import os
import numpy as np
from utils import idx_arr_to_str, str_to_one_hot
from definitions import N_CHARS


class RNNModel(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.n_layers = 1
        self.cell_size = 1024

        self.session = None
        self.tf_saver = None
        self.input_batches = None
        self.seq_lengths = None
        self.init_state = None
        self.true_labels = None

        self.current_state = None
        self.train_step = None
        self.reshaped_softmax = None
        self.predicted_idx = None
        self.accuracy = None

        self.training_step_num = 0
        self.training_step_tf = None
        # self.max_seq_length_tf = None

    @classmethod
    def load_from_model_file(cls, model_file_prefix):
        assert os.path.exists(model_file_prefix + '.meta')

        model = cls(max_seq_length=70)
        model.session = tf.InteractiveSession()

        model.tf_saver = tf.train.import_meta_graph(model_file_prefix + '.meta')
        model.tf_saver.restore(model.session, model_file_prefix)
        tf_graph = tf.get_default_graph()

        model.input_batches = tf_graph.get_tensor_by_name('placeholders/input:0')
        model.seq_lengths = tf_graph.get_tensor_by_name('placeholders/seq_lengths:0')
        model.init_state = tf_graph.get_tensor_by_name('placeholders/init_state:0')
        model.true_labels = tf_graph.get_tensor_by_name('placeholders/labels:0')

        model.reshaped_softmax = tf_graph.get_tensor_by_name('softmax_output:0')
        model.predicted_idx = tf_graph.get_tensor_by_name('prediction:0')
        model.accuracy = tf_graph.get_tensor_by_name('accuracy:0')
        model.current_state = tf_graph.get_tensor_by_name('current_state:0')

        model.train_step = tf_graph.get_operation_by_name('train_step')
        model.inc_train_step = tf_graph.get_operation_by_name('inc_train_step')

        model.training_step_tf = tf_graph.get_tensor_by_name('training_step_num:0')
        model.training_step_num = model.training_step_tf.eval()

        return model

    def init_network(self):
        self._build_network()
        self.session = tf.InteractiveSession()

        init = tf.global_variables_initializer()
        self.session.run(init)

        self.tf_saver = tf.train.Saver(max_to_keep=5)

    def _lstm_cell_with_dropout(self):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True),
                                             output_keep_prob=self.dropout_pkeep)

    def _build_network(self):
        self.training_step_tf = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='training_step_num')
        self.inc_train_step = tf.assign_add(self.training_step_tf, 1, name='inc_train_step')

        with tf.variable_scope('placeholders'):
            self.input_batches = tf.placeholder(tf.uint8, [None, self.max_seq_length, N_CHARS], name='input')
            self.seq_lengths = tf.placeholder(tf.int32, [None], name='seq_lengths')
            self.init_state = tf.placeholder(tf.float32, [self.n_layers, 2, None, self.cell_size],
                                             name='init_state')
            self.true_labels = tf.placeholder(tf.uint8, [None, self.max_seq_length], name='labels')
            self.dropout_pkeep = tf.placeholder(tf.float32, [], name='dropout_pkeep')

        state_per_layer = tf.unstack(self.init_state, axis=0)
        rnn_state_tuple = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1]) for i in range(self.n_layers)])

        layers = [self._lstm_cell_with_dropout() for _ in range(self.n_layers)]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
        cell_output, _current_state = tf.nn.dynamic_rnn(cells, tf.cast(self.input_batches, dtype=tf.float32),
                                                        sequence_length=self.seq_lengths, initial_state=rnn_state_tuple)
        self.current_state = tf.identity(_current_state, name='current_state')  # just to assign a name to the tensor

        cell_output = tf.reshape(cell_output, [-1, self.cell_size])
        weights = tf.Variable(tf.truncated_normal([self.cell_size, N_CHARS], stddev=0.2))
        biases = tf.Variable(tf.truncated_normal([N_CHARS], stddev=0.1))

        logits = tf.add(tf.matmul(cell_output, weights), biases)
        true_output_reshaped = tf.cast(tf.reshape(self.true_labels, [-1]), dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_output_reshaped, logits=logits,
                                                              name='softmax_cross_entropy')
        total_loss = tf.reduce_mean(loss)

        with tf.control_dependencies([self.inc_train_step]):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(total_loss, name='train_step')

        softmax = tf.nn.softmax(logits)
        self.reshaped_softmax = tf.reshape(softmax, [-1, self.max_seq_length, N_CHARS], name='softmax_output')

        self.predicted_idx = tf.argmax(self.reshaped_softmax, axis=2, output_type=tf.int32, name='prediction')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted_idx, tf.cast(self.true_labels, dtype=tf.int32)),
                                               dtype=tf.float32), name='accuracy')

    def process_training_batch(self, batch, batch_label, seq_len, istate, dropout_pkeep):
        feed_dict = {self.input_batches: batch, self.true_labels: batch_label, self.seq_lengths: seq_len,
                     self.init_state: istate, self.dropout_pkeep: dropout_pkeep}

        prediction, ostate, accuracy, _ = self.session.run([self.predicted_idx, self.current_state, self.accuracy,
                                                            self.train_step], feed_dict=feed_dict)
        self.training_step_num = self.training_step_tf.eval()
        return prediction, ostate, accuracy

    def perform_inference(self, batch, seq_len, n_future_preds, input_state=None):
        if not isinstance(input_state, np.ndarray):
            istate = np.zeros(shape=(self.n_layers, 2, batch.shape[0], self.cell_size), dtype=np.float32)
        else:
            istate = input_state

        input = batch
        pred_idx = np.zeros(shape=(batch.shape[0], n_future_preds), dtype=np.uint8)

        for t in range(n_future_preds):
            if (t + 1) % 5 == 0:
                print "[ INFO] Performing inference for t = %d/%d" % (t + 1, n_future_preds)

            feed_dict = {self.input_batches: input, self.seq_lengths: seq_len, self.init_state: istate}
            prediction, ostate = self.session.run([self.predicted_idx, self.current_state], feed_dict=feed_dict)
            # print idx_arr_to_str(prediction[0])

            for i in range(batch.shape[0]):
                # shift input sequence to the left by one character
                input[i, :seq_len[i] - 1] = input[i, 1:seq_len[i]]

                # append last predicted character to the input sequence
                input[i, seq_len[i] - 1] = str_to_one_hot(idx_arr_to_str(np.array([prediction[i, seq_len[i] - 1]])))

                # store predicted character
                pred_idx[i, t] = (prediction[i, seq_len[i] - 1]).astype(np.uint8)

            istate = ostate

        pred_str = [idx_arr_to_str(pred_idx[i]) for i in range(batch.shape[0])]
        return pred_str

    def get_training_vars(self):
        return self.train_step, self.predicted_idx, self.current_state, self.accuracy
