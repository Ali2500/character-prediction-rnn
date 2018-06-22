#!/usr/bin/env python2

from argparse import ArgumentParser
from rnn_model import RNNModel
from definitions import N_CHARS
from utils import str_to_one_hot
from definitions import MODEL_SAVE_DIR
import tensorflow as tf
import numpy as np
import os


def main(args):
    model_file = tf.train.latest_checkpoint(MODEL_SAVE_DIR)
    rnn_model = RNNModel.load_from_model_file(model_file)

    assert os.path.exists(args.text_file)
    with open(args.text_file, 'r') as readfile:
        content = [x.strip('\n') for x in readfile.readlines()]

    batch = np.zeros(shape=(len(content), 70, N_CHARS), dtype=np.uint8)
    seq_lengths = np.array([len(x) for x in content]).astype(np.int32)

    for i in range(len(content)):
        batch[i, :len(content[i])] = str_to_one_hot(content[i])

    prediction = rnn_model.perform_inference(batch, seq_lengths, args.prediction_length)

    filewrite = False
    if args.output:
        outfile = open(args.output, 'w')
        filewrite = True

    for i in range(len(content)):
        out_string = "'%s' --> '%s'\n-------------------------------\n" % (content[i], prediction[i])
        print out_string
        if filewrite:
            outfile.write(out_string)

    if filewrite:
        outfile.close()


if __name__ == '__main__':
    parser = ArgumentParser(description="Script for performing inference from trained RNN model on custom text strings")
    parser.add_argument('--text-file', '-i', required=True)
    parser.add_argument('--output', '-o', required=False)
    parser.add_argument('--prediction-length', type=int, default=100)

    main(parser.parse_args())
