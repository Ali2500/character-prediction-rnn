from utils import str_to_idx_arr, str_to_one_hot, one_hot_to_str, idx_arr_to_str
# import tensorflow as tf
from batch_generator import BatchGenerator
from reuters_article import ReutersArticle
import os
from definitions import DATASET_DIR
import numpy as np
from random import shuffle

pickled_training_articles = os.path.join(DATASET_DIR, 'processing/articles_training.pkl')
pickled_testing_articles = os.path.join(DATASET_DIR, 'processing/articles_testing.pkl')

training_articles = ReutersArticle.unpickle_from_file(pickled_training_articles)
testing_articles = ReutersArticle.unpickle_from_file(pickled_testing_articles)

# x = training_articles[0].text
# x = x.replace('\n', '')
# x = x.replace('    ', '\n    ')
# print x

for article in training_articles:
    article.text = ''.join([chr(x) for x in bytearray(article.text) if 32 <= x <= 126])
for article in testing_articles:
    article.text = ''.join([chr(x) for x in bytearray(article.text) if 32 <= x <= 126])

shuffle(training_articles)
batch_generator = BatchGenerator(training_articles, testing_articles, 2, 70, 20)

istate = np.zeros(shape=(2, 2, 2, 1024), dtype=np.float32)

print batch_generator.batches_per_training_epoch()

i = 0
while batch_generator.n_training_epochs != 1:
    batch_generator.get_training_batch_2(istate)
    i += 1

print i

# for i in range(200):
#     print "*************************************************\n"
#     batch, batch_labels, _, ostate = batch_generator.get_training_batch_2(istate)
#     for j in range(batch.shape[0]):
#         print one_hot_to_str(batch[j])
#
#     raw_input('Enter to continue')
#
#     istate = ostate




# for i in range(10000):
#     batch = batch_generator.get_training_batch(20, 70, skip=4)
#     if i % 100 == 0:
#         print batch_generator.n_training_epochs

# X = tf.Variable(initial_value=0, trainable=True, name='X')
# add_op = tf.assign_add(X, 1)
#
# A = tf.Variable(initial_value=0, trainable=True, name='A')
# add_op_2 = tf.assign_add(A, 1)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# for i in range(5):
#     with tf.control_dependencies([add_op]):
#         sess.run([add_op_2])
#
# print A.eval()
# print X.eval()
#
#
# string = 'abcd\nefgh123 '
# print str_to_idx_arr(string)
# one_hot = str_to_one_hot('Aa12 bc\n')
# print one_hot_to_str(one_hot)