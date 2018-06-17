import numpy as np
from utils import str_to_one_hot, str_to_idx_arr
from definitions import N_CHARS
from math import ceil


class BatchGenerator(object):
    def __init__(self, training_articles, testing_articles, batch_size, seq_length, skip=4):
        self.training_articles = training_articles
        self.testing_articles = testing_articles

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.skip = skip

        self.training_ptr = 0
        self.training_subptr = 0
        self.testing_ptr = 0
        self.testing_subptr = 0

        self.n_training_epochs = 0
        self.n_testing_epochs = 0

        self.training_text = list()
        self.testing_text = list()

        self.next_article_idx = self.batch_size
        self.article_idx = [i for i in range(self.batch_size)]
        self.article_ptrs = [0 for _ in range(self.batch_size)]

        self._parse_articles()

    def _parse_articles(self):
        print "[ INFO] Pre-formatting data"

        for article in self.training_articles:
            text = ''.join([chr(x) for x in bytearray(article.text) if 32 <= x <= 126 or x == 10])
            self.training_text.append(text.replace('\n', '').replace('    ', '\n    '))
        for article in self.testing_articles:
            text = ''.join([chr(x) for x in bytearray(article.text) if 32 <= x <= 126 or x == 10])
            self.testing_text.append(text.replace('\n', '').replace('    ', '\n    '))

    def batches_per_training_epoch(self):
        n_samples = 0
        for text in self.training_text:
            if len(text) < (self.seq_length + 1):
                continue
            n_samples += int(ceil((float(len(text) - self.seq_length - 1) / self.skip)))
        return int(ceil(n_samples / self.batch_size))

    def get_training_batch_2(self, istate):
        batch = np.zeros(shape=(self.batch_size, self.seq_length, N_CHARS), dtype=np.uint8)
        batch_labels = np.zeros(shape=(self.batch_size, self.seq_length), dtype=np.uint8)
        seq_lengths = np.full(self.batch_size, dtype=np.int32, fill_value=self.seq_length)
        updated_istate = istate

        for n in range(self.batch_size):
            if self.article_ptrs[n] + self.seq_length + 1 >= len(self.training_text[self.article_idx[n]]):
                # reached end of article. Start reading new one
                # print "NEW ARTICLE at n =", n
                self.article_idx[n] = self.next_article_idx
                self.article_ptrs[n] = 0
                self.next_article_idx += 1

                # if entire article list is exhausted, repeat from start and increment epoch counter
                if self.next_article_idx == len(self.training_text):
                    self.next_article_idx = 0
                    self.n_training_epochs += 1

                # reset the state for this batch index
                updated_istate[:, :, n, :] = 0.

            batch[n] = str_to_one_hot(self.training_text[self.article_idx[n]][
                                      self.article_ptrs[n]:self.article_ptrs[n] + self.seq_length])
            batch_labels[n] = str_to_idx_arr(self.training_text[self.article_idx[n]][
                                             self.article_ptrs[n] + 1:self.article_ptrs[n] + self.seq_length + 1])

            self.article_ptrs[n] += self.skip

        return batch, batch_labels, seq_lengths, updated_istate

    def get_state_dict(self):
        return {'next-article-idx': np.int(self.next_article_idx),
                'article-idx': np.array(self.article_idx, dtype=np.int),
                'article-ptrs': np.array(self.article_ptrs, dtype=np.int),
                'n-training-epochs': np.int(self.n_training_epochs)}

    def restore_state_dict(self, **kwargs):
        self.next_article_idx = int(kwargs['next-article-idx'])
        self.article_idx = kwargs['article-idx'].tolist()
        self.article_ptrs = kwargs['article-ptrs'].tolist()
        self.n_training_epochs = int(kwargs['n-training-epochs'])

    def get_testing_batch(self, batch_size, seq_length, skip):
        batch = np.zeros(shape=(batch_size, seq_length, N_CHARS), dtype=np.uint8)
        batch_labels = np.zeros(shape=(batch_size, seq_length), dtype=np.uint8)
        seq_lengths = np.full(batch_size, dtype=np.int32, fill_value=seq_length)

        # for brevity:
        art_list = self.testing_articles
        ptr = self.testing_ptr
        subptr = self.testing_subptr

        for i in range(batch_size):
            if subptr + seq_length + 1 >= len(art_list[ptr].text):
                ptr += 1
                subptr = 0
                if ptr >= len(art_list):
                    ptr = 0
                    self.n_testing_epochs += 1

            batch[i] = str_to_one_hot(art_list[ptr].text[subptr:subptr + seq_length])
            batch_labels[i] = str_to_idx_arr(art_list[ptr].text[subptr + 1:subptr + seq_length + 1])

            subptr += skip

        self.testing_ptr = ptr
        self.testing_subptr = subptr

        return batch, batch_labels, seq_lengths