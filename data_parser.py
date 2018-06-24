#!/usr/bin/env python2

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from reuters_article import ReutersArticle
from random import shuffle
from definitions import DATASET_DIR
import os
import pickle


class ReutersDatasetParser(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.sgm_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.sgm')]
        self.sgm_files = sorted(self.sgm_files)
        self.article_list = list()

        if not self.sgm_files:
            raise IOError("No .sgm files were found inside the dataset directory %s" % self.dataset_path)

    def load_articles(self, min_article_length):
        n_discarded_articles = 0

        for i in range(len(self.sgm_files)):
            print "[ INFO] Loading articles from %s" % self.sgm_files[i]
            with open(self.sgm_files[i], 'r') as f:
                content = f.read()

            content = content.replace('<BODY>', '<CONTENT>').replace('</BODY>', '</CONTENT>')
            content = content.replace('<TEXT>', '<ARTICLE>').replace('</TEXT>', '</ARTICLE>')
            soup = BeautifulSoup(content, 'lxml')
            reuters_tags = soup.find_all('reuters')

            for reuters_tag in reuters_tags:
                article = ReutersArticle.from_tag(reuters_tag, os.path.split(self.sgm_files[i])[-1])
                if len(article.text) >= min_article_length:
                    self.article_list.append(article)
                else:
                    n_discarded_articles += 1

        print "[ INFO] Processed a total of %d articles of which %d were discarded and the remaining %d were retained."\
              % (len(self.article_list) + n_discarded_articles, n_discarded_articles, len(self.article_list))

    def save_to_file(self, test_ratio):
        if not self.article_list:
            print "[ERROR] Article list is empty. There is nothing to save."
            return

        processing_dir = os.path.join(self.dataset_path, 'processing')
        if not os.path.exists(processing_dir):
            os.makedirs(processing_dir)

        shuffle(self.article_list)
        n_testing_articles = int(round(len(self.article_list) * test_ratio))

        with open(os.path.join(processing_dir, 'articles_testing.pkl'), 'wb') as writefile:
            pickle.dump(self.article_list[:n_testing_articles], writefile)

        with open(os.path.join(processing_dir, 'articles_training.pkl'), 'wb') as writefile:
            pickle.dump(self.article_list[n_testing_articles:], writefile)


def main(args):
    data_parser = ReutersDatasetParser(DATASET_DIR)
    data_parser.load_articles(args.min_article_length)
    data_parser.save_to_file(args.test_ratio)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--min-article-length', type=int, default=500)

    main(parser.parse_args())
