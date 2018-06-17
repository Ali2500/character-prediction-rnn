import unicodedata
import pickle
import os
from sys import exit


class ReutersArticle(object):
    TOPICS_YES = 'YES'
    TOPICS_NO = 'NO'
    TOPICS_BYPASS = 'BYPASS'

    LEWISSPLIT_TRAIN = 'TRAINING'
    LEWISSPLIT_TEST = 'TEST'
    LEWISSPLIT_NOT_USED = 'NOT-USED'

    CGISPLIT_TRAINING_SET = 'TRAINING-SET'
    CGISPLIT_TESTSET = 'PUBLISHED-TESTSET'

    def __init__(self):
        self.filename = ''
        self.topics_tag = ReutersArticle.TOPICS_BYPASS
        self.lewissplit = ReutersArticle.LEWISSPLIT_NOT_USED
        self.cgisplit = ReutersArticle.CGISPLIT_TRAINING_SET
        self.old_id = -1
        self.new_id = -1

        self.date = ''
        self.mk_notes = ''
        self.topics = ''
        self.places = ''
        self.people = ''
        self.orgs = ''
        self.companies = ''
        self.exchanges = ''
        self.unknown = ''

        self.author = ''
        self.date_line = ''
        self.title = ''
        self.text = ''

    def load_attributes(self, reuters_tag, filename):
        self.filename = filename

        try:
            attrs = reuters_tag.attrs
            self.topics_tag = attrs['topics']
            self.lewissplit = attrs['lewissplit']
            self.cgisplit = attrs['cgisplit']
            self.old_id = attrs['oldid']
            self.new_id = attrs['newid']

        except KeyError as err:
            print "[ERROR] Could not find one or more attributes of the reuters tag: %s" % err
            exit(1)

        try:
            if reuters_tag.date:
                self.date = unicodedata.normalize('NFKD', unicode(reuters_tag.date.string)).encode('ascii', 'ignore')

            if reuters_tag.mknote:
                self.mk_notes = unicodedata.normalize('NFKD', unicode(reuters_tag.mknote.string)).encode('ascii', 'ignore')

            if reuters_tag.topics:
                self.topics = unicodedata.normalize('NFKD', unicode(reuters_tag.topics.string)).encode('ascii', 'ignore')

            if reuters_tag.places:
                self.places = unicodedata.normalize('NFKD', unicode(reuters_tag.places.string)).encode('ascii', 'ignore')

            if reuters_tag.people:
                self.people = unicodedata.normalize('NFKD', unicode(reuters_tag.people.string)).encode('ascii', 'ignore')

            if reuters_tag.orgs:
                self.orgs = unicodedata.normalize('NFKD', unicode(reuters_tag.orgs.string)).encode('ascii', 'ignore')

            if reuters_tag.companies:
                self.companies = unicodedata.normalize('NFKD', unicode(reuters_tag.companies.string)).encode('ascii', 'ignore')

            if reuters_tag.exchanges:
                self.exchanges = unicodedata.normalize('NFKD', unicode(reuters_tag.exchanges.string)).encode('ascii', 'ignore')

            if reuters_tag.unknown:
                self.unknown = unicodedata.normalize('NFKD', unicode(reuters_tag.unknown.string)).encode('ascii', 'ignore')

            if hasattr(reuters_tag.article, 'author'):
                if reuters_tag.article.author:
                    self.author = unicodedata.normalize('NFKD', unicode(reuters_tag.article.author.string)).encode('ascii', 'ignore')

            if hasattr(reuters_tag.article, 'dateline'):
                if reuters_tag.article.dateline:
                    self.date_line = unicodedata.normalize('NFKD', unicode(reuters_tag.article.dateline.string)).encode('ascii', 'ignore')

            if hasattr(reuters_tag.article, 'title'):
                if reuters_tag.article.title:
                    self.title = unicodedata.normalize('NFKD', unicode(reuters_tag.article.title.string)).encode('ascii', 'ignore')

            if hasattr(reuters_tag.article, 'content'):
                if reuters_tag.article.content:
                    self.text = unicodedata.normalize('NFKD', unicode(reuters_tag.article.content.string)).encode('ascii', 'ignore')

        except AttributeError as err:
            print "[ERROR] Failed to load article details from file: %s" % err
            exit(1)

    @classmethod
    def from_tag(cls, tag, filename):
        article = cls()
        article.load_attributes(tag, filename)
        return article

    @classmethod
    def unpickle_from_file(cls, filepath):
        assert os.path.exists(filepath)
        with open(filepath, 'rb') as readfile:
            article_list = pickle.load(readfile)

        return article_list

    @staticmethod
    def get_concat_text(list_of_articles, remove_newline_in_article):
        assert isinstance(list_of_articles, (tuple, list))

        concat_text = ''

        for i in range(len(list_of_articles)):
            text = list_of_articles[i].text
            if remove_newline_in_article:
                text = text.replace('\n', '').replace('\r', '')

            if i:
                concat_text += '\n\n'
            concat_text += text

        return concat_text
