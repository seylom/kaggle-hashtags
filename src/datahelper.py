'''
Created on Nov 21, 2013

@author: seylom
'''

import gensim
from scipy.sparse import hstack
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from features import topic_features, word_features
from features import char_features, wordcount_features


def load_dataset():
    train = pd.read_csv('train.csv', nrows=10000)
    test = pd.read_csv('test.csv')

    return train, test


def get_test_ids(test):
    return test['id']


def get_labels(train):
    ''' returns labels for the data
    '''
    y = train.ix[:, 4:]

    return y


#def get_test_features(data, tfidf_w, tfidf_c, lda, lsa, cvect):
#
#    X_tfidf_words, _ = word_features(data, tfidf=tfidf_w)
#    X_tfidf_chars, _ = char_features(data, tfidf=tfidf_c)
#    X_counts, _ = wordcount_features(data, wordvect=cvect)
#    X_topics, _ = topic_features(X_tfidf_words, lda=lda)
#
#    X_sprs = hstack([X_tfidf_words, X_tfidf_chars, X_topics,
#                                  X_counts])
#
#    #X = X_topics
#    #X_sprs = lsa.transform(X_sprs)
#
#    #X = X_sprs
#    X = X_sprs.toarray()
#
#    features = X
#    return features


#def get_word_features(data):
#    X_tfidf_words, tfidf_words = word_features(data)
#    features = X_tfidf_words.toarray()
#    return features, tfidf_words
#
#
#def get_char_features(data):
#    X_tfidf_chars, tfidf_chars = char_features(data)
#    features = X_tfidf_chars.toarray()
#    return features, tfidf_chars
#
#
#def get_word_count_features(data):
#    X_counts, cvect = wordcount_features(data)
#    features = X_counts.toarray()
#    return features, cvect
#
#
#def get_topics(data, max_topics=500):
#    X_topics, lda = topic_features(data, num_topics=max_topics)
#    features = X_topics.toarray()
#    return features, lda


def stem_tokens(tokens):
    english_stemmer = PorterStemmer()
    return [english_stemmer.stem(token) for token in tokens]


class RegexTokenizer(object):
    def __init__(self):
        self.rx = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        return [t for t in self.rx.tokenize(doc)]


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return stem_tokens([self.wnl.lemmatize(t) for t in word_tokenize(doc)])
