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
from features import get_topic_models_features, get_word_features
from features import get_char_features, get_wordcount_features


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


def get_test_features(data, tfidf_w, tfidf_c, lda, lsa, cvect):

    X_tfidf_words, _ = get_word_features(data, tfidf=tfidf_w)
    X_tfidf_chars, _ = get_char_features(data, tfidf=tfidf_c)
    X_counts, _ = get_wordcount_features(data, wordvect=cvect)
    X_topics, _ = get_topic_models_features(X_tfidf_words, lda=lda)

    X_sprs = hstack([X_tfidf_words, X_tfidf_chars, X_topics,
                                  X_counts])

    #X = X_topics
    #X_sprs = lsa.transform(X_sprs)

    #X = X_sprs
    X = X_sprs.toarray()

    features = X
    return features


def get_all_features(data, max_features=1000):
    '''
    extracts features from the data
    '''

    X_tfidf_words, tfidf_words = get_word_features(data)
    X_tfidf_chars, tfidf_chars = get_char_features(data)
    X_counts, cvect = get_wordcount_features(data)
    X_topics, lda = get_topic_models_features(X_tfidf_words, num_topics=500)

    X = hstack([X_tfidf_words, X_tfidf_chars, X_topics, X_counts])

    lsa = TruncatedSVD(max_features)
#    X = lsa.fit_transform(X)
#    features = X
    features = X.toarray()

    return features, tfidf_words, tfidf_chars, lda, lsa, cvect


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
