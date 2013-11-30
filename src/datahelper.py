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
from utils import get_bucket
import re


def load_dataset():
    train = pd.read_csv('train.csv', nrows=5000)
    test = pd.read_csv('test.csv')

#     p = re.compile("(RT @mention:|RT|@mention)\W", re.I)
#     for i, row in train.iterrows():
#         row['tweet'] = p.sub("", row['tweet'])

    return train, test


def get_test_ids(test):
    return test['id']


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
