'''
Created on Nov 21, 2013

@author: seylom
'''

import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re


def load_dataset():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    p = re.compile("(RT @mention:|RT|@mention|\\n|#)\W", re.I)
    for _, row in train.iterrows():
        row['tweet'] = p.sub("", row['tweet'])

    for _, row in test.iterrows():
        row['tweet'] = p.sub("", row['tweet'])

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
