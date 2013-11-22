'''
Created on Nov 21, 2013

@author: seylom
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import scipy


def get_topic_models(data, n, lda=None):
    corpus = gensim.matutils.Sparse2Corpus(data, documents_columns=False)
    if (lda is None):
        lda = gensim.models.LdaModel(corpus, num_topics=n)
    topics = gensim.matutils.corpus2csc(lda[corpus]).T
    return topics, lda


def load_dataset():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    return train, test


def get_test_ids(test):
    return test['id']


def get_labels(train):
    ''' returns labels for the data
    '''
    y = train.ix[:, 4:]

    return y


def get_test_features(test, tfidf_w, tfidf_c, lda):
    X_tfidf_words = tfidf_w.transform(test)
    X_tfidf_chars = tfidf_c.transform(test)

    X_topics, _ = get_topic_models(X_tfidf_words, 400, lda)
    #X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])
    X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics])
    #X = X_topics

    features = X.toarray()
    return features


def get_train_features(data):
    '''extracts features from the data
    '''
    train_data = data  # tweet column

    #train_data = train['tweet'].map(lambda x: p.sub(" ",x))
    #test_data = test['tweet'].map(lambda x: p.sub(" ",x))

    tfidf_words = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                  max_df=0.8, max_features=1600,
                                  analyzer="word", stop_words='english',
                                  strip_accents='unicode', ngram_range=(1, 3))

    tfidf_chars = TfidfVectorizer(sublinear_tf=True,
                                  min_df=0.001, max_df=0.8,
                                  max_features=1600, analyzer="char",
                            stop_words='english', strip_accents='unicode',
                            ngram_range=(2, 7))

    tfidf_words.fit(train_data)
    tfidf_chars.fit(train_data)

    #tfidf_words.fit(np.hstack((train_data,test_data)))
    #tfidf_chars.fit(np.hstack((train_data,test_data)))

    X_tfidf_words = tfidf_words.transform(train_data)
    X_tfidf_chars = tfidf_chars.transform(train_data)

    X_topics, lda = get_topic_models(X_tfidf_words, 400)
    #X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])
    X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics])
    #X = X_topics
    features = X.toarray()

    return features, tfidf_words, tfidf_chars, lda
