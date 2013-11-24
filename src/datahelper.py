'''
Created on Nov 21, 2013

@author: seylom
'''

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
import scipy
from sklearn.preprocessing import Normalizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, chi2, f_regression


def get_topic_models(data, n, lda=None):
    corpus = gensim.matutils.Sparse2Corpus(data, documents_columns=False)
    if (lda is None):
        lda = gensim.models.LdaModel(corpus, num_topics=n)
    topics = gensim.matutils.corpus2csc(lda[corpus]).T
    return topics, lda


def load_dataset():
    train = pd.read_csv('train.csv', nrows=30000)
    test = pd.read_csv('test.csv')

    return train, test


def get_test_ids(test):
    return test['id']


def get_labels(train):
    ''' returns labels for the data
    '''
    y = train.ix[:, 4:]

    return y


def get_test_features(test, tfidf_w, tfidf_c, lda, cvect):
    X_tfidf_words = tfidf_w.transform(test)
    X_tfidf_chars = tfidf_c.transform(test)
    X_words = cvect.transform(test)

    X_topics, _ = get_topic_models(X_tfidf_words, 500, lda)
    #X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])

    #X_sprs = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics])
    #X_words = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])

    #X_words = lsa.transform(X_words)
    #X_words = nrm.transform(X_words)

    #X_sprs = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics])
    X_sprs = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics,
                                  X_words])

    #X = X_topics
    #X_svd = lsa.transform(X_sprs)
    #X = nrm.transform(X_svd)
    #X_svd = X.toarray()

    X = X_sprs.toarray()

    #X = pca.transform(X)

    features = X
    return features


def get_train_features(data, max_features=200, labels=None):
    '''extracts features from the data
    '''
    train_data = data  # tweet column

    #train_data = train['tweet'].map(lambda x: p.sub(" ",x))
    #test_data = test['tweet'].map(lambda x: p.sub(" ",x))

    tfidf_words = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                  max_df=0.8, max_features=3000,
                                  analyzer="word", stop_words='english',
                                  strip_accents='unicode', ngram_range=(1, 2))

    tfidf_chars = TfidfVectorizer(sublinear_tf=True,
                                  min_df=0.001, max_df=0.8,
                                  max_features=3000, analyzer="char",
                            stop_words='english', strip_accents='unicode',
                            ngram_range=(1, 7))

    cvect = CountVectorizer(max_features=200, max_df=0.6,
                            stop_words='english')

    tfidf_words.fit(train_data)
    tfidf_chars.fit(train_data)
    counts = cvect.fit(train_data)

    #tfidf_words.fit(np.hstack((train_data,test_data)))
    #tfidf_chars.fit(np.hstack((train_data,test_data)))

    X_tfidf_words = tfidf_words.transform(train_data)
    X_tfidf_chars = tfidf_chars.transform(train_data)
    X_counts = cvect.transform(train_data)

    X_words = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])

    lsa = TruncatedSVD(max_features)
    #X_words = lsa.fit_transform(X_words)

    nrm = Normalizer()
    #X_words = nrm.fit_transform(X_words)

    X_topics, lda = get_topic_models(X_tfidf_words, 500)
    #X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars])

    #X = scipy.sparse.hstack([X_words, X_topics])
    X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics, X_counts])
    #X = scipy.sparse.hstack([X_tfidf_words, X_tfidf_chars, X_topics])
    #X = X_topics
    #X = X.toarray()

    features = X.toarray()

    pca = PCA(n_components=2000)
#    features = pca.fit_transform(features)

    return features, tfidf_words, tfidf_chars, lda, cvect


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
