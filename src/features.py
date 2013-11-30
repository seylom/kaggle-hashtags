'''
Created on Nov 24, 2013

@author: seylom
'''

import gensim
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.linear_model import Lasso, Ridge, SGDClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.multiclass import OneVsRestClassifier
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.base import BaseEstimator


def get_ridge_model():
    combined_features = FeatureUnion([('word', CountVectorizer()),
                                      ('char', TfidfVectorizer())])

    pipeline = Pipeline([('features', combined_features),
                         ('clf', Ridge())
                         ])

    parameters = {
        'features__char__analyzer': ['char'],
        'features__char__max_df': [0.5],
        'features__char__max_features': [3000],
        'features__char__ngram_range': [(1, 7)],
        'features__char__stop_words': ['english'],
        'features__word__max_df': [0.5],
        'features__word__max_features': [4000],
        'features__word__ngram_range': [(1, 2), (1, 3)],
        'features__word__stop_words': ['english'],
        'svd__n_components': [200, 300],
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1.0, 10.0],
        #'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
    return pipeline, parameters


def get_three_predictor_model():
    combined_features = FeatureUnion([('word', CountVectorizer()),
                                      ('char', TfidfVectorizer())])

    pipeline = Pipeline([('features', combined_features),
                         ('clf', ThreeModelsEstimator())])

    parameters = {
        'features__char__analyzer': ['char'],
        'features__char__max_df': [0.8],
        'features__char__max_features': [3000],
        'features__char__ngram_range': [(1, 7)],
        'features__char__stop_words': ['english'],
        'features__word__max_df': [0.8],
        'features__word__max_features': [4000],
        'features__word__ngram_range': [(1, 2)],
        'features__word__stop_words': ['english'],
        #'svd__n_components': [300, 600],
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        #'clf__alpha': [10.0],
        #'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
        'clf__estimator1': [
                (SVC(degree=3, gamma=100.0, C=0.001,
                    kernel='rbf'), False),
                (SVC(degree=3, gamma=100.0, C=0.001,
                    kernel='rbf'), False),
                (SVC(degree=3, gamma=10.0, C=0.1,
                    kernel='rbf'), False)],
        'clf__estimator2': [(Ridge(10.0), False)],
        'clf__estimator3': [(ExtraTreesRegressor(min_samples_leaf=10,
                                                min_samples_split=5,
                                                n_estimators=100,
                             criterion='mse'), False),
                            (ExtraTreesRegressor(min_samples_leaf=30,
                                                min_samples_split=5,
                                                n_estimators=500,
                             criterion='mse'), False),
                            (DecisionTreeRegressor(min_samples_leaf=30,
                                                   min_samples_split=5,
                             criterion='mse',
                             max_features='sqrt',
                              max_depth=5), False)],
    }
    return pipeline, parameters


def prob_to_weighted_class_data_transformer(X, Y):
    train = np.vstack([X for _ in xrange(Y.shape[1])])
    y = np.arange(Y.shape[1]).repeat(X.shape[0])
    weights = Y.T.ravel()
    return train, y, weights


class ThreeModelsEstimator(BaseEstimator):
    def __init__(self, estimator1=None,
                       estimator2=None,
                       estimator3=None):
        self.est1 = estimator1
        self.est2 = estimator2
        self.est3 = estimator3

        if self.est1 is None:
            self.est1 = (Ridge(10.0), False)
        if self.est2 is None:
            self.est2 = (Ridge(10.0), False)
        if self.est3 is None:
            self.est3 = (Ridge(10.0), False)

        self.__initialize()

    def __initialize(self):
        self.model1, self.process1 = self.est1
        self.model2, self.process2 = self.est2
        self.model3, self.process3 = self.est3

    def fit(self, X, Y):
        'fit S'
        if self.process1:
            train_s = np.vstack([X for _ in xrange(5)])
            y_s = np.arange(5).repeat(X.shape[0])
            s_weights = Y[:, 0:5].T.ravel()
            self.model1.fit(train_s, y_s, s_weights)
        else:
            self.model1.fit(X, Y[:, 0:5])

        'fit w'
        if self.process2:
            train_w = np.vstack([X for _ in xrange(4)])
            y_w = np.arange(4).repeat(X.shape[0])
            w_weights = Y[:, 5:9].T.ravel()
            self.model2.fit(train_w, y_w, w_weights)
        else:
            self.model2.fit(X, Y[:, 5:9])

        'fit k'
        self.model3.fit(X, Y[:, 9:])

    def predict(self, X):
        pred_s = self.model1.predict(X)
        pred_w = self.model2.predict(X)
        pred_k = self.model3.predict(X)
        predictions = np.hstack((pred_s, pred_w, pred_k))

        return predictions


class FeatureExtractor:
    '''
    This class contains all methods used for feature
    extraction
    '''
    def __init__(self, settings=None):
        self.info = {}
        self.params = {}
        self.__tfidf_words = None
        self.__tfidf_chars = None
        self.__word_vect = None
        self.__char_vect = None
        self.__lda = None
        self.__settings = None
        self.__svd = None
        self.__hash_vect = None

        self.__initialize_settings(settings)

    def __initialize_settings(self, settings):
        #setting up some defaults
        self.__settings = settings

        if self.__settings is None:
            self.__settings = {}
            self.__settings['word'] = 3000
            self.__settings['char'] = 3000
            self.__settings['wordcount'] = 3000
            self.__settings['charcount'] = 3000
            self.__settings['topic'] = 400
            self.__settings['wordhash'] = 3000
        else:
            if 'word' not in self.__settings:
                self.__settings['word'] = 3000
            if 'char' not in self.__settings:
                self.__settings['char'] = 3000
            if 'wordcount' not in self.__settings:
                self.__settings['wordcount'] = 3000
            if 'charcount' not in self.__settings:
                self.__settings['charcount'] = 3000
            if 'topic' not in self.__settings:
                self.__settings['topic'] = 400
            if 'wordhash' not in self.__settings:
                self.__settings['wordhash'] = 3000

    def get_features(self, data, feature_types):
        '''
        '''
        if feature_types is not None:
            all_results = []
            for feature in feature_types:
                if feature == 'word':
                    result = self.__extract_word_features(data)
                elif feature == 'char':
                    result = self.__extract_char_features(data)
                elif feature == 'wordcount':
                    result = self.__extract_wordcount_features(data)
                elif feature == 'charcount':
                    result = self.__extract_charcount_features(data)
                elif feature == 'topic':
                    result = self.__extract_topic_features(data)
                    #print self.__lda.show_topics(formatted=True)
                elif feature == 'wordhash':
                    result = self.__extract_wordhash_features(data)

                if result is not None:
                    all_results.append(result)

            features = hstack(all_results)

#             if self.__svd is None:
#                 self.__svd = TruncatedSVD(n_components=200)
#                 self.__svd.fit(features)
#
#             features = self.__svd.transform(features)

            return features.toarray()
        else:
            return None

    def __extract_word_features(self, data):
        '''
        Extract tfidf word ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_words is None:
            data_result, tfidf = self.__word_features(data,
                        num_features=self.__settings['word'])

            self.__tfidf_words = tfidf
        else:
            data_result, _ = self.__word_features(data,
                            tfidf=self.__tfidf_words)

        return data_result

    def __extract_wordhash_features(self, data):
        '''
        Extract hashvect word ngram from the corpus using scikit learn
        hashvect vectorizer
        '''
        data_result = None
        if self.__hash_vect is None:
            data_result, vect = self.__wordhash_features(data,
                        num_features=self.__settings['wordhash'])

            self.__hash_vect = vect
        else:
            data_result, _ = self.__wordhash_features(data,
                            vect=self.__hash_vect)

        return data_result

    def __extract_char_features(self, data):
        '''
        Extract tfidf char ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_chars is None:
            data_result, tfidf = self.__char_features(data,
                        num_features=self.__settings['char'])

            self.__tfidf_chars = tfidf
        else:
            data_result, _ = self.__char_features(data,
                            tfidf=self.__tfidf_chars)

        return data_result

    def __extract_wordcount_features(self, data):
        '''
        Extract wordcount ngram from the corpus
        '''
        data_result = None
        if self.__word_vect is None:
            data_result, vect = self.__wordcount_features(data,
                        num_features=self.__settings['wordcount'])

            self.__word_vect = vect
        else:
            data_result, _ = self.__wordcount_features(data,
                            wordvect=self.__word_vect)

        return data_result

    def __extract_charcount_features(self, data):
        '''
        Extract charcount ngram from the corpus
        '''
        data_result = None
        if self.__char_vect is None:
            data_result, vect = self.__charcount_features(data,
                        num_features=self.__settings['charcount'])

            self.__char_vect = vect
        else:
            data_result, _ = self.__charcount_features(data,
                            charvect=self.__char_vect)

        return data_result

    def __extract_topic_features(self, data):
        '''
        Extract topics from the corpus using gensim
        '''
        data_result = None
        word_data = self.__extract_wordcount_features(data)
        if self.__lda is None:
            data_result, lda = self.__topic_features(word_data,
                        num_topics=self.__settings['topic'])

            self.__lda = lda
        else:
            data_result, _ = self.__topic_features(word_data,
                            lda=self.__lda)

        return data_result

    def __wordcount_features(self, data, wordvect=None, num_features=10000):
        '''
        extracts word count features from the provided data
        '''
        if wordvect is None:
            wordvect = CountVectorizer(max_features=num_features,
                                       max_df=0.7,
                                       stop_words='english',
                                       ngram_range=(1, 4))
            wordvect.fit(data)

        features = wordvect.transform(data)

        return features, wordvect

    def __charcount_features(self, data, charvect=None, num_features=10000):
        '''
        extracts char count features from the provided data
        '''
        if charvect is None:
            charvect = CountVectorizer(max_features=num_features,
                                       max_df=0.7,
                                       analyzer='char',
                                       stop_words='english',
                                       ngram_range=(1, 4))
            charvect.fit(data)

        features = charvect.transform(data)

        return features, charvect

    def __word_features(self, data, tfidf=None, num_features=3000):
        '''
        extracts word ngram features from the provided data
        '''
        if tfidf is None:
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                    max_df=0.8, max_features=num_features,
                                    analyzer="word", stop_words='english',
                                    strip_accents='unicode',
                                    ngram_range=(1, 4))
            tfidf.fit(data)

        features = tfidf.transform(data)

        return features, tfidf

    def __wordhash_features(self, data, vect=None, num_features=3000):
        '''
        extracts word ngram features from the provided data
        '''
        if vect is None:
            vect = HashingVectorizer(n_features=num_features,
                                    analyzer="word", stop_words='english',
                                    strip_accents='unicode',
                                    ngram_range=(1, 4))
            vect.fit(data)

        features = vect.transform(data)

        return features, vect

    def __char_features(self, data, tfidf=None, num_features=3000):
        '''
        extracts char ngram features from the provided data
        '''
        if tfidf is None:
            tfidf = TfidfVectorizer(sublinear_tf=True,
                                          min_df=0.001, max_df=0.8,
                                          max_features=num_features,
                                          analyzer="char",
                                          stop_words='english',
                                          strip_accents='unicode',
                                          ngram_range=(1, 7))
            tfidf.fit(data)

        features = tfidf.transform(data)

        return features, tfidf

    def __topic_features(self, data, lda=None, num_topics=400):
        '''
        extracts topic models features from the provided data

        data:        a sparse vector provided by scikit TfidfVectorizer
        lda:         a gensim model used to infer topic models for data
        num_topics:  maximum number of topics
        '''
        corpus = gensim.matutils.Sparse2Corpus(data, documents_columns=False)
        if (lda is None):
            lda = gensim.models.LdaModel(corpus, num_topics=num_topics)
        topics = gensim.matutils.corpus2csc(lda[corpus]).T
        return topics, lda
