'''
Created on Nov 21, 2013

@author: seylom
'''

import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.linear_model import Lasso, Ridge, SGDClassifier, Perceptron
from sklearn.svm import SVC, SVR
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer


def get_labels(data):
    ''' returns labels for the data
    '''
    y = data.ix[:, 4:]

    return y


def get_bucket(val, num_buckets=4):
    if val >= 1:
        return num_buckets - 1

    threshold = 1.0 / num_buckets
    return math.floor(val / threshold)


def rmse_score(target, predictions):
    return np.sqrt(np.sum(np.array(np.array(predictions) - target) ** 2) /
                                                    (len(target) * 24.0))


def rmse_score_simple(target, predictions):
    return np.sqrt(np.sum(np.array(np.array(predictions) - target) ** 2) /
                                                    (float(len(target))))


def predict(train_X, train_Y, test, clf):
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_svr(train_X, train_Y, test):
    clf = SVR(kernel='rbf', C=1e2, gamma=0.1)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_lasso(train_X, train_Y, test):
    clf = Lasso(alpha=0.0001, max_iter=1000)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_extra_tree(train_X, train_Y, test, param=30):
    clf = ExtraTreeRegressor(min_samples_leaf=param, min_samples_split=1,
                             criterion='mse')
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_decision_tree(train_X, train_Y, test, param=10):
    clf = DecisionTreeRegressor(min_samples_leaf=param, min_samples_split=1,
                             criterion='mse', max_features=500)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_rfc(train_X, train_Y, test, sample_weight):
    clf = RandomForestClassifier(min_samples_leaf=5, min_samples_split=1,
                            criterion='gini', n_estimators=100, n_jobs=1)
    clf.fit(train_X, train_Y, sample_weight=sample_weight)
    preds = clf.predict_proba(test)
    return preds


def predict_rf(train_X, train_Y, test, param=50):
    clf = RandomForestRegressor(min_samples_leaf=10, min_samples_split=1,
                                verbose=1,
                             criterion='mse', n_estimators=param, n_jobs=1)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_ridge(train_X, train_Y, test, param=10.0):
    clf = Ridge(alpha=param)
    #Ridge(alpha=11.0)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_elasticNet(train_X, train_Y, test):
    clf = ElasticNet(alpha=0.0002, max_iter=10000, l1_ratio=0.9)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_logit(train_X, train_Y, test, param=1.0):
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=param)
    clf.fit(train_X, train_Y)

    return clf.predict_proba(test)[:, -1]


def predict_knn(train_X, train_Y, test, param=15):
    clf = KNeighborsClassifier(n_neighbors=param)
    clf.fit(train_X, train_Y)

    return clf.predict(test)


def predict_and_sub(train_X, train_Y, test, testSampleIds, pred_method):
    print "Training estimator ..."
    preds = pred_method(train_X, train_Y, test)

    print "Saving prediction"
    save_prediction_subs(testSampleIds, preds)
    print "Submission completed!"


def predict_two_models(X_train, y_train, X_test):
    pred_sw = predict(X_train, y_train[:, 0:9], X_test, Ridge(alpha=1.0))
    pred_k = predict(X_train, y_train[:, 9:], X_test,  Ridge(alpha=1.0))
    predictions = np.hstack((pred_sw, pred_k))
    return predictions


def predict_sgd(X_train, y_train, X_test, sample_weight):
    clf = SGDClassifier(loss='log', alpha=0.01, l1_ratio=0, n_jobs=2,
                        n_iter=50)
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    predictions = clf.predict_proba(X_test)
    return predictions


def predict_perceptron(X_train, y_train, X_test, sample_weight):
    clf = Perceptron(alpha=0.01)
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    predictions = clf.predict_proba(X_test)
    return predictions


def predict_svc(X_train, y_train, X_test, sample_weight):
    clf = SVC(degree=3, gamma=0.0,
              kernel='rbf', probability=True)
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    predictions = clf.predict_proba(X_test)
    return predictions


def try_predict_sgd(X_train, y_train, X_test):
    train = np.vstack([X_train for _ in xrange(y_train.shape[1])])
    y = np.arange(y_train.shape[1]).repeat(X_train.shape[0])
    weights = y_train.T.ravel()
    preds = predict_sgd(train, y, X_test, weights)
    return preds


def predict_three_models_sgd_ridge(X_train, y_train, X_test):
    train_s = np.vstack([X_train for _ in xrange(5)])
    y_s = np.arange(5).repeat(X_train.shape[0])
    s_weights = y_train[:, 0:5].T.ravel()
    pred_s = predict_sgd(train_s, y_s, X_test, s_weights)

    train_w = np.vstack([X_train for _ in xrange(4)])
    y_w = np.arange(4).repeat(X_train.shape[0])
    w_weights = y_train[:, 5:9].T.ravel()
    pred_w = predict_sgd(train_w, y_w, X_test, w_weights)

    pred_k = predict_ridge(X_train, y_train[:, 9:], X_test, param=10.0)
    predictions = np.hstack((pred_s, pred_w, pred_k))
    return predictions


def predict_three_models_rfc_ridge(X_train, y_train, X_test):
    train_s = np.vstack([X_train for _ in xrange(5)])
    y_s = np.arange(5).repeat(X_train.shape[0])
    s_weights = y_train[:, 0:5].T.ravel()
    pred_s = predict_rfc(train_s, y_s, X_test, s_weights)

    train_w = np.vstack([X_train for _ in xrange(4)])
    y_w = np.arange(4).repeat(X_train.shape[0])
    w_weights = y_train[:, 5:9].T.ravel()
    pred_w = predict_rfc(train_w, y_w, X_test, w_weights)

    pred_k = predict_ridge(X_train, y_train[:, 9:], X_test, param=10.0)
    predictions = np.hstack((pred_s, pred_w, pred_k))
    return predictions


def predict_three_models(X_train, y_train, X_test):
    pred_s = predict(X_train, y_train[:, 0:5], X_test, Ridge(alpha=10.0))
    pred_w = predict(X_train, y_train[:, 5:9], X_test, Ridge(alpha=10.0))
    pred_k = predict(X_train, y_train[:, 9:], X_test,  Ridge(alpha=10.0))
    predictions = np.hstack((pred_s, pred_w, pred_k))
    return predictions


def predict_multiple_model(X_train, y_train, X_test):
    pred_sw = predict(X_train, y_train[:, 0:9], X_test, Ridge(alpha=1.0))

    pred_k_vals = []
    for i in range(15):
        print "training custom classifier #%d" % (i + 1)
        preds = predict_logit(X_train, y_train[:, i + 9], X_test)
        pred_k_vals.append(np.matrix(preds).transpose())
    pred_k = np.hstack(pred_k_vals)

    predictions = np.hstack((pred_sw, pred_k))

    return predictions


def predict_24_models(X_train, y_train, X_test, clf):
    all_preds = []
    for i in range(24):
        preds = predict(X_train, y_train[:, i], X_test, clf)
        all_preds.append(np.matrix(preds).transpose())
    predictions = np.hstack(all_preds)

    return predictions


def predict_stacked_models():
    test = pd.read_csv('test.csv')

    subs = ['submissions/sub12.csv', 'submissions/sub10.csv',
            'submissions/sub9.csv', 'submissions/sub2.csv']

    predictions = np.zeros((test.shape[0], 24))
    print predictions.shape
    for sub in subs:
        df = pd.read_csv(sub)
        predictions += df.values[:, 1:]

    predictions /= len(subs)
    save_prediction_subs(test['id'], predictions)


def predict_weighted_stacked_models():
    test = pd.read_csv('test.csv')

    pred1 = pd.read_csv('submissions/sub22.csv')
    pred2 = pd.read_csv('submissions/sub23.csv')

    predictions = 0.35 * pred1.values[:, 1:] + 0.65 * pred2.values[:, 1:]
    save_prediction_subs(test['id'], predictions)


def save_prediction_subs(sampleIds, preds):
    prediction = np.array(np.hstack([np.matrix(sampleIds).T, preds]))
    col = '%i,' + '%f,' * 23 + '%f'
    np.savetxt('submissions/sub30.csv', prediction, col, delimiter=',')


def wordcount_vectorizer():
    wordvect = CountVectorizer(max_features=10000,
                           max_df=0.7,
                           stop_words='english',
                           ngram_range=(1, 4))
    return wordvect


def wordngram_vectorizer():
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                    max_df=0.8, max_features=3000,
                                    analyzer="word", stop_words='english',
                                    strip_accents='unicode',
                                    ngram_range=(1, 4))
    return tfidf


def charngram_vectorizer():
    tfidf = TfidfVectorizer(sublinear_tf=True,
                                  min_df=0.001, max_df=0.8,
                                  max_features=3000,
                                  analyzer="char",
                                  stop_words='english',
                                  strip_accents='unicode',
                                  ngram_range=(1, 7))
    return tfidf


def build_elasticnet_pipeline():
    pipeline = Pipeline([('wordngram', wordngram_vectorizer()),
                         ('char', TfidfVectorizer())
                         ('clf', ElasticNet())
                         ])
    return pipeline


def build_ridge_pipeline():
    pipeline = Pipeline([('charngrams', charngram_vectorizer()),
                         ('tfidf_wordcount', wordcount_vectorizer())
                         ('clf', Ridge(alpha=10.0))
                         ])
    return pipeline
