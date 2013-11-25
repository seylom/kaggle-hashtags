'''
Created on Nov 21, 2013

@author: seylom
'''

import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.multiclass import OneVsRestClassifier
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def rmse_score(target, predictions):
    return np.sqrt(np.sum(np.array(np.array(predictions) - target) ** 2) /
                                                    (len(target) * 24.0))


def predict(train_X, train_Y, test, clf):
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_lasso(train_X, train_Y, test):
    clf = Lasso(alpha=0.0001)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_ridge(train_X, train_Y, test):
    clf = Ridge(alpha=1.0)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_elasticNet(train_X, train_Y, test):
    clf = ElasticNet(alpha=0.0002, max_iter=10000, l1_ratio=0.9)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test)
    return preds


def predict_logit(train_X, train_Y, test):
    clf = LogisticRegression(tol=1e-8, penalty='l2')
    clf.fit(train_X, train_Y)

    return clf.predict_proba(test)[:, -1]


def predict_knn(train_X, train_Y, test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_X, train_Y)

    return clf.predict_proba(test)[:, -1]


def predict_and_sub(train_X, train_Y, test, testSampleIds, pred_method):
    print "Training estimator ..."
    preds = pred_method(train_X, train_Y, test)

    print "Saving prediction"
    save_prediction_subs(testSampleIds, preds)
    print "Submission completed!"


def predic_two_models(X_train, y_train, X_test):
    pred_sw = predict(X_train, y_train[:, 0:9], X_test, Ridge(alpha=1.0))
    pred_k = predict(X_train, y_train[:, 9:], X_test, SVR())
    predictions = np.hstack((pred_sw, pred_k))
    return predictions


def predic_multiple_model(X_train, y_train, X_test):
    pred_sw = predict(X_train, y_train[:, 0:9], X_test, Ridge(alpha=1.0))

    pred_k_vals = []
    for i in range(15):
        print "training logit #%d" % (i + 1)
        preds = predict_logit(X_train, y_train[:, i + 9], X_test)
        pred_k_vals.append(np.matrix(preds).transpose())
    pred_k = np.hstack(pred_k_vals)

    predictions = np.hstack((pred_sw, pred_k))

    return predictions


def get_bucket(val, num_buckets=4):
    threshold = 1.0 / num_buckets
    return math.floor(val / threshold)


def predict_24_models(X_train, y_train, X_test, clf):
    all_preds = []
    for i in range(24):
        preds = predict(X_train, y_train[:, i], X_test, clf)
        all_preds.append(np.matrix(preds).transpose())
    predictions = np.hstack(all_preds)

    return predictions


def predict_stacked_models():
    test = pd.read_csv('test.csv')
    subs = ['sub12.csv', 'sub10.csv', 'sub9.csv']

    predictions = np.zeros((test.shape[0], 24))
    print predictions.shape
    for sub in subs:
        df = pd.read_csv(sub)

        predictions += df.values[:, 1:]

    predictions /= len(subs)
    save_prediction_subs(test['id'], predictions)


def save_prediction_subs(sampleIds, preds):
    prediction = np.array(np.hstack([np.matrix(sampleIds).T, preds]))
    col = '%i,' + '%f,' * 23 + '%f'
    np.savetxt('sub17.csv', prediction, col, delimiter=',')
