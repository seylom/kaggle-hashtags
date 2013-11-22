'''
Created on Nov 21, 2013

@author: seylom
'''

import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge


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
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=7)
    clf.fit(train_X, train_Y)

    return clf.predict_proba(test)[:, -1]


def predict_and_sub(train_X, train_Y, test, testSampleIds, pred_method):
    print "Training estimator ..."
    preds = pred_method(train_X, train_Y, test)

    print "Saving prediction"
    save_prediction_subs(testSampleIds, preds)
    print "Submission completed!"


def predic_multiple_model(X_train, y_train, X_test):
    pred_s = predict(X_train, y_train[:, 0:5], X_test, Ridge(alpha=1.0))
    pred_w = predict(X_train, y_train[:, 5:9], X_test, Ridge(alpha=1.0))
    pred_k = predict(X_train, y_train[:, 9:], X_test, Lasso(alpha=0.0001))

    predictions = np.hstack((pred_s, pred_w, pred_k))

    return predictions


def predict_24_models(X_train, y_train, X_test, clf):
    all_preds = []
    for i in range(24):
        preds = predict(X_train, y_train[:, i], X_test, clf)
        all_preds.append(np.matrix(preds).transpose())
    predictions = np.hstack(all_preds)

    return predictions


def save_prediction_subs(sampleIds, preds):
    prediction = np.array(np.hstack([np.matrix(sampleIds).T, preds]))
    col = '%i,' + '%f,' * 23 + '%f'
    np.savetxt('sub7.csv', prediction, col, delimiter=',')
