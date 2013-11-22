'''
Created on Nov 21, 2013

@author: seylom
'''

import numpy as np
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.cross_validation import KFold
from utils import predict_ridge
from datahelper import load_dataset, get_train_features, get_test_features
from datahelper import get_labels, get_test_ids, get_test_ids
from utils import predict_and_sub


def rmse_score(target, predictions):
    return np.sqrt(np.sum(np.array(np.array(predictions) - target) ** 2) /
                                                    (len(target) * 24.0))


def get_meta_features(train, test):
    print "meta features extraction ..."
    meta_train, tfidf_w, tfidf_c, lda = get_train_features(train)
    meta_test = get_test_features(test, tfidf_w, tfidf_c, lda)

    return meta_train, meta_test


def get_meta_test_features(data, tfidf_w, tfidf_c, lda):
    print "meta features extraction ..."
    features = get_test_features(data, tfidf_w, tfidf_c, lda)
    return features


def train_models():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

#    train_X = get_features(train)
#    train_Y = get_labels(train)
#
#    test_X = get_features(test)
#    test_ids = get_test_ids(test)
#
    n_samples = len(train_Y)

#    print ("n_samples: %d, n_features: %d" %  train_X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    #predict_and_sub(train_X, train_Y.values, test.toarray(),
#                    test_ids, predict_ridge)

    #predictions = predict_ridge(X_train,y_train,X_test)
    #predictions = predic_24_models(X_train,y_train,X_test)
    #predictions = predic_multiple_model(X_train,y_train,X_test)
    #predictions = predict_24_models(X_train,y_train,X_test, Ridge(alpha=1.0))

    #score = rmse_score(y_test,predictions)
    #print 'RMSE score: %.6f' % score

#    for train_ix, test_ix in KFold(len(X_train), n_folds=5):
#        train_raw = X_train[train_ix]
#        train_labels = y_train[train_ix]
#        test_raw = X_train[test_ix]
#
#        #train_features, tfidf_w, tfidf_c, lda = get_train_features(train_raw)
#        #test_features = get_test_features(test_raw, tfidf_w, tfidf_c, lda)
#
#        meta_train, meta_test = get_meta_features(train_raw, test_raw)
#
#        pred_cv = predict_ridge(meta_train, train_labels, meta_test)
#        print 'RMSE score: %.6f' % rmse_score(y_train[test_ix], pred_cv)

#        pred_cv = predict_ridge(X_train[train_ix],y_train[train_ix],
#                                X_test[test_ix])
#        print 'RMSE score: %.6f' % rmse_score(y_test[test_ix],pred_cv)

#    for train_ix,test_ix in KFold(len(X_train),n_folds = 5):
#        pred_cv = predict_ridge(X_train[train_ix],y_train[train_ix],
#                                X_test[test_ix])
#        print 'RMSE score: %.6f' % rmse_score(y_test[test_ix],pred_cv)

    test_ids = get_test_ids(test)
    meta_train_X, meta_test_X = get_meta_features(train_X, test_X)

    predict_and_sub(meta_train_X, train_Y.values, meta_test_X,
                    test_ids, predict_ridge)

    duration = time() - t0
    print "training time: %fs" % duration

if __name__ == "__main__":
    train_models()
