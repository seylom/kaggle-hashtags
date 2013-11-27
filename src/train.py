'''
Created on Nov 21, 2013

@author: seylom
'''

import numpy as np
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import make_scorer
from sklearn.cross_validation import KFold
from utils import predict_ridge, predict_lasso, predict_elasticNet
from utils import  predict_multiple_model, predict_stacked_models, predict_rf
from utils import predict_knn, predict_three_models, predict_24_models
from utils import predict_two_models, predict_and_sub, predict_logit
from utils import predict_extra_tree, predict_decision_tree
from datahelper import load_dataset, get_test_ids
from utils import get_labels
from utils import rmse_score, rmse_score_simple
from features import FeatureExtractor
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression


def train_single():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    n_samples = len(train_Y)

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    predictions = []

    params = [1.0]
    #np.arange(1, 5)

    #tune ridge parameter
    for i in range(24):
        print "=> training target #%d" % i
        loop_start = True
        num_fold = 2
        rmse_best = 100

        feature_type = ['char']

        preds = None
        for param in params:
            #print "Cross validation for param = %.2f" % param
            rmse_avg = 0
            for train_ix, test_ix in KFold(len(X_train), n_folds=num_fold):
                train_raw = X_train[train_ix]
                train_labels = y_train[train_ix]
                test_raw = X_train[test_ix]

                fx = FeatureExtractor(settings={'word': 1000})
                meta_train = fx.get_features(train_raw, feature_type)
                meta_test = fx.get_features(test_raw, feature_type)

                if loop_start == True:
#                    print ("================================================")
#                    print ("n_samples: %d, n_features: %d" % meta_train.shape)
                    loop_start = False

                pred_cv = predict_decision_tree(meta_train, train_labels[:, i],
                                        meta_test,
                                        param=param)
                score_val = rmse_score_simple(y_train[test_ix, i], pred_cv)
                print 'RMSE score: %.6f' % score_val

                rmse_avg += score_val / float(num_fold)

            print 'Average RMSE score: %.6f' % rmse_avg
            if rmse_avg < rmse_best:
                best_alpha = param

        gx = FeatureExtractor(settings={'word': 100})
        Xd_train = gx.get_features(X_train, feature_type)
        Xd_test = gx.get_features(X_test, feature_type)

        preds = predict_decision_tree(Xd_train, y_train[:, i], Xd_test,
                                      param=param)
        predictions.append(preds)
        print 'best RMSE %.2f' % rmse_best
        print 'Best alpha is %.2f' % best_alpha

    full_preds = np.matrix(predictions).T
    print full_preds.shape
    print 'Overall RMSE score: %.6f' % rmse_score(y_test, full_preds)

    duration = time() - t0
    print "training time: %fs" % duration


def train_models():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    n_samples = len(train_Y)

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    loop_start = True
    num_fold = 3
    rmse_best = 100
    alpha_range = np.arange(8, 15, 0.2)

    fx = FeatureExtractor()

    feature_types = ['word', 'char', 'wordcount', 'topic']

    #tune ridge parameter
    for alpha in alpha_range:
        print "Cross validation for alpha = %.2f" % alpha
        rmse_avg = 0
        for train_ix, test_ix in KFold(len(X_train), n_folds=num_fold):
            train_raw = X_train[train_ix]
            train_labels = y_train[train_ix]
            test_raw = X_train[test_ix]

            meta_train = fx.get_features(train_raw, ['word'])
            meta_test = fx.get_features(test_raw,  ['word'])

            if loop_start == True:
                print ("================================================")
                print ("n_samples: %d, n_features: %d" % meta_train.shape)
                loop_start = False

            pred_cv = predict_ridge(meta_train, train_labels, meta_test,
                                    param=alpha)
            score_val = rmse_score(y_train[test_ix], pred_cv)
            print 'RMSE score: %.6f' % score_val

            rmse_avg += score_val / float(num_fold)

        print 'Average RMSE score: %.6f' % rmse_avg
        if rmse_avg < rmse_best:
            best_alpha = alpha

    print 'best RMSE %.2f' % rmse_best
    print 'Best alpha is %.2f' % best_alpha


def train_blend():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    n_samples = len(train_Y)

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    loop_start = True
    num_fold = 5
    rmse_avg = 0

    feature_type = ['char']

    for train_ix, test_ix in KFold(len(X_train), n_folds=num_fold):
        train_raw = X_train[train_ix]
        train_labels = y_train[train_ix]
        test_raw = X_train[test_ix]

        meta_train1, meta_test1 = get_extracted_features(['wordcount', 'char'],
                                                       train_raw, test_raw)

        meta_train2, meta_test2 = get_extracted_features(['word', 'topic'],
                                                       train_raw, test_raw)

        if loop_start == True:
            print ("================================================")
            print ("n_samples: %d, n_features: %d" % meta_train1.shape)
            loop_start = False

        pred_cv1 = predict_ridge(meta_train1, train_labels, meta_test1)
        pred_cv2 = predict_ridge(meta_train2, train_labels, meta_test2,
                                 param=2.5)

        pred_cv = 0.7 * pred_cv1 + 0.3 * pred_cv2

        score_val1 = rmse_score(y_train[test_ix], pred_cv1)
        score_val2 = rmse_score(y_train[test_ix], pred_cv2)

        score_val = rmse_score(y_train[test_ix], pred_cv)

        print 'RMSE score model 1: %.6f' % score_val1
        print 'RMSE score model 2: %.6f' % score_val2

        print 'RMSE score for blended model: %.6f' % score_val

        rmse_avg += score_val / float(num_fold)

    print 'Average RMSE %.6f' % rmse_avg

#    test_ids = get_test_ids(test)
#    meta_train_X, meta_test_X = get_meta_features(train_X, test_X)
#
#    print ("n_samples: %d, n_features: %d" % meta_train_X.shape)
#
#    predict_and_sub(meta_train_X, train_Y.values, meta_test_X,
#                    test_ids, predict_ridge)
#
    duration = time() - t0
    print "training time: %fs" % duration


def train():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    n_samples = len(train_Y)

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    loop_start = True
    num_fold = 5
    rmse_avg = 0

    feature_type = ['wordcount', 'char']

#    for train_ix, test_ix in KFold(len(X_train), n_folds=num_fold):
#        train_raw = X_train[train_ix]
#        train_labels = y_train[train_ix]
#        test_raw = X_train[test_ix]
#
#        meta_train, meta_test = get_extracted_features(feature_type,
#                                                       train_raw, test_raw)
#
#        if loop_start == True:
#            print ("================================================")
#            print ("n_samples: %d, n_features: %d" % meta_train.shape)
#            loop_start = False
#
#        pred_cv = predict_ridge(meta_train, train_labels, meta_test)
#
#        score_val = rmse_score(y_train[test_ix], pred_cv)
#
#        print 'RMSE score: %.6f' % score_val
#
#        rmse_avg += score_val / float(num_fold)
#
#    print 'Average RMSE %.6f' % rmse_avg

    test_ids = get_test_ids(test)
    meta_train_X, meta_test_X = get_extracted_features(feature_type,
                                                       train_X, test_X)

    print ("n_samples: %d, n_features: %d" % meta_train_X.shape)

    predict_and_sub(meta_train_X, train_Y.values, meta_test_X,
                    test_ids, predict_ridge)

    duration = time() - t0
    print "training time: %fs" % duration


def get_extracted_features(feature_type, train, test):
    fx = FeatureExtractor()
    meta_train = fx.get_features(train, feature_type)
    meta_test = fx.get_features(test, feature_type)
    return meta_train, meta_test


if __name__ == "__main__":
    #train_models()
    #train_single()
    #train_blend()
    train()
