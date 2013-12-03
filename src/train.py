'''
Created on Nov 21, 2013

@author: seylom
'''
from __future__ import print_function

import numpy as np
from time import time
from pprint import pprint
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from utils import predict_ridge
from utils import predict_24_models
from utils import predict_and_sub
from utils import  save_prediction_subs
from datahelper import load_dataset, get_test_ids
from utils import get_labels
from utils import rmse_score
from features import FeatureExtractor
from scipy.sparse import hstack
from sklearn.linear_model import  Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from features import get_ridge_model
from features import  get_advanced_ridge2


def get_extracted_features(feature_type, train, test):
    """
    Extracts specified features from training and testing set

    parameters
    ----------

    feature_type: array of feature string (valid values: word, wordcont, char)
    train: the training set
    test : the testing set

    Returns:
    -------
        tuple: train, test
            features extracted from training and testing

    """

    fx = FeatureExtractor()
    meta_train = fx.get_features(train, feature_type)
    meta_test = fx.get_features(test, feature_type)
    return meta_train, meta_test


def do_cross_val(X_train, y_train, feature_type, estimator, nfolds=3):
    """
    Performs n fold cross validation on the dataset for a
    specified model and reports the rmse score

    parameters
    ----------

    X_train: numpy matrix (n_samples x n_features)
    y_train: numpy matrix (n_samples x n_targets)

    feature_type : array of string
        an array of string to be used for feature extraction
        valid values: word, wordcount, char

    nfolds: int, optional
        the number of folds

    Returns:
    -------
        tuple: train, test
            features extracted from training and testing

    """

    rmse_avg = 0
    loop_start = True

    for train_ix, test_ix in KFold(len(X_train), n_folds=nfolds):
        train_raw = X_train[train_ix]
        train_labels = y_train[train_ix]
        test_raw = X_train[test_ix]

        meta_train, meta_test = get_extracted_features(feature_type,
                                                train_raw, test_raw)

        if loop_start == True:
            print ("================================================")
            print ("n_samples: %d, n_features: %d" % meta_train.shape)
            loop_start = False

        pred_cv = predict_24_models(meta_train, train_labels,
                                                 meta_test,
                            Ridge(alpha=10.0))

        score_val = rmse_score(y_train[test_ix], pred_cv)

        print ('RMSE score: %.6f' % score_val)

        rmse_avg += score_val / float(nfolds)

    return rmse_avg


def do_submission():
    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    feature_type = ['wordcount', 'char']
    test_ids = get_test_ids(test)
    meta_train_X, meta_test_X = get_extracted_features(feature_type,
                                                       train_X, test_X)

    print ("n_samples: %d, n_features: %d" % meta_train_X.shape)

    predict_and_sub(meta_train_X, train_Y.values, meta_test_X,
                    test_ids, predict_ridge)


def do_gridsearch(X_train, y_train, pipeline, parameters, scorer):
    """
    performs grid search on the provided data and
    returns the best estimator parameters

    parameters:
    ----------

    X_train: numpy matrix (n_samples x n_features)
    y_train: numpy matrix (n_samples x n_targets)

    pipeline: Scikit learn Pipeline object
    parameters: a dictionary of parameter values for the pipeline
    scorer: the score function to be used

    """
    grid_search = GridSearchCV(pipeline, parameters, verbose=1,
                               scoring=scorer)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.6f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return grid_search.best_estimator_


def train():
    train, _ = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)

    n_samples = len(train_Y)

    X_train, _, y_train, _ = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    t0 = time()

    feature_type = ['wordcount', 'char']

    rmse_avg = do_cross_val(X_train, y_train,
                            feature_type, nfolds=3)

    print ('Average RMSE %.6f' % rmse_avg)

    duration = time() - t0
    print ("training time: %fs" % duration)


def train_model():

    train, _ = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)

    n_samples = len(train_Y)

    X_train, _, y_train, _ = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    scorer = make_scorer(rmse_score, greater_is_better=False)

    pipeline, parameters = get_ridge_model()
    #pipeline, parameters = get_three_predictor_model()
    #pipeline, parameters = get_elasticnet_model()
    #pipeline, parameters = get_three_predictor_model2()
    #pipeline, parameters = get_three_predictor_model3()
    #pipeline, parameters = get_ridge_model2()
    #pipeline, parameters = get_ridge_model3()
    #pipeline, parameters = get_advanced_ridge()

    do_gridsearch(X_train, y_train, pipeline, parameters, scorer)


def train_final():
    """
    train a model using grid search for parameter estimation
    """

    train, test = load_dataset()
    train_X = train['tweet']
    train_Y = get_labels(train)
    test_X = test['tweet']

    tfidf1 = TfidfVectorizer(max_df=0.6,
                             min_df=0.0000003,
                             stop_words='english',
                             strip_accents='unicode',
                             token_pattern='\w{1,}',
                             max_features=5000,
                             norm='l2',
                             use_idf=False,
                             smooth_idf=False,
                             ngram_range=(1, 3))

    tfidf2 = TfidfVectorizer(max_df=0.6,
                            analyzer='char',
                            min_df=0.00001,
                            stop_words='english',
                            strip_accents='unicode',
                            norm='l2',
                            max_features=5000,
                            ngram_range=(1, 7),
                            smooth_idf=False,
                            use_idf=False,
                            )

    tfidf1.fit(np.hstack((train_X, test_X)))
    tfidf2.fit(np.hstack((train_X, test_X)))

    train_X1 = tfidf1.transform(train_X)
    train_X2 = tfidf2.transform(train_X)

    train_X = hstack([train_X1, train_X2]).tocsr()

    n_samples = len(train_Y)

    X_train, _, y_train, _ = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2,
        random_state=1)

    scorer = make_scorer(rmse_score, greater_is_better=False)

    pipeline, parameters = get_advanced_ridge2()
    #pipeline, parameters = get_three_predictor_model()
    #pipeline, parameters = get_elasticnet_model()
    #pipeline, parameters = get_three_predictor_model2()
    #pipeline, parameters = get_three_predictor_model3()
    #pipeline, parameters = get_ridge_model2()
    #pipeline, parameters = get_ridge_model3()
    #pipeline, parameters = get_advanced_ridge()

    best_estimator = do_gridsearch(X_train, y_train, pipeline, parameters,
                                   n_jobs=5, verbose=1, scoring=scorer)

    #predict test data
    test_1 = tfidf1.transform(test_X)
    test_2 = tfidf2.transform(test_X)

    test_d = hstack([test_1, test_2])

    final_preds = best_estimator.predict(test_d)
    save_prediction_subs(test['id'], final_preds)


if __name__ == "__main__":
    #train_models()
    #train_single()
    #train_blend()
    #train()
    #predict_stacked_models()
    #predict_weighted_stacked_models()
    train_model()
    #train_final()
