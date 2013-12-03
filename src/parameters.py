'''
Created on Nov 25, 2013

@author: seylom
'''

#Ridge Model: on 10000 samples
#
#Best score: -0.163809
#Best parameters set:
#    clf__alpha: 10.0
#    features__char__analyzer: 'char'
#    features__char__max_df: 0.8
#    features__char__max_features: 3000
#    features__char__ngram_range: (1, 7)
#    features__char__stop_words: 'english'
#    features__word__max_df: 0.8
#    features__word__max_features: 4000
#    features__word__ngram_range: (1, 2)
#    features__word__stop_words: 'english

#Ridge Model: on 5000 samples
#
#Best score: -0.168251
#Best parameters set:
#    clf__alpha: 10.0
#    features__char__analyzer: 'char'
#    features__char__max_df: 0.5
#    features__char__max_features: 3000
#    features__char__ngram_range: (1, 7)
#    features__char__stop_words: 'english'
#    features__char__use_idf: True
#    features__word__max_df: 0.5
#    features__word__max_features: 4000
#    features__word__ngram_range: (1, 2)
#    features__word__stop_words: 'english'

#===========================================================

#Best score: -0.168281
#Best parameters set:
#    clf__estimator1: (SVC(C=0.001, cache_size=200, class_weight=None,
#                        coef0=0.0,degree=3,
#  gamma=100.0, kernel='rbf', max_iter=-1, probability=False,
#  random_state=None, shrinking=True, tol=0.001, verbose=False), False)
#    clf__estimator2: (Ridge(alpha=10.0, copy_X=True, fit_intercept=True,
#    ``````````````````````````````````max_iter=None,
#   normalize=False, solver='auto', tol=0.001), False)
#    clf__estimator3: (ExtraTreesRegressor(bootstrap=False,
#                compute_importances=None,
#          criterion='mse', max_depth=None, max_features='auto',
#          min_density=None, min_samples_leaf=10, min_samples_split=5,
#          n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
#          verbose=0), False)
#    features__char__analyzer: 'char'
#    features__char__max_df: 0.8
#    features__char__max_features: 3000
#    features__char__ngram_range: (1, 7)
#    features__char__stop_words: 'english'
#    features__word__max_df: 0.8
#    features__word__max_features: 4000
#    features__word__ngram_range: (1, 2)
#    features__word__stop_words: 'english'

#Elastic Net model : 5000 samples
#
#Best score: -0.170170
#Best parameters set:
#    clf__alpha: 0.0002
#    clf__l1_ratio: 0.8
#    clf__max_iter: 10000
#    features__char__analyzer: 'char'
#    features__char__max_df: 0.7
#    features__char__max_features: 3000
#    features__char__ngram_range: (1, 7)
#    features__char__stop_words: 'english'
#    features__word__max_df: 0.7
#    features__word__max_features: 4000
#    features__word__ngram_range: (1, 2)
#    features__word__stop_words: 'english'
#    svd__n_components: 1000


#Multi model (SVM + RF + SGD_logit)
#
#Best score: -0.168281
#Best parameters set:
#    clf__estimator1: (SVC(C=0.001, cache_size=200, class_weight=None,
#                            coef0=0.0, degree=3,
#  gamma=100.0, kernel='rbf', max_iter=-1, probability=False,
#  random_state=None, shrinking=True, tol=0.001, verbose=False), False)
#    clf__estimator2: (RandomForestClassifier(bootstrap=True,
#                    compute_importances=None,
#            criterion='gini', max_depth=None, max_features='auto',
#            min_density=None, min_samples_leaf=10, min_samples_split=1,
#            n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
#            verbose=0), False)
#    clf__estimator3: (SGDClassifier(alpha=0.01, class_weight=None,
#                                epsilon=0.1, eta0=0.0,
#       fit_intercept=True, l1_ratio=0, learning_rate='optimal', loss='log',
#       n_iter=200, n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
#       rho=None, shuffle=False, verbose=0, warm_start=False), True)
#    features__char__analyzer: 'char'
#    features__char__max_df: 0.8
#    features__char__max_features: 3000
#    features__char__ngram_range: (1, 7)
#    features__char__stop_words: 'english'
#    features__word__max_df: 0.8
#    features__word__max_features: 4000
#    features__word__ngram_range: (1, 2)
#    features__word__stop_words: 'english'
