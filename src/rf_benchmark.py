'''
Created on Nov 15, 2013

@author: seylom
'''
 
import csv
import loader
import numpy as np
import pandas as pd
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import  OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, ridge,BayesianRidge
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model.ridge import RidgeClassifier
from scipy import sparse
from sklearn.preprocessing import Binarizer
from sklearn import preprocessing
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import scipy as sc
import re

def stopWords():
    text = open('stopwords.txt').read()
    words = text.split(",") 
    return words
    
def rmse_score(target,predictions):
    return np.sqrt(np.sum(np.array(np.array(predictions)-target)**2)/ (len(target)*24.0))
    
def load_data():
      
    train = pd.read_csv('train.csv',nrows=30000)
    test = pd.read_csv('test.csv')
    
    idx = 0
    state_dic = {}
    #state = train.ix[:,2]
    state_map = []
    tweets = []
    #build map, clean tweets
    
    p = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    
    for i,row in train.iterrows():
        tweets.append(p.sub(" ",row['tweet']))
        if row['state'] not in state_dic:
            state_dic[row['state']] = idx
            idx +=1
        state_map.append(state_dic[row['state']])
        
    #train_data = train['tweet'].map(lambda x: p.sub(" ",x))
    #test_data = test['tweet'].map(lambda x: p.sub(" ",x))
    
    tfidf = TfidfVectorizer(sublinear_tf=True,min_df=0.001, max_df=0.8, max_features = 10000,
                            stop_words = 'english', strip_accents='unicode')
    
    tfidf.fit(np.hstack((train['tweet'],test['tweet'])))
    
    X_tfidf = tfidf.transform(train['tweet'])
 
    X_state = np.matrix(state_map).transpose()

    #dvect = DictVectorizer()
    #X_state = dvect.fit_transform(state)
    
    #encoder = preprocessing.OneHotEncoder(categorical_features='all', n_values='auto')
    #X_res = encoder.fit_transform(X_state)
    X_res = X_state
    
    #X_state = encoder.transform(X_state)
    
    #binarizer = Binarizer()
    #X_res =binarizer.fit_transform(X_state)
                                
    #state = train['state'].values.transpose()
    #state = np.array(train.ix[:,2])
    #state = np.zeros((30000,2))
    
    #sparse_state = sparse.csr_matrix(X_res)
    #print X_tfidf.toarray().shape
    #print state.shape
    
    #X = sparse.hstack([X_tfidf, sparse_state])
    #X = X_tfidf
          
    #X = np.hstack((X_tfidf.toarray(),X_state)) 
    #X = sparse.hstack([X_tfidf, X_res])
    X = X_tfidf
    X = X.toarray()
    y = train.ix[:,4:]
    #y_indicator = LabelBinarizer().fit(y).transform(y)
    X_test = tfidf.transform(test['tweet'])
    test_ids = test['id']
    
    return X,y, X_test,test_ids

def create_benchmark(): 
    
    train_X, train_Y, test, test_ids = load_data()
    n_samples = len(train_Y)
    
    print("n_samples: %d, n_features: %d" % train_X.shape)
    
    tune = 0
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.2, random_state=1)
       
    t0 = time()
    if not tune:
        #clf = KNeighborsClassifier(n_neighbors=10)
        #clf = MultinomialNB(alpha=.01)
        #clf = NearestCentroid()
        #clf = RandomForestClassifier(n_estimators = 50, n_jobs = 1, verbose = True)
        #clf = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
        #              normalize=False, positive=False, precompute='auto', tol=0.0001,
        #              warm_start=False)
        
        clf = Ridge(alpha=1.0) 
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)    
        
        score = rmse_score(y_test,predictions)
        
        print 'RMSE score: %.6f' % score
        
#        final_clf = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, solver='auto', tol=0.001)
#        final_clf.fit(train_X, train_Y.values)
#        test_preds = final_clf.predict(test.toarray())    
#        
#        prediction = np.array(np.hstack([np.matrix(test_ids).T, test_preds])) 
#        col = '%i,' + '%f,'*23 + '%f'
#        np.savetxt('sub2.csv', prediction,col, delimiter=',')
    else:
        ###############################################################################
        # Set the parameters by cross-validation
        tuned_parameters_RF = [{'n_estimators':[50,75], 'min_samples_split':[1], 'max_depth':[1,5,10], 'min_samples_leaf':[1,5,10]}]

        tuned_parameters_NB = [{'n_neighbors':[5,10,15], 'leaf_size':[5,10,15,20]}]
        tuned_parameters_NN = [{'n_neighbors':[15,20], 'leaf_size':[20,30,40], 'metric':['minkowski'],'weights':['uniform','distance']}]
        tuned_parameters_OVR = [{ "estimator__C": [1,2,4,8], "estimator__kernel": ["poly","rbf"], "estimator__degree":[1, 2, 3, 4],}]
        tuned_parameters_Ridge = [{'alpha':[0.1,1.0,10.0]}]

        # Create a classifier: a support vector classifier
        # classifier = svm.SVC(C=2.82, cache_size=2000, coef0=0.0, gamma=0.0078, kernel='rbf',
        #                      probability=False, shrinking=True, tol=0.001, verbose=True)
          
        my_scorer = make_scorer(rmse_score, greater_is_better=False)

        scores = ['mean_squared_error']
     
        for score in scores:
            print("")
            print("Tuning hyper-parameters for %s" % score)
            print("")
         
    #        reg = GridSearchCV(SVR(kernel='rbf'), tuned_parameters, cv=skf, n_jobs=-1, scoring=score)
    #        reg.fit(X, Y)
    
            #clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters_NN, cv=5, scoring = my_scorer)
            #clf = GridSearchCV(OneVsRestClassifier(SVC(kernel="poly")), tuned_parameters_OVR, cv=5, scoring = score)
            #clf = GridSearchCV(OneVsRestClassifier(MultinomialNB()), tuned_parameters_OVR, cv=5, scoring = score)
            #clf = GridSearchCV(RandomForestClassifier(), tuned_parameters_RF, cv=5, scoring = my_scorer) 
            clf = GridSearchCV(Ridge(), tuned_parameters_Ridge, cv=3, scoring = my_scorer) 
            clf.fit(X_train, y_train)
         
            print("Best parameters set found on development set:")
            print("")
            print(clf.best_estimator_)
         
            print("")
            print("Grid scores on development set:")
            print("")
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
         
            print("")
            
            Y_pred = clf.best_estimator_.predict(X_test) 
            rms_error = rmse_score(y_test,Y_pred)
            
            print 'RMSE score: %.6f' % rms_error
 
            
    duration = time() - t0;    
    print "training time: %fs" % duration;  
    
#    prediction = np.array(np.hstack([np.matrix(test['id']).T, test_preds])) 
#    col = '%i,' + '%f,'*23 + '%f'
#    np.savetxt('sub1.csv', prediction,col, delimiter=',')
    

if __name__ == "__main__":
    create_benchmark()