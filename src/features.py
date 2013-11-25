'''
Created on Nov 24, 2013

@author: seylom
'''

import gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_wordcount_features(data, wordvect=None, num_features=500):
    '''
    extracts word count features from the provided data
    '''
    if wordvect is None:
        wordvect = CountVectorizer(max_features=num_features, max_df=0.6,
                                stop_words='english')
        wordvect.fit(data)

    features = wordvect.transform(data)

    return features, wordvect


def get_word_features(data, tfidf=None, num_features=1600):
    '''
    extracts word ngram features from the provided data
    '''
    if tfidf is None:
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                      max_df=0.8, max_features=num_features,
                                      analyzer="word", stop_words='english',
                                      strip_accents='unicode',
                                      ngram_range=(1, 3))
        tfidf.fit(data)

    features = tfidf.transform(data)

    return features, tfidf


def get_char_features(data, tfidf=None, num_features=1600):
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


def get_topic_models_features(data, lda=None, num_topics=400):
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
