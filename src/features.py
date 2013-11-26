'''
Created on Nov 24, 2013

@author: seylom
'''

import gensim
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from parameters import max_features_best_sub
from sklearn.decomposition import TruncatedSVD


def get_full_features(train, test, num_features=1000):
    meta_train, tfidf_w, tfidf_c, cvect, lda = get_all_features(train,
                                         max_features_best_sub)

    result = get_all_features(test, max_features_best_sub,
                                 tfidf_words=tfidf_w,
                                 tfidf_chars=tfidf_c,
                                 lda=lda, cvect=cvect)

    meta_test = result[0]

    return meta_train, meta_test


def get_word_features(train, test, num_features=max_features_best_sub['word']):
    meta_train, tfidf = word_features(train, num_features=num_features)
    testinfo = word_features(test, tfidf=tfidf)

    meta_train = meta_train.toarray()
    meta_test = testinfo[0].toarray()
    return meta_train, meta_test


def get_char_features(train, test, num_features=max_features_best_sub['char']):
    meta_train, tfidf = char_features(train, num_features=num_features)
    testinfo = char_features(test, tfidf=tfidf)

    meta_train = meta_train.toarray()
    meta_test = testinfo[0].toarray()
    return meta_train, meta_test


def get_wordcount_features(train, test,
                           num_features=max_features_best_sub['wordcount']):

    meta_train, vect = wordcount_features(train, num_features=num_features)
    testinfo = wordcount_features(test, wordvect=vect)

    meta_train = meta_train.toarray()
    meta_test = testinfo[0].toarray()
    return meta_train, meta_test


def get_topic_features(train, test,
                       num_features=max_features_best_sub['topic']):

    word_train, tfidf = word_features(train, num_features=num_features)
    word_test, _ = word_features(test, tfidf=tfidf)

    meta_train, lda = topic_features(word_train, num_topics=num_features)
    testinfo = topic_features(word_test, lda=lda)

    meta_train = meta_train.toarray()
    meta_test = testinfo[0].toarray()
    return meta_train, meta_test


def get_all_features(data, settings, tfidf_words=None,
                     tfidf_chars=None,
                     cvect=None,
                     lda=None, num_features=1000):
    '''
    extracts features from the data
    '''

    X_tfidf_words, tfidf_words = word_features(data, tfidf=tfidf_words,
                                        num_features=settings['word'])

    X_tfidf_chars, tfidf_chars = char_features(data, tfidf=tfidf_chars,
                                        num_features=settings['char'])

    X_counts, cvect = wordcount_features(data, wordvect=cvect,
                                         num_features=settings['wordcount'])

    X_topics, lda = topic_features(X_tfidf_words, lda=lda,
                                   num_topics=settings['topic'])

    X = hstack([X_tfidf_words, X_tfidf_chars, X_topics, X_counts])

#    lsa = TruncatedSVD(max_features)
#    X = lsa.fit_transform(X)
#    features = X
    features = X.toarray()

    return features, tfidf_words, tfidf_chars, cvect, lda


def wordcount_features(data, wordvect=None, num_features=10000):
    '''
    extracts word count features from the provided data
    '''
    if wordvect is None:
        wordvect = CountVectorizer(max_features=num_features,
                                   max_df=0.7,
                                stop_words='english',
                                ngram_range=(1, 2))
        wordvect.fit(data)

    features = wordvect.transform(data)

    return features, wordvect


def word_features(data, tfidf=None, num_features=3000):
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


def char_features(data, tfidf=None, num_features=3000):
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


def topic_features(data, lda=None, num_topics=400):
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
        self.__lda = None
        self.__settings = None
        self.__svd = None

        self.__initialize_settings(settings)

    def __initialize_settings(self, settings):
        #setting up some defaults
        self.__settings = settings

        if self.__settings is None:
            self.__settings = {}
            self.__settings['word'] = 3000
            self.__settings['char'] = 3000
            self.__settings['wordcount'] = 10000
            self.__settings['topic'] = 500
        else:
            if 'word' not in self.__settings:
                self.__settings['word'] = 3000
            if 'char' not in self.__settings:
                self.__settings['char'] = 3000
            if 'wordcount' not in self.__settings:
                self.__settings['wordcount'] = 10000
            if 'topic' not in self.__settings:
                self.__settings['topic'] = 500

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
                elif feature == 'topic':
                    result = self.__extract_topic_features(data)

                if result is not None:
                    all_results.append(result)

            features = hstack(all_results)

            if self.__svd is None:
                self.__svd = TruncatedSVD(n_components=1000)
                self.__svd.fit(features)

            final_features = self.__svd.transform(features)

            return final_features
        else:
            return None

    def __extract_word_features(self, data):
        '''
        Extract tfidf word ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_words is None:
            data_result, tfidf = word_features(data,
                        num_features=self.__settings['word'])

            self.__tfidf_words = tfidf
        else:
            data_result, _ = word_features(data,
                            tfidf=self.__tfidf_words)

        return data_result

    def __extract_char_features(self, data):
        '''
        Extract tfidf char ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_chars is None:
            data_result, tfidf = char_features(data,
                        num_features=self.__settings['char'])

            self.__tfidf_chars = tfidf
        else:
            data_result, _ = char_features(data,
                            tfidf=self.__tfidf_chars)

        return data_result

    def __extract_wordcount_features(self, data):
        '''
        Extract wordcount ngram from the corpus
        '''
        data_result = None
        if self.__word_vect is None:
            data_result, vect = wordcount_features(data,
                        num_features=self.__settings['wordcount'])

            self.__word_vect = vect
        else:
            data_result, _ = wordcount_features(data,
                            wordvect=self.__word_vect)

        return data_result

    def __extract_topic_features(self, data):
        '''
        Extract topics from the corpus using gensim
        '''
        data_result = None
        word_data = self.__extract_word_features(data)
        if self.__lda is None:
            data_result, lda = topic_features(word_data,
                        num_topics=self.__settings['topic'])

            self.__lda = lda
        else:
            data_result, _ = topic_features(word_data,
                            lda=self.__lda)

        return data_result
