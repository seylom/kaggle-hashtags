'''
Created on Nov 24, 2013

@author: seylom
'''

import gensim
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from parameters import max_features_best_sub
from sklearn.decomposition import TruncatedSVD


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
        self.__char_vect = None
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
            self.__settings['wordcount'] = 3000
            self.__settings['charcount'] = 3000
            self.__settings['topic'] = 400
        else:
            if 'word' not in self.__settings:
                self.__settings['word'] = 3000
            if 'char' not in self.__settings:
                self.__settings['char'] = 3000
            if 'wordcount' not in self.__settings:
                self.__settings['wordcount'] = 3000
            if 'charcount' not in self.__settings:
                self.__settings['charcount'] = 3000
            if 'topic' not in self.__settings:
                self.__settings['topic'] = 400

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
                elif feature == 'charcount':
                    result = self.__extract_charcount_features(data)
                elif feature == 'topic':
                    result = self.__extract_topic_features(data)

                if result is not None:
                    all_results.append(result)

            features = hstack(all_results)

            if self.__svd is None:
                self.__svd = TruncatedSVD(n_components=200)
                self.__svd.fit(features)

            features = self.__svd.transform(features)

            return features
        else:
            return None

    def __extract_word_features(self, data):
        '''
        Extract tfidf word ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_words is None:
            data_result, tfidf = self.__word_features(data,
                        num_features=self.__settings['word'])

            self.__tfidf_words = tfidf
        else:
            data_result, _ = self.__word_features(data,
                            tfidf=self.__tfidf_words)

        return data_result

    def __extract_char_features(self, data):
        '''
        Extract tfidf char ngram from the corpus using scikit learn
        tfidf vectorizer
        '''
        data_result = None
        if self.__tfidf_chars is None:
            data_result, tfidf = self.__char_features(data,
                        num_features=self.__settings['char'])

            self.__tfidf_chars = tfidf
        else:
            data_result, _ = self.__char_features(data,
                            tfidf=self.__tfidf_chars)

        return data_result

    def __extract_wordcount_features(self, data):
        '''
        Extract wordcount ngram from the corpus
        '''
        data_result = None
        if self.__word_vect is None:
            data_result, vect = self.__wordcount_features(data,
                        num_features=self.__settings['wordcount'])

            self.__word_vect = vect
        else:
            data_result, _ = self.__wordcount_features(data,
                            wordvect=self.__word_vect)

        return data_result

    def __extract_charcount_features(self, data):
        '''
        Extract charcount ngram from the corpus
        '''
        data_result = None
        if self.__char_vect is None:
            data_result, vect = self.__charcount_features(data,
                        num_features=self.__settings['charcount'])

            self.__char_vect = vect
        else:
            data_result, _ = self.__charcount_features(data,
                            charvect=self.__char_vect)

        return data_result

    def __extract_topic_features(self, data):
        '''
        Extract topics from the corpus using gensim
        '''
        data_result = None
        word_data = self.__extract_word_features(data)
        if self.__lda is None:
            data_result, lda = self.__topic_features(word_data,
                        num_topics=self.__settings['topic'])

            self.__lda = lda
        else:
            data_result, _ = self.__topic_features(word_data,
                            lda=self.__lda)

        return data_result

    def __wordcount_features(self, data, wordvect=None, num_features=10000):
        '''
        extracts word count features from the provided data
        '''
        if wordvect is None:
            wordvect = CountVectorizer(max_features=num_features,
                                       max_df=0.7,
                                       stop_words='english',
                                       ngram_range=(1, 4))
            wordvect.fit(data)

        features = wordvect.transform(data)

        return features, wordvect

    def __charcount_features(self, data, charvect=None, num_features=10000):
        '''
        extracts char count features from the provided data
        '''
        if charvect is None:
            charvect = CountVectorizer(max_features=num_features,
                                       max_df=0.7,
                                       analyzer='char',
                                       stop_words='english',
                                       ngram_range=(1, 4))
            charvect.fit(data)

        features = charvect.transform(data)

        return features, charvect

    def __word_features(self, data, tfidf=None, num_features=3000):
        '''
        extracts word ngram features from the provided data
        '''
        if tfidf is None:
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.001,
                                    max_df=0.8, max_features=num_features,
                                    analyzer="word", stop_words='english',
                                    strip_accents='unicode',
                                    ngram_range=(1, 4))
            tfidf.fit(data)

        features = tfidf.transform(data)

        return features, tfidf

    def __char_features(self, data, tfidf=None, num_features=3000):
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

    def __topic_features(self, data, lda=None, num_topics=400):
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
