from __future__ import print_function
import pandas as pd
import numpy as np

import os
import codecs
import pickle
import json

import numpy as np
import pandas as pd
import random

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import mpld3
from hotel_map import Hotel_Map


class Hotel_Tfidf(Hotel_Map):

    ''' Class takes hotel names and hotel reviews and builds a tfidf vector'''

    def __init__(self, path):

        super().__init__(path)
        # self.map = map
        # self.hotel_names = map.hotel_names
        # self.hotel_reviews = map.hotel_reviews


    def vectorize(self):

        ''' make tfidf vector '''

        combined = list(zip(self.hotel_names, self.hotel_reviews))

        # randomly shuffle the hotels, keeping the names in order with the reviews
        random.shuffle(combined)

        # retrieve hotel names and hotel reviews in the new order, after shuffling
        self.hotel_names[:], self.hotel_reviews[:] = zip(*combined)

        # for testing purposes, you can use this snippet to run kmeans on 100
        # hotels of the whole corpus.
        # self.reviews_grouped = [self.hotel_reviews[i:i + 100] for i in range(0,
        #                     len(self.hotel_reviews), 100)]
        # print("There are {} hotels in our sample.".format(
        #                                             len(self.reviews_grouped[0])))

        # initiate TfidfVectorizer instance with the set hyperparameters that were mentioned above
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=self._tokenize_and_stem, ngram_range=(1, 3))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.hotel_reviews).toarray()

        # saving the matrix by dumping into a pickle file. set *protocol=2* if you are using python2
        # joblib.dump(self.tfidf_matrix,  'data/tfidf_matrix2.pkl')
        #tfidf_matrix = joblib.load('tfidf_matrix2.pkl')

        self.terms = self.tfidf_vectorizer.get_feature_names()
        # calculate cosine similarities
        self.dist = 1 - cosine_similarity(self.tfidf_matrix)
