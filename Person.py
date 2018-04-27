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
from sklearn.metrics.pairwise import linear_kernel

from scipy.cluster.hierarchy import ward, dendrogram

import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import mpld3
from hotel_map import Hotel_Map


class Person(object):

    def __init__(self, city):

        self.city = city


    def _get_top_values(self, lst, n, labels):
        '''
        INPUT: LIST, INTEGER, LIST
        OUTPUT: LIST

        Given a list of values, find the indices with the highest n values.
        Return the labels for each of these indices.

        e.g.
        lst = [7, 3, 2, 4, 1]
        n = 2
        labels = ["cat", "dog", "mouse", "pig", "rabbit"]
        output: ["cat", "pig"]
        '''
        return [labels[i] for i in np.argsort(lst)[-1:-n-1:-1]]


    def search(self):

        ''' Get keyword searches from user '''

        user_inputs = []

        while True:
            keyboard = input("Enter search queries or quit: ")
            if keyboard.lower() == 'quit':
                print('\n')
                break
            user_inputs.append(keyboard)

        self.searches = [line.strip() for line in user_inputs]

        # use vectorizer to transform user input
        self.tokenized_inputs = self.city.tfidf_vectorizer.transform(self.searches)

        self.vectors = self.city.tfidf_matrix

        self.cosine_similarities = linear_kernel(self.tokenized_inputs, self.vectors)

        self.h_names = self.city.hotel_names

        for i, item in enumerate(self.searches):
            print(item)
            print(self._get_top_values(self.cosine_similarities[i], 3, self.h_names))
            print()
