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


class Sentiment_Tfidf(Hotel_Map):

    ''' Class takes hotel names and hotel reviews and builds a tfidf vector'''

    def __init__(self, path):

        super().__init__(path)
        # self.map = map
        # self.hotel_names = map.hotel_names
        # self.hotel_reviews = map.hotel_reviews


    def _get_stemmed_tokenized(self):

        ''' get list of all words stemmed and tokenized '''
        self.hotel_vocab_stemmed = []
        self.hotel_vocab_tokenized = []

        for i in self.hotel_reviews:
            allwords_stemmed = self._tokenize_and_stem(i)
            self.hotel_vocab_stemmed.extend(allwords_stemmed)
            allwords_tokenized = self._tokenize_only(i)
            self.hotel_vocab_tokenized.extend(allwords_tokenized)

        self.hotel_vocab_frame = pd.DataFrame({'words': self.hotel_vocab_tokenized},
                                index = self.hotel_vocab_stemmed)
        print('Total stemmed and tokenized words: {}'.format(len(self.hotel_vocab_stemmed)))


    def _tokenize_and_stem(self, text):

        '''
        define a tokenizer and stemmer which returns the set of stems in the
        text that it is passed
        '''

        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        self.stems = [self.stemmer.stem(t) for t in filtered_tokens]
        return self.stems


    def _tokenize_only(self, text):

        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens