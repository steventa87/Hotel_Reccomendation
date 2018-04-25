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


class Hotel_Map(object):


    def __init__(self, path):

        self.path = path
        self.rating_headers = [
        'doc_id','hotel_name','hotel_url','street','city','state','country',
        'zip','class','price', 'num_reviews','CLEANLINESS','ROOM','SERVICE',
        'LOCATION','VALUE','COMFORT','overall_ratingsource']
        self.review_headers = ['date', 'review_title', 'full_review', 'nan']


        # self.df_ratings = pd.read_csv(self.path + '.csv', names = self.rating_headers
        #             ).drop(0)

        # load nltk's English stopwords as variable called 'stopwords'
        self.stopwords = nltk.corpus.stopwords.words('english')

        # load nltk's SnowballStemmer as variabled 'stemmer'
        self.stemmer = SnowballStemmer("english")


    def _get_file_names(self):

        ''' get all file_names under city '''
        self.file_names = os.listdir(self.path)


    def _get_hotel_reviews(self, file_name):

        ''' get all hotel reviews for city '''

        reviews = []

        df_reviews = pd.read_csv(self.path + file_name,
        names=self.review_headers, sep='\t', encoding='latin-1').drop('nan', axis=1)

        for review in df_reviews['full_review']:
            reviews.append(str(review))

        corpus = " ".join(reviews)

        return file_name, corpus


    def _get_all_hotels_and_reviews(self):

        ''' makes list of all hotels and hotel reviews '''

        self.hotel_names = []
        self.hotel_reviews = []

        for idx, name in enumerate(self.file_names):
            # for some reason file #31 isn't working
            if idx == 31:
                continue
            hotel, all_reviews = self._get_hotel_reviews(name)
            self.hotel_names.append(hotel)
            self.hotel_reviews.append(all_reviews)

        print('There are {} hotels in the dataset.'.format(
                                                        len(self.hotel_names)))


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


    def make_map(self):

        ''' run all helper functions to get hotels and reviews '''
        self._get_file_names()
        self._get_all_hotels_and_reviews()
        self._get_stemmed_tokenized()
        self.hotel_vocab_frame.head()
