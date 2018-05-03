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

import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import mpld3

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Sentiment_Cluster(object):
        
    ''' Take in hotel and splits all reviews into sentiment clusters for topic modeling '''

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

        # lists to hold our sentiment clusters
        self.positives = []
        self.negatives = []
        self.neutrals = []
        self.all_sent_clusters = [self.positives, self.negatives, self.neutrals]

# ******************** importing data ******************** #
        
    def _get_file_names(self):

        ''' get all file_names under city '''
        self.file_names = os.listdir(self.path)


    def _get_hotel_reviews(self, file_name):

        ''' get all hotel reviews for city '''

        self.reviews = []

        df_reviews = pd.read_csv(self.path + file_name,
        names=self.review_headers, sep='\t', encoding='latin-1').drop('nan', axis=1)

        for review in df_reviews['full_review']:
            self.reviews.append(str(review))

    
    def cluter_review_sentiments(self):
        
        ''' splits all reviews into Positive, Negative, or Neutral clusters '''
        
        self.sent_analyzer = SentimentIntensityAnalyzer()
        
        # loop through all reviews and apply analyzer
        for review in self.reviews:
            if self.sent_analyzer.polarity_scores(review)['compound'] >= .3:
                self.positives.append(review)
            if self.sent_analyzer.polarity_scores(review)['compound'] <= -.3:
                self.negatives.append(review)
            else:
                self.neutrals.append(review)

# ******************** preprocessing ******************** #


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
    
'''    
# ******************** TFIDF ******************** #

    def vectorize_clusters(self):
        
        # Go through each sentiment cluster and make tdidf of each
    
    for sent in 
    # initiate TfidfVectorizer instance with the set hyperparameters that were mentioned above
    self.tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                             min_df=0.2, stop_words='english',
                             use_idf=True, tokenizer=self._tokenize_and_stem, ngram_range=(1, 3))
    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.hotel_reviews).toarray() # *****changed use self.hotel_reviews instead of self.reviews_grouped[0]
    print("tfidf shape: {}".format(self.tfidf_matrix.shape))

    # saving the matrix by dumping into a pickle file. set *protocol=2* if you are using python2
    # joblib.dump(self.tfidf_matrix,  'data/tfidf_matrix2.pkl')
    #tfidf_matrix = joblib.load('tfidf_matrix2.pkl')

    self.terms = self.tfidf_vectorizer.get_feature_names()
    # calculate cosine similarities
'''

