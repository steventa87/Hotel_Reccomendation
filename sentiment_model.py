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

import matplotlib.pyplot as plt

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import mpld3

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

from hotel_kmeans import Hotel_Kmeans


class Sentiment_Model(object):

    ''' Take in hotel and splits all reviews into sentiment clusters for topic modeling '''

    def __init__(self, city, file_name):

        self.city = city
        self.file_name = file_name
        self.path = os.getcwd() + '/data/' + self.city + '/'
        #
        self.review_headers = ['date', 'review_title', 'full_review', 'nan']
        #
        # load nltk's English stopwords as variable called 'stopwords'
        self.stopwords = nltk.corpus.stopwords.words('english')
        #
        # load nltk's SnowballStemmer as variabled 'stemmer'
        self.stemmer = SnowballStemmer("english")

        # lists to hold our sentiment clusters
        self.positives = []
        self.negatives = []
        self.neutrals = []

# ******************** importing data ******************** #

    def _get_hotel_reviews(self):

        ''' get all hotel reviews for city '''

        self.reviews = []

        df_reviews = pd.read_csv(self.path + self.file_name,
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

        total_reviews = len(self.positives) + len(self.negatives) + len(self.neutrals)
        print("Total reviews: {}".format(str(total_reviews)))
    # ******************** Count Vectorizer ******************** #

    def vectorize_clusters(self):

        ''' Go through each sentiment cluster and make tdidf of each '''

        self.tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                        stop_words = 'english',
                                        ngram_range=(3, 3),
                                        lowercase = True,
                                        token_pattern = r'\b[a-zA-Z]{3,}\b'
                                        #max_df = 1,
                                        #min_df = 1)
                                        )
        self.dtm_tf = self.tf_vectorizer.fit_transform(self.positives)
        print("positives count vectorizer shape: {}".format(self.dtm_tf.shape))

        # For negatives

        self.neg_tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                        stop_words = 'english',
                                        ngram_range=(3, 3),
                                        lowercase = True,
                                        token_pattern = r'\b[a-zA-Z]{3,}\b'
                                        #max_df = 1,
                                        #min_df = 1)
                                        )
        self.neg_dtm_tf = self.neg_tf_vectorizer.fit_transform(self.negatives)
        print("negatives count vectorizer shape: {}".format(self.neg_dtm_tf.shape))

        # LDA to lower dimensionality

        # we choose 5 topics to match our 5 rating categories of:
        # CLEANLINESS, ROOM, SERVICE, LOCATION, VALUE
        self.lda_tf = LatentDirichletAllocation(n_topics=5)
        self.lda_tf.fit(self.dtm_tf)
        print("Clustering for Positives: CLEANLINESS, ROOM, SERVICE, LOCATION, VALUE")
        # fit neg model
        self.neg_lda_tf = LatentDirichletAllocation(n_topics=5)
        self.neg_lda_tf.fit(self.neg_dtm_tf)
        print("Clustering for Negatives: CLEANLINESS, ROOM, SERVICE, LOCATION, VALUE")

        self.visualize_topic_models()

    # ******************** Visualization ******************** #

    def visualize_topic_models(self):

        ''' Visual topic models in jupyter notebook. Will not work in Jupyter Lab '''

        # visualiza CV for pos
        print("run: {}".format('pyLDAvis.sklearn.prepare(hotel.lda_tf, hotel.dtm_tf, hotel.tf_vectorizer)'))

        # visualiza CV for neg
        print("run: {}".format('pyLDAvis.sklearn.prepare(hotel.neg_lda_tf, hotel.neg_dtm_tf, hotel.neg_tf_vectorizer)'))
