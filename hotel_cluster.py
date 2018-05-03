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
from hotel_tfidf import Hotel_Tfidf


class Hotel_Cluster(Hotel_Tfidf):

    ''' Class takes hotel tfidf and prints out K-means clusers'''

    def __init__(self, path):

        super().__init__(path)


    def get_clusters(self):

        ''' makes and prints clusters of our hotels '''

        # self.num_clusters = 5
        # *****changed
        self.num_clusters = 10

        self.km = KMeans(n_clusters=self.num_clusters)

        print("Fitting K-means")
        self.km.fit(self.tfidf_matrix)

        self.clusters = self.km.labels_.tolist()

        ranks = []

        for i in range(0,len(self.hotel_names)):
            ranks.append(i)

        # every group has 1000 ramdonly shuffled hotels, change indeces
        # self.hotels = { 'hotel_name': self.hotel_names[:100], 'rank': ranks[:100],
        #         'hotel_reviews': self.hotel_reviews[:100], 'cluster': self.clusters}

        self.hotels = { 'hotel_name': self.hotel_names, 'rank': ranks,
                'hotel_reviews': self.hotel_reviews, 'cluster': self.clusters}

        self.frame = pd.DataFrame(self.hotels, index = [self.clusters] , columns = ['rank',
                'hotel_name', 'hotel_reviews', 'cluster'])

        grouped = self.frame['rank'].groupby(self.frame['cluster'])

        print("Top terms per cluster:")
        print()
        self.order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        for i in range(self.num_clusters):
            print("Cluster %d words:" % i, end='')
            for ind in self.order_centroids[i, :20]:
                print(' %s' % self.hotel_vocab_frame.loc[self.terms[ind].split(' ')
                    ].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
            print()
            print()
            print("Cluster %d hotels:" % i, end='')
            for title in self.frame.loc[i]['hotel_name'].values.tolist():
                print(' %s,' % title, end='')
            print()
            print()
