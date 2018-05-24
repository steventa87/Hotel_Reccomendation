import sys
import pickle
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import os
from math import trunc
# don't use Flask anymore. Just do python server.py in terminal

from Person import Person

app = Flask(__name__)

# load in san francisco tfidf
san_francisco = pickle.load(open('san_francisco.pkle', 'rb'))

# headers for ratings dataframe
rating_headers = ['doc_id','hotel_name','hotel_url','street','city','state','country','zip','class','price','num_reviews','CLEANLINESS','ROOM','SERVICE','LOCATION','VALUE', 'COMFORT','overall_ratingsource']

# load in dataframe
df_hotels_san_francisco = pd.read_csv('data/san-francisco.csv', names = rating_headers).drop(0)

review_headers = ['date', 'review_title', 'full_review', 'nan']


# capitalizes all words in a string
def cap_all(long_string):
    return " ".join(word.capitalize() for word in long_string.split())


#decorator function will look for '/' and GET
@app.route('/', methods = ['GET'])
def home():
    return render_template('mpg.html')


@app.route('/search', methods = ['POST'])
def search_output():
    # req is request data from JS
    req = request.get_json()
    c = req['search']

    # prediction = list(model.predict([[c,h,w]]))
    # sends to mpg.js /inference and sends results to variable response

    # instantiate a new customer
    picky_customer = Person(san_francisco)

    # make a new search
    # search should be 'financial district near shopping and trendy'
    hotel_recs = picky_customer.search(c)
    inside_hotel_recs = hotel_recs[0]
    # return jsonify({'cylinders':c, 'horsepower':h, 'weight': w,
    #                 'predictions': prediction[0]})

    hotel_names = []

    # get the clean hotel name based on doc id
    for name in inside_hotel_recs:
        hotel_names.extend(df_hotels_san_francisco[df_hotels_san_francisco['doc_id'] == name]['hotel_name'].values)

    # after getting hotel names, capitalize all words
    for i, name in enumerate(hotel_names):
        hotel_names[i] = cap_all(name)

    # get top 3 reccomendation scores
    top_scores = [int(num*100) for num in sorted(picky_customer.cosine_similarities[0])[::-1][:3]]
    # get hotel's most recent reviews
    max_len = 180
    hotel_reviews = []

    for name in inside_hotel_recs:
        review = pd.read_csv('data/san-francisco/' + name,
                         names=review_headers ,sep='\t', encoding='latin-1').drop('nan', axis=1).head(1)
        hotel_reviews.append(review['full_review'].values[0][:max_len] + "...")

    # send results over to website
    return jsonify({'hotel1': hotel_names[0],
            'hotel2': hotel_names[1],
            'hotel3': hotel_names[2],
            'review1': hotel_reviews[0],
            'review2': hotel_reviews[1],
            'review3': hotel_reviews[2],
            'progressBar1': str(top_scores[0]),
            'progressBar2': str(top_scores[1]),
            'progressBar3': str(top_scores[2])
            })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3333, debug=True)
