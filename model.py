#import
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
import time


def write(data):
    with open('cars.csv','a') as f:
        line = f"{data['mpg']},{data['cylinders']},0,{data['horsepower']},{data['weight']},0,0,0,0\n"
        f.write(line)


def train():
    df = pd.read_csv('cars.csv')
    # y is lower case because it's a vector
    y = df.mpg
    # X is uppercase because it's a matrix
    X = df[['cylinders', 'horsepower', 'weight']]

    model = LinearRegression()
    model.fit(X, y)
    # pickling is taking something in ram and writing it to disk
    pickle.dump(model, open('linreg.p', 'wb'))


def reload():
    requests.get('http://localhost:3333/reload')


if __name__ == '__main__':
    while True:
        data = fetch()
        write(data)
        train()
        time.sleep(1)
        print(data)
