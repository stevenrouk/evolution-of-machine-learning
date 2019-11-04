import os
import sys
sys.path.append('.')

import pickle

import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np

from flask import Flask, render_template, request, redirect, url_for

from forms import SearchForm
from src.analysis.inspect_topics import softmax
from src.analysis.topic_names import TOPIC_NAMES_3, TOPIC_NAMES_10, TOPIC_NAMES_20, TOPIC_NAMES_LOOKUP

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.join(os.path.split(FILE_DIRECTORY)[0], 'src')
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')

# Set up database connection
conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

# Read in data
df = pd.read_csv(ML_ONLY_FILEPATH)

# Load model
model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_model.pkl')
vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')
weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_weights_W.pkl')

print('loading model')
with open(model_filename, 'rb') as f:
    nmf_model = pickle.load(f)

print('loading vectorizer')
with open(vectorizer_filename, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

print('loading weights')
with open(weights_filename, 'rb') as f:
    W = pickle.load(f)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'

@app.route('/')
def index():
    return render_template('index.html')

def get_paper_loadings(idx):
    return W[idx] / sum(W[idx])

@app.route('/papers')
def papers():
    page = request.args.get('page', type=int)
    if page:
        data = df.iloc[(page-1)*20:page*20]
        page_num = page
    else:
        data = df.iloc[:20]
        page_num = 1
    
    data['loadings'] = list(map(get_paper_loadings, data.index))

    return render_template('papers.html', data=data, page_num=page_num, topics=TOPIC_NAMES_LOOKUP[10])

@app.route('/data')
def data():
    data = df.iloc[0]
    return render_template('data.html', data=data)

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        #return redirect(url_for('index'))
        #return form.search.data
        return redirect(url_for('results', query=form.search.data))
    return render_template('search.html', form=form)

@app.route('/results')
def results():
    query = request.args.get('query', type=str)
    if not query:
        return redirect(url_for('search'))

    vec = tfidf_vectorizer.transform([query])
    loadings = nmf_model.transform(vec)
    return str(loadings)

# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
