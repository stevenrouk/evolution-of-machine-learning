import os
import sys
sys.path.append('.')

import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np

from flask import Flask, render_template, request

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

# Create Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/papers')
def papers():
    page = request.args.get('page', type=int)
    if page:
        return render_template('papers.html', data=df.iloc[(page-1)*20:page*20], page_num=page)
    
    return render_template('papers.html', data=df.iloc[:20], page_num=1)

@app.route('/data')
def data():
    data = df.iloc[0]
    return render_template('data.html', data=data)

# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
