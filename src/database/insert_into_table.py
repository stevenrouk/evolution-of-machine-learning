import os
import sys
sys.path.append('.')

import pickle
import random

from bokeh.embed import components
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')

# Read in data
df = pd.read_csv(ML_ONLY_FILEPATH)

# Set up database connection
conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

for i, row in tqdm(df.iterrows(), total=len(df)):
    pass
    cur.execute(
        """INSERT INTO papers
            (identifier, url, title, set_spec, subjects, authors, dates, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        tuple(val for val in row.values)
    )
conn.commit()
