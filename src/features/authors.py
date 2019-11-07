from collections import Counter
import os
import pickle

import click
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models.ldamulticore import LdaMulticore

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')


@click.group()
def cli():
    pass


@cli.command()
def create_author_counts():
    outfile = os.path.join(DATA_DIRECTORY_PROCESSED, 'author_counts.pkl')
    df = pd.read_csv(ML_ONLY_FILEPATH)
    author_counts = Counter('-----'.join(df['authors'].values.flatten()).split('-----'))
    with open(outfile, 'wb') as f:
        pickle.dump(author_counts, f)


@cli.command()
@click.option('--n', default=10, help='Number of authors.')
def get_top_authors(n):
    filepath = os.path.join(DATA_DIRECTORY_PROCESSED, 'author_counts.pkl')
    with open(filepath, 'rb') as f:
        author_counts = pickle.load(f)
    print(author_counts.most_common(n))


if __name__ == "__main__":
    cli()
