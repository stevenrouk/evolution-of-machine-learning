import os
import sys
sys.path.append('.')

from collections import Counter
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

from src.analysis.topic_names import TOPIC_NAMES_3, TOPIC_NAMES_10, TOPIC_NAMES_20, TOPIC_NAMES_LOOKUP

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


def print_influential_words_per_topic(H, vocabulary):
    """
    Print the most influential words of each latent topic.
    """
    for i, row in enumerate(H):
        top_ten = np.argsort(row)[::-1][:10]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_ten]))
        print(H[i, top_ten])
        print()


def get_influential_words_per_topic(H, vocabulary):
    """
    Print the most influential words of each latent topic.
    """
    word_loadings_data = []
    for i, row in enumerate(H):
        top_ten = np.argsort(row)[::-1][:10]
        words = vocabulary[top_ten]
        loadings = H[i, top_ten]
        word_loadings_data.append((i, words, loadings))

    return word_loadings_data


def get_year_df(df, date_col, year):
    year = str(year)
    return df[df[date_col].map(lambda x: x.split('-')[0]) == year]


def get_topics_for_year(df, num_topics=3, print_or_return='print'):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_ml = tfidf_vectorizer.fit_transform(df['description'])
    features = np.array(tfidf_vectorizer.get_feature_names())
    nmf_model = NMF(n_components=num_topics, random_state=42)
    W = nmf_model.fit_transform(tfidf_ml)
    H = nmf_model.components_

    if print_or_return == 'print':
        print_influential_words_per_topic(H, features)
    elif print_or_return == 'return':
        return get_influential_words_per_topic(H, features)
    else:
        print("'print_or_return' must be either 'print' or 'return'")
        return


@cli.command()
@click.option('--n-components', default=3, help='Number of NMF topics to train on.')
@click.option('--year', default=2019)
def print_topics_by_year(n_components, year=2000):
    """
    Look at topic loadings by year.
    """
    # Load data
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')
    df_ml['first_date'] = df_ml['dates'].str.split(',').str[0]
    df_ml_year = get_year_df(df_ml, 'first_date', year)
    get_topics_for_year(df_ml_year, num_topics=n_components)


@cli.command()
@click.argument('n-components', nargs=-1, type=int)
@click.option('--start-year', default=2000)
@click.option('--end-year', default=2019)
@click.option('--outfile', type=str, default=None)
def create_year_topics_df(n_components, start_year, end_year, outfile):
    """
    Look at topic loadings by year.
    """
    try:
        if outfile is None:
            n_components_filepath = '_'.join([str(x) for x in n_components])
            outfile = os.path.join(MODELS_DIRECTORY, f'year_topics_{start_year}_{end_year}_{n_components_filepath}.pkl')
        # Load data
        df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')
        df_ml['first_date'] = df_ml['dates'].str.split(',').str[0]
        data = []
        for year in range(start_year, end_year + 1):
            print(year)
            df_ml_year = get_year_df(df_ml, 'first_date', year)
            for n in n_components:
                print(n)
                word_loadings_data = get_topics_for_year(df_ml_year, num_topics=n, print_or_return='return')
                for topic_idx, words, loadings in word_loadings_data:
                    data.append((year, n, topic_idx, words, loadings))
            print("********************")
        
        year_topics_df = pd.DataFrame(data, columns=['year', 'num_topics', 'topic_idx', 'words', 'loadings'])
        with open(outfile, 'wb') as f:
            pickle.dump(year_topics_df, f)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    cli()
