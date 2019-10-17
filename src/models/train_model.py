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
@click.option('--n-components', default=10, help='Number of NMF topics.')
@click.option('--save-model', default=True)
@click.option('--save-weights', default=True)
@click.option('--save-vectorizer', default=True)
def create_nmf_model(n_components, save_model=True, save_weights=True, save_vectorizer=True):
    model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_model.pkl')
    vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')
    weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_weights_W.pkl')

    if os.path.exists(model_filename):
        print('model already created')
        return

    print('loading data')
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')

    if os.path.exists(vectorizer_filename):
        print('loading existing vectorizer')
        with open(vectorizer_filename, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_ml = tfidf_vectorizer.transform(df_ml['description'])
    else:
        print('creating and fitting tfidf-vectorizer')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_ml = tfidf_vectorizer.fit_transform(df_ml['description'])
    #features = np.array(tfidf_vectorizer.get_feature_names())

    print('creating and training NMF model')
    nmf_model = NMF(n_components=n_components, random_state=42)
    W = nmf_model.fit_transform(tfidf_ml)
    #H = nmf_model.components_
    #print_influential_words_per_topic(H, features)

    if save_model:
        print('saving model file')
        with open(model_filename, 'wb') as f:
            pickle.dump(nmf_model, f)

    if save_vectorizer:
        if not os.path.exists(vectorizer_filename):
            print('saving vectorizer file')
            with open(vectorizer_filename, 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
        else:
            print('vectorizer file already exists')
    
    if save_weights:
        if not os.path.exists(weights_filename):
            print('saving weights file')
            with open(weights_filename, 'wb') as f:
                pickle.dump(W, f)
        else:
            print('weights file already exists')

if __name__ == "__main__":
    cli()
