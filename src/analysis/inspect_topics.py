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

from topic_names import TOPIC_NAMES_3, TOPIC_NAMES_10, TOPIC_NAMES_20, TOPIC_NAMES_LOOKUP

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
@click.option('--n-components', default=10, help='Number of NMF topics (to look up model).')
def topic_words(n_components):
    """
    Print the most influential words of each latent topic.
    """
    model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_model.pkl')
    vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')

    if not os.path.exists(model_filename):
        print("model doesn't exist")
        return
    if not os.path.exists(vectorizer_filename):
        print("vectorizer doesn't exist")
        return

    print('loading model')
    with open(model_filename, 'rb') as f:
        nmf_model = pickle.load(f)

    print('loading vectorizer')
    with open(vectorizer_filename, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    features = np.array(tfidf_vectorizer.get_feature_names())
    H = nmf_model.components_

    for i, row in enumerate(H):
        top_ten = np.argsort(row)[::-1][:10]
        print('topic', i)
        print('-->', ' '.join(features[top_ten]))
        print(H[i, top_ten])
        print()

@cli.command()
@click.option('--n-components', default=10, help='Number of NMF topics (to look up model).')
@click.option('--topic-idx', default=0, help='Index of the topic.')
@click.option('--n-documents', default=5, help='Number of documents to print.')
@click.option('--title-only', is_flag=True, default=False, help='Only print the title of the document.')
def topic_documents(n_components=10, topic_idx=0, n_documents=5, title_only=False):
    """
    Print the most relevant documents for each latent topic.
    """
    model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_model.pkl')
    vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')
    weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_weights_W.pkl')

    if not os.path.exists(model_filename):
        print("model doesn't exist")
        return
    if not os.path.exists(vectorizer_filename):
        print("vectorizer doesn't exist")
        return
    if not os.path.exists(weights_filename):
        print("weights file doesn't exist")
        return

    print('loading model')
    with open(model_filename, 'rb') as f:
        nmf_model = pickle.load(f)

    print('loading vectorizer')
    with open(vectorizer_filename, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    print('loading weights')
    with open(weights_filename, 'rb') as f:
        W = pickle.load(f)
    
    print('loading DataFrame')
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')
    df_ml = df_ml.reset_index(drop=True)
    
    document_topic_loadings = W[:, topic_idx]
    top_document_idxs = np.argsort(document_topic_loadings)[::-1][:n_documents]
    top_document_descriptions = df_ml['description'][top_document_idxs].values
    top_document_titles = df_ml['title'][top_document_idxs].values

    print()
    print('*' * 70)
    print()
    print(f'TOPIC {topic_idx} - {TOPIC_NAMES_LOOKUP[n_components][topic_idx]}')
    print()
    for i, title in enumerate(top_document_titles):
        print('Title:', ' '.join(title.split()))
        if not title_only:
            print()
            print('Description:\n')
            print(top_document_descriptions[i])
        print()
        print('*' * 70)
        print()

if __name__ == "__main__":
    cli()