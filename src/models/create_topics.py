import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Set global font size
plt.rcParams.update({'font.size': 16})

# Paths
SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')


def hand_label_topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    for i, row in enumerate(H):
        top_ten = np.argsort(row)[::-1][:10]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_ten]))
        label = input('please label this topic: ')
        hand_labels.append(label)
        print()
    return hand_labels


def print_influential_words_per_topic(H, vocabulary):
    '''
    Print the most influential words of each latent topic.
    '''
    for i, row in enumerate(H):
        top_ten = np.argsort(row)[::-1][:10]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_ten]))
        print(H[i, top_ten])
        print()


def get_top_words_by_loadings(H, features):
    return features[np.argsort(np.sum(H, axis=0))[::-1][:30]]


if __name__ == "__main__":
    # Load data
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_ml = tfidf_vectorizer.fit_transform(df_ml['description'])
    features = np.array(tfidf_vectorizer.get_feature_names())
    nmf_model = NMF(n_components=10, random_state=42)
    W = nmf_model.fit_transform(tfidf_ml)
    H = nmf_model.components_

    # This model specifically returns roughly these topics:
    hand_labeled_features = [
        'machine learning / time series',
        'optimization',
        'neural networks / deep learning',
        'reinforcement learning',
        'bayesian',
        'graphs / graph ML',
        'Generative adversarial networks',
        'image classification',
        'clustering',
        'optimal solutions'
    ]
