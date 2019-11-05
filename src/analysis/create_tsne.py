from collections import Counter
import os
import pickle
import string

import click
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models.ldamulticore import LdaMulticore
from sklearn.manifold import TSNE

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

word_color_mappings = {
    0.1: "\033[0;30;47m{}\033[0;30;0m",
    0.5: "\033[0;30;46m{}\033[0;30;0m",
    1: "\033[0;30;43m{}\033[0;30;0m",
    2: "\033[0;30;45m{}\033[0;30;0m",
    3: "\033[0;30;41m{}\033[0;30;0m"
}


@click.group()
def cli():
    pass


@cli.command()
@click.option('--n-components', default=10, help='Number of NMF topics (to look up model).')
def create(n_components):
    tsne_outfile = os.path.join(MODELS_DIRECTORY, f'tsne_{n_components}.pkl')
    tsen_weights_outfile = os.path.join(MODELS_DIRECTORY, f'tsne_{n_components}_weights_W.pkl')
    weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_weights_W.pkl')

    if not os.path.exists(weights_filename):
        print("weights file doesn't exist")
        return

    print('loading weights')
    with open(weights_filename, 'rb') as f:
        W = pickle.load(f)

    print('instantiating model')
    tsne_model = TSNE(n_components=2)
    print('fitting model')
    W_tsne = tsne_model.fit_transform(W)

    try:
        print('saving tsne')
        with open(tsne_outfile, 'wb') as f:
            pickle.dump(tsne_model, f)
        print('saving weights')
        with open(tsen_weights_outfile, 'wb') as f:
            pickle.dump(W_tsne, f)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    cli()
