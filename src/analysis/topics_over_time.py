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

TOPIC_NAMES_3 = [
    'non-network ML models', #0
    'optimization / convergence', #1
    'neural networks' #2
]

TOPIC_NAMES_10 = [
    'machine learning / time series', #0
    'gradient / optimization / convergence', #1
    'neural networks / deep learning', #2
    'reinforcement learning', #3
    'variational bayesian', #4
    'graphs / graph ML', #5
    'ML attacks / GANs', #6
    'image / text / classification', #7
    'clustering', #8
    'algorithms / regret / optimization' #9
]

TOPIC_NAMES_20 = [
    '*** ? (privacy / large datasets?)', #0
    'gradient / optimization / convergence', #1
    'neural networks / deep learning', #2
    'reinforcement learning', #3
    'predictive modeling', #4
    'graphs / graph ML', #5
    'ML attacks / GANs', #6
    'classification', #7
    '*** ? (multi-armed bandit / regret?)', #8
    'matrices', #9
    '*** ? (ml research frameworks?)', #10
    'clustering', #11
    'feature selection', #12
    'kernel methods / Hilbert space', #13
    'NLP', #14
    'CNNs / images', #15
    'variational bayesian', #16
    'domain adaptation / transfer learning', #17
    'audio / speech recognition', #18
    'time series' #19
]

TOPIC_NAMES_LOOKUP = {
    3: TOPIC_NAMES_3,
    10: TOPIC_NAMES_10,
    20: TOPIC_NAMES_20
}


@click.group()
def cli():
    pass

@cli.command()
@click.option('--topic-idx', default=0, help='Index of the topic to look at')
@click.option('--n-components', default=10, help='Number of NMF topics (to look up model).')
@click.option('--show-outliers', is_flag=True, default=False)
@click.option('--start-year', default=2000)
@click.option('--end-year', default=2019)
@click.option('--all-topics', is_flag=True, default=False)
@click.option('--use-topic-names', is_flag=True, default=False)
@click.option('--output', default='')
def topic_evolution(
        topic_idx,
        n_components,
        show_outliers=False,
        start_year=2000,
        end_year=2019,
        all_topics=False,
        use_topic_names=False,
        output=''):
    """
    Look at the change in topic prevalence over time.
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

    features = np.array(tfidf_vectorizer.get_feature_names())
    H = nmf_model.components_

    print('loading DataFrame')
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')
    df_ml['first_date'] = df_ml['dates'].str.split(',').str[0]
    df_ml['year'] = df_ml['first_date'].map(lambda x: x.split('-')[0])
    years = df_ml['year']
    years = years.reset_index(drop=True)

    print('creating boxplot')

    sym = None if show_outliers else ''
    if all_topics:
        nrows = (W.shape[1] + 3) // 4
        ncols = 4
        last_row_min_idx = 4 * (nrows - 1)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        plt.rcParams.update({'font.size': 10})
        plt.tight_layout()
        #plt.suptitle("Prevalence of Topics Over Time")
        for topic_idx in range(W.shape[1]):
            W_series = pd.Series(W[:, topic_idx])
            W_series.name = 'topic_loadings'
            W_series = W_series.reset_index(drop=True)

            vals = []
            for year in range(start_year, end_year + 1):
                year = str(year)
                vals.append(W_series[years == year])

            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])
            axs = axs.flatten()
            if topic_idx >= last_row_min_idx:
                show_xlabel = True
            else:
                show_xlabel = False
            if use_topic_names:
                title = TOPIC_NAMES_LOOKUP[n_components][topic_idx]
            else:
                title = f'Topic {topic_idx}'
            create_topic_evolution_boxplot(
                vals,
                labels=range(start_year, end_year + 1),
                sym=sym,
                title=title,
                ax=axs[topic_idx],
                show_xlabel=show_xlabel,
                tight=True
            )
    else:
        W_series = pd.Series(W[:, topic_idx])
        W_series.name = 'topic_loadings'
        W_series = W_series.reset_index(drop=True)

        vals = []
        for year in range(start_year, end_year + 1):
            year = str(year)
            vals.append(W_series[years == year])

        fig, axs = plt.subplots(1, 1)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.flatten()
        if use_topic_names:
            title = TOPIC_NAMES_LOOKUP[n_components][topic_idx]
        else:
            title = f'Topic {topic_idx}'
        create_topic_evolution_boxplot(
            vals,
            labels=range(start_year, end_year + 1),
            sym=sym,
            title=title,
            ax=axs[0]
        )
        plt.gcf().subplots_adjust(bottom=0.18)
    if output:
        plt.savefig(output)
    else:
        plt.show()

def create_topic_evolution_boxplot(vals, labels, sym=None, title=None, ax=None, show_xlabel=True, tight=False):
    _ = ax.boxplot(vals, sym=sym)
    _ = ax.set_title(title)
    _ = ax.set_xticklabels(labels=labels, rotation=90)

    if tight:
        _ = ax.set_xticklabels(labels=labels, fontsize=6)
        _ = ax.set_title(title, fontsize=10)
        _ = ax.set_yticklabels('')
    if not show_xlabel:
        _ = ax.set_xticklabels('')


if __name__ == "__main__":
    cli()