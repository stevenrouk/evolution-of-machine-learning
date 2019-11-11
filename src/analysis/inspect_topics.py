import os
import pickle
import string

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .topic_names import TOPIC_NAMES_LOOKUP

# Set global font size
plt.rcParams.update({'font.size': 16})

# Paths
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
    weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_{n_components}_weights_W.pkl')

    if not os.path.exists(weights_filename):
        print("weights file doesn't exist")
        return

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


def softmax(v, temperature=1.0):
    '''
    A heuristic to convert arbitrary positive values into probabilities.
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s


def analyze_article(paper_idx, descriptions, titles, W, hand_labels):
    '''
    Print an analysis of a single NYT articles, including the article text
    and a summary of which topics it represents. The topics are identified
    via the hand-labels which were assigned by the user.
    '''
    print('Title:', titles[paper_idx])
    print()
    print('Description:\n', descriptions[paper_idx])
    probs = softmax(W[paper_idx], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        print('--> {:.2f}% {}'.format(prob * 100, label))
    print()


def get_colorful_words(topic_idx, paper_words, word_topic_loading_lookup, word_color_mappings):
    colorful_paper_description = []
    for word in paper_words:
        word_loadings = word_topic_loading_lookup.get(word)
        if word_loadings is not None:
            topic_loading = word_loadings[topic_idx]
            if topic_loading < 0.1:
                colorful_paper_description.append(word_color_mappings[0.1].format(word))
            elif topic_loading < 0.5:
                colorful_paper_description.append(word_color_mappings[0.5].format(word))
            elif topic_loading < 1:
                colorful_paper_description.append(word_color_mappings[1].format(word))
            elif topic_loading < 2:
                colorful_paper_description.append(word_color_mappings[2].format(word))
            else:
                colorful_paper_description.append(word_color_mappings[3].format(word))
        else:
            colorful_paper_description.append(word)

    return colorful_paper_description


def print_article_colored_by_word_loadings(df_ml, document_idx, word_topic_loading_lookup, topic_labels=None):
    translation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    paper_words = df_ml['description'][document_idx].lower().translate(translation).split()
    for topic_idx in range(10):
        colorful_words = get_colorful_words(topic_idx, paper_words, word_topic_loading_lookup, word_color_mappings)
        if topic_labels:
            print(f'TOPIC {topic_idx} - {topic_labels[topic_idx]}')
        else:
            print(f'TOPIC {topic_idx}')
        print()
        print(' '.join([colorful_words[i] for i in range(len(colorful_words))]))
        print()
        print('*' * 70)
        print()


@cli.command()
@click.option('--n-components', default=10, help='Number of NMF topics (to look up model).')
@click.option('--document-idx', default=None, type=int, help='Index of the document.')
@click.option('--document-title', default=None, type=str, help='Title of the document.')
def get_document_report(n_components, document_idx=None, document_title=None):
    """
    Print a topic report for a specific document.
    """
    if document_idx is None and document_title is None:
        print('you must enter either document-idx or document-title')
        return

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

    features = np.array(tfidf_vectorizer.get_feature_names())
    H = nmf_model.components_

    word_topic_loading_lookup = {}
    for i in range(len(features)):
        word_topic_loading_lookup[features[i]] = H[:, i]

    print()
    print('*' * 70)
    print()

    if document_idx is None:
        relevant_indices = df_ml[df_ml['title'].str.lower().str.contains(document_title)].index
        if len(relevant_indices) > 1:
            print("title string isn't unique, returning first result")
            document_idx = relevant_indices[0]
        elif len(relevant_indices) == 0:
            print('returned zero titles with that title string, try again')
        else:
            document_idx = relevant_indices[0]

    analyze_article(document_idx, df_ml['description'], df_ml['title'], W, TOPIC_NAMES_LOOKUP[n_components])

    print()
    print('*' * 70)
    print()

    # print paragraphs, one for each topic, words colored by topic loadings
    print_article_colored_by_word_loadings(df_ml, document_idx, word_topic_loading_lookup, topic_labels=TOPIC_NAMES_LOOKUP[n_components])


if __name__ == "__main__":
    cli()
