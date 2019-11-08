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

from flask import Flask, render_template, request, redirect, url_for

from forms import SearchForm, BigSearchForm
from src.analysis.inspect_topics import softmax
from src.analysis.topic_names import TOPIC_NAMES_3, TOPIC_NAMES_10, TOPIC_NAMES_20, TOPIC_NAMES_LOOKUP
from src.analysis.document_similarities import get_similar_doc_idxs_to_loadings, get_similar_doc_idxs_to_tfidf
from src.visualization.bokeh_demo import get_bokeh_plot
from src.visualization.scatter_plot import get_two_topic_scatterplot, get_tsne_scatterplot
from src.visualization.box_plot import get_boxplot, get_boxplot_demo
from src.visualization.bar_chart import get_barchart

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.join(os.path.split(FILE_DIRECTORY)[0], 'src')
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')

# Read in data
df = pd.read_csv(ML_ONLY_FILEPATH)

# Load model and associated files
model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_model.pkl')
vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')
weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_weights_W.pkl')
tfidf_vectorized_corpus_filename = os.path.join(MODELS_DIRECTORY, f'tfidf_vectorized_corpus.pkl')

print('loading model')
with open(model_filename, 'rb') as f:
    nmf_model = pickle.load(f)

print('loading vectorizer')
with open(vectorizer_filename, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

print('loading weights')
with open(weights_filename, 'rb') as f:
    W = pickle.load(f)

print('loading tf-idf vectorized corpus')
with open(tfidf_vectorized_corpus_filename, 'rb') as f:
    tfidf_corpus = pickle.load(f)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
if os.environ['USER'] == 'stevenrouk':
    app.config['DEBUG'] = True
else:
    app.config['DEBUG'] = False


def normalize_paper_loadings(loadings):
    return loadings / sum(loadings)


def get_paper_loadings(idx):
    #return W[idx] / sum(W[idx])
    return normalize_paper_loadings(W[idx])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for('index')) 


@app.route('/blog-example')
def blog_example():
    return render_template('blog-example.html')


@app.route('/get-random-paper')
def get_random_paper():
    return redirect(url_for('report', paper_idx=random.choice(df.index)))


@app.route('/papers')
def papers():
    page = request.args.get('page', type=int)
    if page is not None and page > 0:
        data = df.iloc[(page-1)*20:page*20]
        page_num = page
        if len(data) == 0:
            return redirect(url_for('papers'))
    else:
        data = df.iloc[:20]
        page_num = 1

    data['loadings'] = list(map(get_paper_loadings, data.index))

    return render_template('papers.html', data=data, page_num=page_num, topics=TOPIC_NAMES_LOOKUP[10])


@app.route('/report')
def report():
    paper_idx = request.args.get('paper_idx', type=int)
    if paper_idx is None:
        return redirect(url_for('index'))

    data = df.iloc[paper_idx]
    data['loadings'] = get_paper_loadings(data.name)

    # Get similar docs by loadings
    similar_doc_idxs = get_similar_doc_idxs_to_loadings(data['loadings'].reshape(1, -1), W)
    similar_doc_idxs = similar_doc_idxs[similar_doc_idxs != paper_idx]
    similar_docs = df.iloc[similar_doc_idxs[:10]]

    # Get similar docs by tfidf
    tfidf_similar_doc_idxs = get_similar_doc_idxs_to_tfidf(tfidf_corpus[paper_idx], tfidf_corpus)
    tfidf_similar_doc_idxs = tfidf_similar_doc_idxs[tfidf_similar_doc_idxs != paper_idx]
    tfidf_similar_docs = df.iloc[tfidf_similar_doc_idxs[:10]]

    return render_template(
        'report.html',
        data=data,
        topics=TOPIC_NAMES_LOOKUP[10],
        similar_documents=similar_docs,
        similar_documents_tfidf=tfidf_similar_docs
    )


@app.route('/search', methods=['GET', 'POST'])
def search():
    search_form = SearchForm()
    big_search_form = BigSearchForm()
    if search_form.submit1.data and search_form.validate_on_submit():
        return redirect(url_for('results', query=search_form.search.data))
    if big_search_form.submit2.data and big_search_form.validate_on_submit():
        return redirect(url_for('loadings_results', query=big_search_form.search.data))
    return render_template('search.html', search_form=search_form, big_search_form=big_search_form)


@app.route('/loadings-results')
def loadings_results():
    query = request.args.get('query', type=str)
    if not query:
        return redirect(url_for('search'))

    # Get similar docs by loadings
    vec = tfidf_vectorizer.transform([query])
    loadings = nmf_model.transform(vec)
    similar_doc_idxs = get_similar_doc_idxs_to_loadings(loadings, W)
    normalized_loadings = normalize_paper_loadings(loadings[0])

    # Get similar docs by tfidf
    tfidf_similar_doc_idxs = get_similar_doc_idxs_to_tfidf(vec, tfidf_corpus)
    tfidf_similar_docs = df.iloc[tfidf_similar_doc_idxs[:10]]

    return render_template(
        'text-loadings-results.html',
        query=query,
        topics=TOPIC_NAMES_LOOKUP[10],
        loadings=normalized_loadings,
        similar_docs=df.iloc[similar_doc_idxs[:10]],
        similar_documents_tfidf=tfidf_similar_docs
    )


@app.route('/results')
def results():
    query = request.args.get('query', type=str)
    if not query:
        return redirect(url_for('search'))

    vec = tfidf_vectorizer.transform([query])
    loadings = nmf_model.transform(vec)
    similar_doc_idxs = get_similar_doc_idxs_to_loadings(loadings, W)

    # Get similar docs by tfidf
    tfidf_similar_doc_idxs = get_similar_doc_idxs_to_tfidf(vec, tfidf_corpus)
    tfidf_similar_docs = df.iloc[tfidf_similar_doc_idxs[:10]]

    return render_template(
        'results.html',
        query=query,
        data=df.iloc[similar_doc_idxs[:10]],
        similar_documents_tfidf=tfidf_similar_docs
    )


@app.route('/data-visualizations')
def data_visualizations():
    return render_template('list-of-data-visualizations.html')


@app.route('/bokeh-demo')
def bokeh_demo():
    plot = get_bokeh_plot()
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/bokeh-scatter-plot')
def bokeh_scatter_plot():
    data = pd.DataFrame({
        'x': W[:, 0],
        'y': W[:, 1]
        })
    plot = get_two_topic_scatterplot(data, 'x', 'y')
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/bokeh-small-scatter-plot')
def bokeh_small_scatter_plot():
    with open(os.path.join(MODELS_DIRECTORY, 'tsne_10_weights_W.pkl'), 'rb') as f:
        W_tsne = pickle.load(f)
    data = pd.DataFrame({
        'x': W_tsne[:, 0][:100],
        'y': W_tsne[:, 1][:100]
        })
    plot = get_two_topic_scatterplot(data, 'x', 'y')
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/tsne')
def tsne():
    topic_idx = request.args.get('topic_idx', type=int)
    with open(os.path.join(MODELS_DIRECTORY, 'tsne_10_weights_W.pkl'), 'rb') as f:
        W_tsne = pickle.load(f)
    if topic_idx is not None:
        pass
    else:
        topic_idx = 0
    loadings = W[:, topic_idx]
    data = pd.DataFrame({
        'x': W_tsne[:, 0],
        'y': W_tsne[:, 1],
        'topic_loadings': loadings,
        'titles': df['title']
        })
    title = f"Points colored by how much they belong to the \"{TOPIC_NAMES_LOOKUP[10][topic_idx]}\" category"
    plot = get_tsne_scatterplot(data, 'x', 'y', title=title, color_col='topic_loadings')
    script, div = components(plot)
    return render_template('tsne.html', plot_div=div, plot_script=script, topic_idxs=range(10), topic_idx=topic_idx)


@app.route('/boxplot-demo')
def boxplot_demo():
    plot = get_boxplot_demo()
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/boxplot')
def boxplot():
    plot = get_boxplot()
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/change-over-time')
def change_over_time():
    return render_template('change-over-time.html')


@app.route('/how-topics-are-defined')
def how_topics_are_defined():
    features = np.array(tfidf_vectorizer.get_feature_names())
    H = nmf_model.components_

    topic_names = TOPIC_NAMES_LOOKUP[10]
    top_words = []
    word_loadings = []

    for i, row in enumerate(H):
        top = np.argsort(row)[::-1][:30]
        top_words.append(features[top])
        word_loadings.append(H[i, top])

    scripts = []
    divs = []
    for i in range(10):
        data = pd.DataFrame({
            'x': top_words[i],
            'y': word_loadings[i]
            })
        color = cm.Set3.colors[i]
        color = tuple(round(c * 255) for c in color)
        hover_fill_color = tuple(abs(c - 40) for c in color)
        color = '#%02x%02x%02x' % color
        hover_fill_color = '#%02x%02x%02x' % hover_fill_color
        plot = get_barchart(data,
            'x',
            'y',
            title=topic_names[i],
            bar_color=color,
            hover_fill_color=hover_fill_color,
            y_axis_label='Loadings'
        )
        script, div = components(plot)
        scripts.append(script)
        divs.append(div)

    return render_template(
        'how-topics-are-defined.html',
        topic_names=topic_names,
        top_words=top_words,
        word_loadings=word_loadings,
        scripts=scripts,
        divs=divs,
    )


# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
