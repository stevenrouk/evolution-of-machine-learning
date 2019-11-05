import os
import sys
sys.path.append('.')

import pickle

from bokeh.embed import components
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np

from flask import Flask, render_template, request, redirect, url_for

from forms import SearchForm
from src.analysis.inspect_topics import softmax
from src.analysis.topic_names import TOPIC_NAMES_3, TOPIC_NAMES_10, TOPIC_NAMES_20, TOPIC_NAMES_LOOKUP
from src.analysis.document_similarities import get_similar_doc_idxs_to_loadings
from src.visualization.bokeh_demo import get_bokeh_plot
from src.visualization.scatter_plot import get_two_topic_scatterplot
from src.visualization.box_plot import get_boxplot, get_boxplot_demo

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.join(os.path.split(FILE_DIRECTORY)[0], 'src')
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')

# Set up database connection
conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

# Read in data
df = pd.read_csv(ML_ONLY_FILEPATH)

# Load model
model_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_model.pkl')
vectorizer_filename = os.path.join(MODELS_DIRECTORY, f'vectorizer_tfidf.pkl')
weights_filename = os.path.join(MODELS_DIRECTORY, f'nmf_10_weights_W.pkl')

print('loading model')
with open(model_filename, 'rb') as f:
    nmf_model = pickle.load(f)

print('loading vectorizer')
with open(vectorizer_filename, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

print('loading weights')
with open(weights_filename, 'rb') as f:
    W = pickle.load(f)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'


def get_paper_loadings(idx):
    return W[idx] / sum(W[idx])


@app.route('/')
def index():
    return render_template('index.html')


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
    similar_doc_idxs = get_similar_doc_idxs_to_loadings(data['loadings'].reshape(1, -1), W)
    similar_doc_idxs = similar_doc_idxs[similar_doc_idxs != paper_idx]
    similar_docs = df.iloc[similar_doc_idxs[:10]]
    return render_template('report.html', data=data, topics=TOPIC_NAMES_LOOKUP[10], similar_documents=similar_docs)


@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        #return redirect(url_for('index'))
        #return form.search.data
        return redirect(url_for('results', query=form.search.data))
    return render_template('search.html', form=form)


@app.route('/results')
def results():
    query = request.args.get('query', type=str)
    if not query:
        return redirect(url_for('search'))

    vec = tfidf_vectorizer.transform([query])
    loadings = nmf_model.transform(vec)
    similar_doc_idxs = get_similar_doc_idxs_to_loadings(loadings, W)

    return render_template('results.html', query=query, data=df.iloc[similar_doc_idxs[:10]])


@app.route('/topic-comparison')
def topic_comparison():
    return render_template('topic-comparison.html')


@app.route('/bokeh-demo')
def bokeh_demo():
    plot = get_bokeh_plot()
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/bokeh-scatter-plot')
def bokeh_scatter_plot():
    plot = get_two_topic_scatterplot(W[:, 0], W[:, 2])
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


@app.route('/tsne')
def tsne():
    with open(os.path.join(MODELS_DIRECTORY, 'tsne_10_weights_W.pkl'), 'rb') as f:
        W_tsne = pickle.load(f)
    plot = get_two_topic_scatterplot(W_tsne[:, 0], W_tsne[:, 1])
    script, div = components(plot)
    return render_template('data-visualization.html', plot_div=div, plot_script=script)


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
    return "Need to implement."


# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
