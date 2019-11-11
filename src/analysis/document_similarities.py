import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_doc_idxs_to_loadings(loadings, W):
    sims = cosine_similarity(loadings, W)[0]

    return np.argsort(sims)[::-1]


def get_similar_doc_idxs_to_tfidf(tfidf_vec, all_docs):
    sims = cosine_similarity(tfidf_vec, all_docs)[0]

    return np.argsort(sims)[::-1]
