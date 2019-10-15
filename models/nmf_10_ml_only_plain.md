# NMF Model - 10 Topics - Machine Learning Papers Only - Minimal Feature Engineering

These topics come out looking surprisingly good with very little processing!

```python
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
```

**topic 0 — machine learning / time series**

data learning machine time real analysis methods series classification sets

_[3.42728368 0.91277008 0.83532128 0.76814321 0.57162787 0.56180456
 0.55081958 0.54949722 0.48263448 0.47657914]_

**topic 1 — optimization**

optimization gradient convex matrix convergence stochastic problems method rank descent

_[1.0687652  1.02994206 0.97256241 0.8057191  0.80124985 0.73425117
 0.6541769  0.6343447  0.59084966 0.57058573]_

**topic 2 — neural networks / deep learning**

neural networks network deep training layer convolutional layers architecture architectures

_[1.8918856  1.83349385 1.81388113 1.2940825  0.72098239 0.60196585
 0.56018151 0.47429603 0.45948335 0.43142566]_

**topic 3 — reinforcement learning**

learning policy reinforcement agent rl agents control policies tasks reward

_[1.64126279 1.21072374 0.98319234 0.75104212 0.61998511 0.50565936
 0.49893129 0.49017762 0.4519282  0.44605572]_

**topic 4 — bayesian**

model models inference latent bayesian variational variables distribution gaussian posterior

_[1.64899387 1.40214615 1.0670907  0.90237447 0.82564889 0.72161334
 0.60184801 0.5413618  0.54109951 0.51124757]_

**topic 5 — graphs / graph ML**

graph graphs node nodes embedding structure edges network embeddings spectral

_[3.30040344 1.18060054 0.63559337 0.52764288 0.44510942 0.35780055
 0.26843218 0.25184101 0.22677716 0.21922406]_

**topic 6 — Generative adversarial networks**

adversarial attacks examples attack training robustness perturbations generative gan gans

_[2.63931028 1.14803923 0.94938991 0.78292505 0.59747972 0.58807406
 0.52772717 0.47664619 0.43747483 0.36871781]_

**topic 7 — image classification**

image task classification domain features images tasks model feature text

_[0.80407606 0.63467325 0.59190993 0.58487246 0.58369072 0.53298466
 0.51654471 0.49513119 0.46166163 0.44471193]_

**topic 8 — clustering**

clustering clusters cluster means algorithm spectral data algorithms points mixture

_[2.90242702 0.80282082 0.64550643 0.53641857 0.47609756 0.36148499
 0.29730119 0.25763193 0.2478913  0.21267755]_

**topic 9 — optimal solutions**

algorithm regret bounds bound problem optimal sample algorithms complexity lower

_[1.03135853 0.81881251 0.7700604  0.69906453 0.61737304 0.56426939
 0.5075246  0.47556412 0.43375194 0.42272133]_

---

These are the words with the highest summed loadings:

```python
array(['data', 'learning', 'graph', 'clustering', 'model', 'adversarial',
       'algorithm', 'networks', 'models', 'network', 'method', 'based',
       'training', 'neural', 'methods', 'deep', 'problem', 'algorithms',
       'proposed', 'approach', 'using', 'time', 'performance', 'results',
       'propose', 'paper', 'classification', 'new', 'machine', 'gradient'],
      dtype='<U65')
```