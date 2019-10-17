# Steven Rouk, Capstone 2 - The Evolution of Machine Learning

_Analysis of the evolution of the field of machine learning as discovered through natural language processing (NLP) techniques applied to research papers on [arXiv.org](https://arxiv.org/)._

---

**You can find me on LinkedIn here: [Steven Rouk - LinkedIn](https://www.linkedin.com/in/stevenrouk/)**

---

## Table of Contents
1. [Motivation](#motivation)
2. [The Data](#the-data)
3. [Graph Theory Terminology](#graph-theory-terminology)
4. [Representing the Data Computationally](#representing-the-data-computationally)
5. [Questions & Answers](#questions--answers)
    * [Who's the most connected? (Max Degree: In-Degree and Out-Degree)](#whos-the-most-connected-max-degree-in-degree-and-out-degree)
    * [How many distinct networks are there? (Component Analysis)](#how-many-distinct-networks-are-there-component-analysis)
    * [Who's friendly, and who's gossipy? (Sharing Reciprocity)](#whos-friendly-and-whos-gossipy-sharing-reciprocity)
    * [If you start at a random subreddit, where do you end up? (Random Walk Analysis)](#if-you-start-at-a-random-subreddit-where-do-you-end-up-random-walk-analysis)
    * [How do we visualize massive graphs? (Big Graph Data Visualization: Random Node Sampling)](#how-do-we-visualize-massive-graphs-big-graph-data-visualization-random-node-sampling)
6. [Future Research](#future-research)
7. [Technologies & Techniques Used](#technologies--techniques-used)
8. [Gallery](#gallery)

## Motivation

### A Brief History

It's no overstatement to say that machine learning—and data science more broadly—is revolutionizing our society.

Just for a bit of perspective:

- The first web browser was made available to the public in 1991.
- Deep Blue (by IBM) beat Garry Kasparov in chess in 1997.
- Google was founded in 1998.
- The first iPhone was released in 2007.
- Watson (once again by IBM) beat Jeopardy champions Brad Rutter and Ken Jennings in 2011.
- AlphaGo (by Google/DeepMind) beat a 9-dan professional Go player in 2016.
- The first self-driving car completed the first DARPA Grand Challenge in 2005. By the end of 2016, Google's fleet of self-driving cars had completed over 2,000,000 autonomous miles.
- In 2018, an NLP AI by Alibaba outperformed Stanford students on a reading and comprehension test, and Google releases "Duplex", an AI assistant with an incredibly human-like voice that can make reservations for you.

And keep in mind that thirty years ago, most homes in the US didn't have a personal computer. Ten years ago, most people in the US weren't using smartphones. These days, the computing power available to the average person is astronomically more than was available in previous years, which is democratizing access to the computing power needed to accomplish incredible technological feats using data. As one article puts it, thanks to Moore's Law ["Your smartphone is millions of times more powerful than all of NASA’s combined computing in 1969"](https://www.zmescience.com/research/technology/smartphone-power-compared-to-apollo-432/).

Whereas previously you needed supercomputers and teams of researchers to create algorithms even capable of recognizing hand-written digits, these days millions of people have the computing power needed to create sophisticated facial recognition algorithms. (And for those who don't posess the computing power on their personal machines, they can simply purchase it through platforms like [AWS](https://aws.amazon.com/machine-learning/).)

### A Rapidly Evolving Field

With this kind of quickly changing technological landscape, I was curious as to characteristics of machine learning and how those have changed over the last few decades. As a data scientist and machine learning practitioner, I'm constantly looking for ways to better understand the field and keep up with developments. If I could find a way to analyze the recent trajectory of machine learning, I would be in a better place from which to put it to good use.

## The Data

To get a sense of the evolution of machine learning, I turned to the research paper hosting website [arXiv.org](https://arxiv.org/) (which is a service of Cornell University), a site which bills itself as "Open access to 1,605,550 e-prints in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics." (Note: Thanks to [github.com/niderhoff/nlp-datasets](https://github.com/niderhoff/nlp-datasets) for pointing me in the right direction when I was looking for datasets.)

<img src="images/arXiv.png" alt="arXiv.org">

<sub><b></b> arXiv.org </sub>

ArXiv graciously makes the metadata for their whole collection of research papers available through an open API, which meant that I could download descriptions for all 1.6 million papers. For a single API call, the metadata was returned as an XML object with 1000 records, where each record looked like this:

<img src="images/arXiv_XML.png" alt="arXiv XML example">

<sub><b></b> Research Paper XML Record </sub>

### Pulling the Data

The first task was to pull all of the metadata and store it locally for analysis. A quick back-of-the-napkin calculation told me I should expect the full data pull to run about 3 GB in size, so I chose to store each response as a raw XML text file initially for later processing and analysis. In total, I made over 1,600 API calls over the course of 24 hours to pull all of the data.

### Processing the Data

While the full dataset was downloading, I wrote scripts to process the XML data into a series of structured CSV files with just the information I wanted: id, url, title, set, subjects, authors, date, and description. I ended up primarily using title, set, subjects, date, and description in my analysis, although I would like to conduct further analysis using authors.

After processing the XML files to individual CSVs, I used another script to combine all of these CSVs into a single CSV that could be directly loaded as a pandas DataFrame. I created another CSV file with just research papers that had the phrase "machine learning" in one of their subjects.

The final processed CSV file ended up being 1.7 GB, and the machine learning subset CSV file ended up as 81 MB. (This was down from a raw data size of 3.1 GB.)

## Exploratory Data Analysis (EDA)

First, I looked at the general distribution of papers by subject on arXiv.

<img src="images/number_of_papers_per_subject_macro.png" alt="arXiv papers by subject">

<sub><b></b> arXiv Papers by Subject </sub>

As you can see, the majority of the papers published on arXiv are physics papers—although according to their own analysis, this might be changing with the rise of machine learning. (See https://arxiv.org/help/stats/2018_by_area/index for more.)

What I was really interested in was the research papers specifically related to machine learning. I filtered the data down to just these papers by finding all papers that had "machine learning" as part of one of the subjects listed. There were 48,564 such papers, which accounted for about 3% of the total number of papers pulled from the site. The distribution of the subjects listed for these papers is shown below.

<img src="images/top_twenty_machine_learning_subjects.png" alt="machine learning paper subjects">

<sub><b></b> Machine Learning Paper Subjects </sub>

Although the two subjects with "machine learning" in them are at the top of the list (which we would expect), the subsequent top subjects give us some view into what various sub-fields of machine learning might be: computer vision, language, optimization, robotics, etc.

This subset of over 48,000 paper descriptions was the corpus that I worked with after this point.

### Machine Learning Papers Over Time

One of the most startling results from this analysis was also one of the simplest: looking at the number of machine learning papers published over time.

<img src="images/machine_learning_papers_over_time.png" alt="machine learning papers published over time">

<sub><b></b> Machine Learning Papers Published Over Time </sub>

There are only 33 papers from before the year 2000 with the subject of machine learning, and until 2007 there were less than 100 new papers per year. In 2007, there were 122 new machine learning papers—in 2012, there were 1,629—and in 2019 so far, there have been 14,504 new papers on machine learning. (And still have over two months left in the year!)

It's hard to underscore how significant of a shift this is. Not only has machine learning already had a monumental impact on society, but each year more time and energy is devoted to developing it as a field. It's hard to imagine what the next ten to twenty years are going to bring with this kind of investment being poured into machine learning.

And, fittingly, this underscores the need for applying techniques like machine learning to understand the field of machine learning!

## Topic Modeling

The heart of the analysis involving using the natural language processing (NLP) technique of topic modeling to discover latent topics in the corpus of research paper descriptions. (Note: I didn't include the paper titles in my model, although I did display them as part of the analysis of my topics.) By using topic modeling to understand the research papers, I was essentially trying to discover cohesive sub-fields of machine learning. I used the technique of non-negative matrix factorization (NMF) to accomplish this.

### Text Featurization and Model Hyperparameters

I was able to achieve surprisingly good results with very little preprocessing or fine-tuning. The steps shown below comprised my primary analysis pipeline for the majority of the project.

1. First, I converted the paper descriptions into tf-idf vectors using scikit-learn's TfidfVectorizer. I removed stop words as part of this process.
2. Then, I fit an NMF model using the number of topics I was interested in fitting. For the majority of the analysis, I looked at 10 topics and got good results—however, I also looked at 3, 15, 20, and 30 topics, which gave additional insight into the data.
3. Finally, I used the factored matrices `W` (the document-topic matrix) and `H` (the vocabulary-topic matrix) to look at the relationship between the text and the latent topics I had extracted.

Although I usually would have done more feature engineering through word lemmatization, custom stop words, or vocabulary reduction, I was surprised to find that my NMF model didn't appear to need it to return very good results. ()

### 10 Topics

So what topics was I able to discover? Using an NMF hyperparameter of 10 topics, these were the words most indicative of the ten topics:

| Topic Number | Topic Words |
| --- | --- |
| 0 | data learning machine time real analysis methods series classification sets |
| 1 | optimization gradient convex matrix convergence stochastic problems method rank descent |
| 2 | neural networks network deep training layer convolutional layers architecture architectures |
| 3 | learning policy reinforcement agent rl agents control policies tasks reward |
| 4 | model models inference latent bayesian variational variables distribution gaussian posterior |
| 5 | graph graphs node nodes embedding structure edges network embeddings spectral |
| 6 | adversarial attacks examples attack training robustness perturbations generative gan gans |
| 7 | image task classification domain features images tasks model feature text |
| 8 | clustering clusters cluster means algorithm spectral data algorithms points mixture |
| 9 | algorithm regret bounds bound problem optimal sample algorithms complexity lower |

--------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
