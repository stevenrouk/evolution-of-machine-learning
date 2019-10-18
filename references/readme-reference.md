# Steven Rouk, Capstone 1 - Finding Patterns in Social Networks Using Graph Data

_Analysis of the evolution of the field of machine learning as seen through research papers on [arXiv.org](https://arxiv.org/)._

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

_(Note: Due the nature of the content in this dataset, there is some inappropriate language.)_

## Motivation

Network data—and more generally graph data—is everywhere.

Google models websites and knowledge as graphs. Facebook models social networks as graphs. Netflix models movie-watchers as graphs. Scientists model molecules and proteins as graphs. Airlines model flights as graphs.

And in this data exploration, we're looking at Reddit hyperlink networks as graphs.

<img src="images/colorful-graph-1.png" alt="colorful graph 1" width="400" height="400">

<sub><b>Figure: </b> Example graph using the pyvis library </sub>

With the prevalence of problems that can be viewed as graph problems, I wanted to know how to conduct an analysis on graph data. This isn't an idle interest for me, either. I think that most of the big problems we face today are fundamantally problems of social networks and information flow, problems like: improving our democracies; maintaining freedom of speech while combatting misleading information; connecting people to necessary resources; fighting extreme wealth inequality; creating effective organizations and communities; etc.

By learning how to represent and analyze graphs, I can learn how to help construct a better society for us all.

_(This was also an interesting and enlightening analysis project for me as someone who has spent only a couple hours on Reddit ever.)_

## The Data

In this project, I worked with the Stanford [Social Network: Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) dataset made available through [SNAP](https://snap.stanford.edu/index.html), the Stanford Network Analysis Platform.

<img src="images/snap_logo.png" alt="Stanford SNAP Logo">

<sub><b></b> The Stanford SNAP Logo </sub>

Specifically, this dataset catalogues hyperlinks between subreddits over the course of 2.5 years from January 2014 through April 2017. (A "subreddit" is a community or forum on Reddit dedicated to a specific topic. Subreddits are the basic community structure on Reddit.)

Intuitively, you can think about the data this way:

- Someone posts something on one subreddit: for example, the subreddit [askreddit](https://www.reddit.com/r/AskReddit/). This original post can be about anything.
- Someone else links to that post on a different subreddit, essentially sharing the content as something they think the subreddit would find interesting: for example, the original post on "askreddit" could get shared to the subreddit "bestof". This post could be positive ("look at this great post!"), negative ("this post is stupid"), or anywhere in between.

In this dataset, the original post is said to be in the **TARGET** subreddit, and the shared post referencing the TARGET is said to be in the **SOURCE** subreddit. So in our example above, "askreddit" is the TARGET and "bestof" is the SOURCE.

The dataset doesn't include the original post text or who posted it, but it does include a long array of properties about the post text. These properties include:

- Number of words
- Positive and negative sentiment
- 65 [LIWC](http://liwc.wpengine.com/) metrics, such as LIWC_Work, LIWC_Relig, LIWC_Swear, and LIWC_Anger.

The dataset also includes a timestamp, and a label of whether the reposted post is negative or positive/neutral towards the original post.

Here are a few example rows from the data:

| SOURCE_SUBREDDIT | TARGET_SUBREDDIT | POST_ID | TIMESTAMP | LINK_SENTIMENT | PROPERTIES |
| --- | --- | --- | --- | --- | --- |
| leagueoflegends | teamredditteams | 1u4nrps | 2013-12-31 16:39:58 | 1 | 345.0,298.0, ... |
| theredlion | soccer | 1u4qkd | 2013-12-31 18:18:37 | -1 | 101.0,98.0, ... |

## Graph Theory Terminology

Compared to working with normal tabular or text data, graph data introduced a whole new glossary of terms, topics, and methods that have to be used to discuss the data. This section will provide a brief introduction to the most important terms used in graph theory. Other terms will be defined as needed throughout the README.

These definitions come from (or are adapted from) the Wikipedia page [Glossary of graph theory terms](https://en.wikipedia.org/wiki/Glossary_of_graph_theory_terms), which was an invaluable asset during this project. Beneath each definition, there is a description of how the term translates to this specific dataset.

<img src="images/wikipedia-graph.svg" alt="Graph Image, from Wikipedia">

<sub><b>Figure: </b> Example Undirected Graph with 6 Nodes and 7 Edges, from Wikipedia </sub>

1. **graph.** The fundamental object of study in graph theory, a system of vertices connected in pairs by edges. Often subdivided into directed graphs or undirected graphs according to whether the edges have an orientation or not.
    * _The graph that we're working with is the system of all hyperlinks shared between subreddits over our 2.5 year period._
2. **vertex / node.** A vertex (plural vertices) is (together with edges) one of the two basic units out of which graphs are constructed.
    * _In our dataset, each node is a specific subreddit._
3. **edge.** An edge is (together with vertices) one of the two basic units out of which graphs are constructed. Each edge has two (or in hypergraphs, more) vertices to which it is attached, called its endpoints. Edges may be directed or undirected; undirected edges are also called lines and directed edges are also called arcs or arrows. In an undirected simple graph, an edge may be represented as the set of its vertices, and in a directed simple graph it may be represented as an ordered pair of its vertices.
    * _In our dataset we have directed edges which represent a post in one subreddit (the SOURCE subreddit) that references a post in another subreddit (the TARGET subreddit)._
4. **undirected graph.** An undirected graph is a graph in which the two endpoints of each edge are not distinguished from each other.
    * _In this project, we can create a representation of our graph which is undirected. In the undirected graph representation, edges just show that one of the subreddits has shared from the other (but we don't know which).
5. **directed graph.** A directed graph is one in which the edges have a distinguished direction, from one vertex to another.
    * _The hyperlink sharing graph is fundamentally a directed graph because we have information about one subreddit sharing a post from another. It means something different for subreddit A to share a post from B than for B to share a post from A._
6. **multigraph.** A multigraph is a graph that allows multiple adjacencies. (I.e., a graph where two nodes can have multiple edges running between them.)
    * _Our graph can be represented as a multigraph, where each hyperlink share between two subreddits is another directed edge between those two nodes._
7. **adjacent.** The relation between two vertices that are both endpoints of the same edge. (I.e., if two nodes are connected by an edge they are considered adjacent.)
    * _If two nodes in our dataset are adjacent, then that means one of them has shared a post from the other._
8. **degree.** The degree of a vertex in a graph is its number of incident edges. The degree of a graph G (or its maximum degree) is the maximum of the degrees of its vertices. In a directed graph, one may distinguish the _in-degree_ (number of incoming edges) and _out-degree_ (number of outgoing edges).
    * _The in-degree of a subreddit (node) in our dataset is the number of times that its posts have been referenced in another subreddit. The out-degree of a subreddit is the number of times that it has referenced posts from another subreddit._
9. **network.** A graph in which attributes (e.g. names) are associated with the nodes and/or edges.
    * _We're dealing with a network since our nodes have attributes (e.g. the subreddit name) and our edges have attributes (e.g. the array of text properties, or the timestamp, or the "weight" of an edge (which can mean various things)).

## Representing the Data Computationally

There are multiple ways to represent graphs computationally. For example, simple graphs can be represented as a square matrix where the row and column indices represent the individual nodes, a value of 1 at position (i,j) represents that there is an edge running from node i to node j, and a value of 0 at that position means there isn't. This matrix is usually called the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix). However, this approach alone isn't sufficient to represent more complicated data, such as networks where the nodes and edges have various attributes.

In the following sections we'll explore how more complex graphs can be represented.

### Non-Graph Representations: Pandas and Python

Before diving into a completely new domain, it can make sense to try your usual tools on the problem. Before I looked at how to represent nodes and edges in a graph-specific data structure, I pulled the data into [pandas](https://pandas.pydata.org/) and looked at some basic statistics.

<img src="images/histogram-of-link-sentiment.png" alt="Histogram of Link Sentiment">

<sub><b>Figure: </b> Histogram of Link Sentiment </sub>

```python
# Number of unique subreddits we're looking at.
print(len(set(df['SOURCE_SUBREDDIT']).union(set(df['TARGET_SUBREDDIT']))))
* 67180
```

Although I could have continued analyzing the data in pandas, I wanted to see what interesting analyses came out of graph-specific approaches.

### Custom Graph Class: Python Dictionaries and Classes

Before turning to off-the-shelf libraries, I wanted to see how much progress I could make on my own, using classes and built-in Python data structures. I spent the first day of the project working with toy datasets and building classes to build, navigate, and manipulate graphs.

In Python, you can represent graphs using dictionaries where the keys are nodes and the values are lists of nodes (representing edges). For example, here is a toy undirected graph dataset:

<img src="images/toy-graph-1.png" alt="Toy Undirected Graph" width="400" height="400">

<sub><b>Figure: </b> A toy undirected graph that I used for developing custom graph classes. There was a lot of whiteboard doodling during this project. </sub>

And here is the corresponding dictionary representation in Python:

```python
def simple_undirected_graph():
    """Returns a dictionary representing an example of an undirected graph."""
    g = {
        'A': ['B'],
        'B': ['A', 'C', 'D'],
        'C': ['B', 'D'],
        'D': ['B', 'C'],
        'E': []
    }

    return g
```

To modify the graph (such as add a node, add an edge, remove a node, etc.) you can write helper functions that operate on this dictionary. And to coordinate all of this further, you can wrap the data and the methods up into a class. You can see the code I wrote to represent graphs using dictionaries and classes in [UndirectedGraph.py](https://github.com/stevenrouk/capstone1-online-communities/blob/master/src/UndirectedGraph.py) and [DirectedGraph.py](https://github.com/stevenrouk/capstone1-online-communities/blob/master/src/DirectedGraph.py) in the repo.

I did try loading and analyzing the Reddit hyperlink using these custom-built classes, which worked fairly well for smaller subsets of the data. Larger subsets ran slowly at first, but I was able to **speed up my load time by 99%** by making my classes more efficient.

However, as fun as it would have been to completely rebuild graph data structures myself, there are robust and highly efficient libraries that already exist. On the second day of the project, I turned my attention to those.

### NetworkX

[NetworkX](https://networkx.github.io/) is a robust, widely-used library for storing and analyzing graph data in Python, which specific classes for graphs ([Graph](https://networkx.github.io/documentation/stable/reference/classes/graph.html)), directed graphs ([DiGraph](https://networkx.github.io/documentation/stable/reference/classes/digraph.html)), and multigraphs ([MultiDiGraph](https://networkx.github.io/documentation/stable/reference/classes/multidigraph.html)). Using these classes and my custom [DataLoader](https://github.com/stevenrouk/capstone1-online-communities/blob/master/src/DataLoader.py) class, I was able to load the full dataset in under a minute.

Now, NetworkX provided an easy API to access nodes and edges:

```python
import networkx as nx
from src.example_graphs import simple_undirected_graph

# The dictionary graph shown earlier
simple_graph = simple_undirected_graph()

G = nx.from_dict_of_lists(simple_graph)

print(G.nodes)
* ['A', 'B', 'C', 'D', 'E']

print(G.edges)
* [('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

print(G['B'])
* {'A': {}, 'C': {}, 'D': {}} # the extra {} here is a dictionary for edge attributes
```

With the graph loaded into NetworkX objects, I was ready to conduct analysis.

## Questions & Answers

For the majority of my analysis, I only worked with connections between nodes, direction of connections between nodes, and numbers of connections between nodes. A big area for future research is bringing in the text properties of the shared posts.

Let's get started!

### Who's the most connected? (Max Degree: In-Degree and Out-Degree)

My first question of the data was essentially the classic one we might ask about a network: "Who's the most popular?" This is what we're looking at here with "degree".

<img src="images/example-color-nodes-by-degree.png" alt="Example of coloring nodes by degree" width="600" height="600">

<sub><b>Figure: </b> Example of coloring nodes by degree with a demo graph </sub>

#### In-Degrees: Edges coming into a node

_Casually: "Who's getting talked about? Who's getting their stuff shared?"_

| subreddit | In-Degree (Uniques) | In-Degree (Total) |
| --- | --- | --- |
| askreddit | 5448 | 24295 |
| iama | 4508 | 11624 |
| pics | 3335 | 11728 |
| funny | 3031 | 10201 |
| videos | 2644 | 9216 |
| todayilearned | 2589 | 10292 |
| worldnews | 1770 | 8901 |
| gaming | 1746 | 5584 |
| news | 1610 | 7005 |
| gifs | 1591 | -- |
| politics | -- | 5511 |

#### Out-Degrees: Edges going out of a node

_Casually: "Who's gossiping about others? Who's sharing things from others?"_

| subreddit | Out-Degree (Uniques) | Out-Degree (Total) |
| --- | --- | --- |
| bestof | 3111 | 21170 |
| subredditdrama | 3020 | 23458 |
| titlegore | 2469 | 9501 |
| drama | 1413 | 5778 |
| hailcorporate | 939 |  -- |
| shitredditsays | 923 | 7394 |
| switcharoo | 918 | 5999 |
| the_donald | 798 |  -- |
| shitamericanssay | 793 | 5509 |
| botsrights | 792 | -- |
| shitpost | -- | 6658 |
| circlebroke2 | -- | 6089 |
| shitstatistssay | -- | 4278 |

So if we're specifically interested in the question of popularity, we only need to look at the in-degree numbers since out-degree is only a measure of how much a subreddit "talks" about others. And the popularity winner is... askreddit!

There are some other interesting results here:

**In-Degree Results**
- All of the top "in-degree" nodes look pretty official and fairly mainstream. They're fairly broad in topic ("funny", "news"), and the top two explicitly encourage interaction ("askreddit", "iama").
- You can also get a sense here for what most people find interesting: funny things, relatable human narrative, news, games, etc.
- It seems to make sense for top in-degree nodes to be more mainstream and widely applicable. Things that are more mainstream are going to be referenced and shared by more different kinds of people.

**Out-Degree Results**
- The top "out-degree" nodes look a little more edgy and niche on average; for example, "haircorporate" and "shitstatistssay". You can't tell what many of them are just by looking (e.g. "circlebroke2", "switcharoo", "titlegore") because of slang / jargon.
- There are also several top "out-degree" nodes explicitly for critiquing/complaining about things: "subredditdrama" and "circlebroke2", for example.
- However, there are also aggregation subreddits like "bestof" and "shitredditsays", which compiles the "best of" Reddit.
- It makes sense for aggregation-oriented subreddits to post the most links to other subreddits.

<img src="images/color-by-degree.png" alt="Example of coloring nodes by degree" width="500" height="500">

<sub><b>Figure: </b> Example of coloring nodes by degree with Reddit data </sub>

<img src="images/color-by-degree-zoomed.png" alt="Coloring by degree, zoomed in" width="500" height="500">

<sub><b>Figure: </b> Zoomed in so you can see the subreddit names </sub>

<img src="images/jiggly-graph-high-res.gif" alt="Example of interactive graph" width="500" height="500">

<sub><b>Figure: </b> In the actual HTML file, you can interact with the graph! (It's very jiggly!) </sub>

### How many distinct networks are there? (Component Analysis)

I actually stumbled my way into this question after somehow discovering that some of the nodes were disconnected. I was then able to find a NetworkX function that calculated the number of distinct, separated graphs (known as "components").

```python
# Not implemented for directed, so we'll cast to undirected.
connected_components = list(nx.connected_components(G_combined.to_undirected()))
component_length_counts = Counter([len(x) for x in connected_components])

# key-value pairs are: "Number of nodes in component": "how many graphs have that many nodes"
print(component_length_counts)
* Counter({
    65648: 1,
        9: 1,
        8: 1,
        7: 2,
        6: 2,
        5: 3,
        4: 14,
        3: 42,
        2: 646
})
```

### Who's friendly, and who's gossipy? (Sharing Reciprocity)

Another question I had was, "How might we see which subreddits tend to share things from each other?" To do this I created a metric called **reciprocity**.

#### Reciprocity Definition

I defined reciprocity between two nodes _n_ and _m_ as:

<a href="https://www.codecogs.com/eqnedit.php?latex=nm\&space;reciprocity=\frac{n_m}{n_m&space;&plus;&space;m_n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?nm\&space;reciprocity=\frac{n_m}{n_m&space;&plus;&space;m_n}" title="nm\ reciprocity=\frac{n_m}{n_m + m_n}" /></a>

where _n_<sub>_m_</sub> is the number of out-going edges from _n_ to _m_ and _m_<sub>_n_</sub> is the number of out-going edges from _m_ to _n_ (which is equivalent to in-coming edges to _n_ from _m_).

Intuitively, reciprocity is simply the fraction of traffic between _n_ and _m_ that originates from _n_. A reciprocity value of 1 means that _n_ shares things about _m_ but _m_ never shares things back (the "ignored" or "gossip" or "paparazzi" scenario); a reciprocity value of 0 means that _n_ gets things shared by _m_, but never shares things back (the "popular kid" or "movie star" scenario); and reciprocity values close to 0.5 mean that the sharing is about equal back and forth between _n_ and _m_ (the "friendly" scenario—unless perhaps it isn't friendly talk being exchanged).

#### Reciprocity Analysis - "Friendly" Scenario

There are about 34 pairs of subreddits (68 total) who have a reciprocity value between 0.4 and 0.6 and have shared over 100 things back and forth with each other. Here are some of them:

| subreddit #1 | subreddit #2 | reciprocity (from #1 - #2) | total shares |
| --- | --- | --- | --- |
| buildapc | techsupport | 0.48 | 351 |
| destinythegame | crucibleplaybook | 0.46 | 286 |
| subredditdrama | drama | 0.46 | 250 |
| hearthstone | competitivehs | 0.52 | 237 |
| smashbros | ssbpm | 0.47 | 212 |

So it looks like the subreddits that share back and forth the most are the ones with very similar topics, which makes sense. It's hard to tell without digging further, but I might hypothesize that these are friendly sharing relationships.

#### Reciprocity Analysis - "Ignored" Scenario

What about those subreddits that are always sharing thing from another subreddit, but never get any love back? Well, it turns out that a lot of these are the same subreddits as we saw in the top out-degree nodes earlier—the aggregators / sharers / complainers: "bestof", "circlebroke2", "shitredditsays", etc. There were also a huge number of "circlejerk" subreddits that showed up in this list: 9 of the top 33 most-ignored subreddits (for high share volume) had "circlejerk" in the name. Shows you something about the internet, I suppose. (Or just Reddit.)

### If you start at a random subreddit, where do you end up? (Random Walk Analysis)

The next question I had pertained to navigating the full graph. If you were to be dropped anywhere in the share network (any subreddit) and follow a random hyperlink to another subreddit, what subreddits would you stumble your way into?

To run this experiment, I created a RandomWalk class that held information about the graph, the current node, and the nodes that had been seen on the walk. I then created a lot of these walks and looked at the resulting aggregated nodes seen.

One interesting result that came of this was that most random walks through the graph dead-end at some point, where there are no out-bound edges to follow. In other words, you will usually eventually find yourself in a subreddit that hasn't shared anything from any other subreddit.

<img src="images/histogram-of-random-walk-steps.png" alt="Histogram of the number of steps taken during random walks" width=60% height=60%>

<sub><b>Figure: </b> Histogram of the number of steps taken during random walks </sub>

On the other hand, you may have originally gotten dropped into a subreddit that only ever linked to one other subreddit, and that subreddit linked back (one of the 2-node components mentioned above). In that case, you could only cycle back and forth, back and forth, forever and ever. Hope you like the Beach Boys.

<img src="images/beach-boys.png" alt="The Beach Boys subreddits, stuck in an endless loop" width=40% height=40%>

<sub><b>Figure: </b> The Beach Boys subreddits, stuck in an endless loop </sub>

But as far as which subreddits you would see the most, the answer isn't too surprising—you're going to see the most popular subreddits many times. (askreddit, iama, etc.)

#### Which subreddits are the most well-connected? (Centrality and PageRank)

This notion of random walks got me wondering—is there any way to determine how "central" any given node is to the network? Is this what I was trying to find with my "nodes seen" analysis?

It turns out there is a notion of node centrality, which also appears to be connected to the PageRank algorithm that Google based their search engine on. When you calculate that and look at the top nodes, you see the usual suspects—the most popular subreddits that we had found earlier, which also aligns with the nodes we saw on our random walks. I'd love to do more research here and see how correlated all of these metrics are.

### How do we visualize massive graphs? (Big Graph Data Visualization: Random Node Sampling)

All of this led me to what was probably my favorite part of the whole project—trying to visualize the graph.

First, let's just say that trying to visualize tens of thousands of data points in a graph doesn't go well.

#### First Attempt — Plotting Graph Subsets

We can plot subsets of the graph, but even these blow up very quickly...

<img src="images/science.png" alt="science subgraph" width="500" height="500">

<sub><b>Figure: </b> The subgraph of science and its adjacent nodes </sub>

<img src="images/science-animalrights.png" alt="science and animal rights subgraph" width="500" height="500">

<sub><b>Figure: </b> The subgraph of science, animal rights, and their adjacent nodes </sub>

<img src="images/science-animalrights-books.png" alt="science, animal rights, and books subgraph" width="500" height="500">

<sub><b>Figure: </b> The subgraph of science, animal rights, books, and their adjacent nodes </sub>

But, it's still better than nothing!

#### Second Attempt - Coloring the Graph

We can also color the graphs by metrics related to the nodes, such as node degree, which we saw earlier...

<img src="images/science-animalrights-books-colored-by-degree.png" alt="science, animal rights, and books subgraph, colored by degree" width="500" height="500">

<sub><b>Figure: </b> The subgraph of science, animal rights, books, and their adjacent nodes, colored by degree </sub>

#### Third Attempt - Random Subgraphs

As you can see in the previous graphs, one of the big problems here is that many subreddits are connected to dozens or hundreds of other subreddits, which makes the graph get out of hand very quickly.

All of this brought me to an idea: what if instead of showing _all_ of the neighboring nodes, we randomly selected a few and only showed them? And then what if we did the same for those nodes?

When you work this out (see my RandomSubgraph class if you're interested), you get something that looks like this...

<img src="images/random-sample-1.png" alt="askreddit small random sample" width="600" height="600">

<sub><b>Figure: </b> Small random sample, starting with askreddit </sub>

Here's what the graph looks like as it's growing...

<img src="images/subgraph-growth.gif" alt="growth of a random sample subgraph" width="400" height="400">

<sub><b>Figure: </b> Growth of a random sample subgraph </sub>

By fiddling with the parameters, you can get either a higher or lower percentage of nodes in the graph, and by tweaking some of the internals of the RandomSubgraph class I was also able to sample more of the nodes for nodes that didn't have many neighbors in an attempt to branch throughout more of the full graph structure. Color by degree again and you get this...

<img src="images/big-graph-colored-by-degree.png" alt="askreddit full graph random sample" width="600" height="600">

<sub><b>Figure: </b> Attempt at randomly sampling from the full graph, starting with askreddit </sub>

Which, frankly, is _beautiful_. And although we've only been coloring by degree so far, you can color by anything...

<img src="images/big-graph-colored-by-degree-rainbow.png" alt="askreddit full graph random sample rainbow!" width="600" height="600">

<sub><b>Figure: </b> Full graph sample, rainbow-ified! </sub>

Even though the rainbow graph isn't very useful, it does get us thinking about how we might be able to use the power of coloring and graph visualization to discover hidden patterns that are harder to detect.

## Future Research

I'm only just scratching the surface here, but I discovered a lot. Here are some areas for future research into both this dataset and graph data more generally.

- Look at Text Properties
    * I managed to discover a lot during this project, which is even more interesting because I was only analyzing the structure of the graph itself (along with the node names). I could discover a huge amount if I continued exploring the text properties in this dataset.
- More Random Sampling and Random Walks
    * I figured out how to use randomness in a few different ways to aid in my analysis of the graph. More work could definitely be done in this area.
- Graph Theory
    * Learn more about how to characterize graphs, and how to navigate them.
    * Dive into graph algorithms and metrics to expand my toolbox.
- Graph Machine Learning
    * Try applying some machine learning to graphs. There's a lot of interesting work being done here.
- Data Visualization
    * Clustering / communities / node aggregation. These are future topics for exploration, but I didn't get to them here.
    * Additional data visualization tools like Cytoscape (and py2cytoscape or the Dash Cytoscape wrapper), NetworkX Viewer, or perhaps even trying to translate graph data to geospatial and visualize it using Folium or another geospatial visualization tool. 
- Data Structures
    * Other graph data storage and analysis tools, like Neo4j, GraphX, etc.

## Technologies & Techniques Used

Technologies:
- Python
- [NetworkX](https://networkx.github.io/)
- [pyvis](https://pyvis.readthedocs.io/en/latest/) - Graph visualization with ability to load graphs from NetworkX.
- pandas
- NumPy
- Jupyter Notebooks

Techniques:
- Graph Theory
- Network Analysis
- Graph Visualization
- Reciprocity
- Centrality / PageRank
- Random Walks
- Random Sampling

## Gallery

Here's a gallery of some other cool images from this project!

<img src="images/ask-reddit-10-sphere.png" alt="" width="300" height="300"> <img src="images/adjacent-nodes-sphere.png" alt="" width="300" height="300">

<img src="images/animal-rights-15-random-sample.png" alt="" width="300" height="300"> <img src="images/barbell.png" alt="" width="300" height="300">

<img src="images/watts-strogatz.png" alt="" width="300" height="300"> <img src="images/complete-graph.png" alt="" width="300" height="300">

<img src="images/vegan-10.png" alt="" width="300" height="300"> <img src="images/rainbow2.png" alt="" width="300" height="300">

<img src="images/the-donald-rainbow.png" alt="" width="300" height="300"> <img src="images/ask-reddit-greyscale.png" alt="" width="300" height="300">

<img src="images/ask-reddit-jumbled.png" alt="" width="300" height="300"> <img src="/images/reddit-10-full.png" alt="" width="300" height="300">