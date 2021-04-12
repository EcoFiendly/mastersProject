import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# t-SNE imports
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

home = os.path.expanduser('~')

with open(home+"/Data/comb/doc_topic_comp.pkl", "rb") as f:
    doc_topic_comp = pickle.load(f)
    f.close()

### plot t-SNE clustering chart
# array of topic weights
arr = pd.DataFrame(doc_topic_comp).fillna(0).values
# keep well separated points
arr = arr[np.amax(arr,axis=1) > 0.35]
# dominant topic number in each doc
topic_num = np.argmax(arr,axis=1)
# tSNE dimension reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)
# plot topic clusters using bokeh
output_notebook()
n_topics = 20
my_colors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title = "t-SNE Clustering of {} LDA Topics".format(n_topics),
    plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=my_colors[topic_num])
plot.savefig(home+"/Data/comb/tsne_plot.svg")