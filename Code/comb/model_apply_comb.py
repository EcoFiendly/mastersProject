#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud
# from PIL import Image

# t-SNE imports
# from sklearn.manifold import TSNE
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Label
# from bokeh.io import output_notebook
# import matplotlib.colors as mcolors

# import cleaned df
with open("../../Data/df_clean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# load trained model
with open("../../Data/comb_global/model_17.pkl", "rb") as f:
    model = pickle.load(f)
    f.close()

# load corpus
corpus = corpora.MmCorpus("../../Data/comb_global/bow_corpus.mm")

# load dictionary
dic = corpora.Dictionary.load("../../Data/comb_global/dic.dict")

# apply model to produce list of topic composition for each document
topic_comp = [model[corpus[i]] for i in range(len(corpus))]

def topics_to_df(topics, num_topics):
    """
    Function converts the topic composition of documents to a dataframe
    """
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

# generate dataframe of document topic composition
doc_topic_comp = pd.concat([topics_to_df(topics, 17) for topics in topic_comp]).reset_index(drop=True).fillna(0)

with open("../../Data/comb_global/doc_topic_comp.pkl", "wb") as f:
    pickle.dump(doc_topic_comp, f)
    f.close()

with open("../../Data/comb_global/doc_topic_comp.pkl", "rb") as f:
    doc_topic_comp = pickle.load(f)
    f.close()

# ### plot t-SNE clustering chart
# # array of topic weights
# arr = pd.DataFrame(doc_topic_comp).fillna(0).values
# # keep well separated points
# arr = arr[np.amax(arr,axis=1) > 0.35]
# # dominant topic number in each doc
# topic_num = np.argmax(arr,axis=1)
# # tSNE dimension reduction
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
# tsne_lda = tsne_model.fit_transform(arr)
# # plot topic clusters using bokeh
# output_notebook()
# n_topics = 20
# my_colors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
# plot = figure(title = "t-SNE Clustering of {} LDA Topics".format(n_topics),
#     plot_width=900, plot_height=700)
# plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1]) #, color=my_colors[topic_num])
# show(plot)

# create list of redlistCategory, realm, systems
# red_list_cat = (df[df['rationale']!='']['redlistCategory']).values.tolist()

red_list_cat = []
realm = []
systems = []
for i in ['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions']:
    red_list_cat.extend(df[df[i]!='']['redlistCategory'].values.tolist())
    realm.extend(df[df[i]!='']['realm'].values.tolist())
    systems.extend(df[df[i]!='']['systems'].values.tolist())

# initialise dataframe for topic composition with corresponding red_list_cat, realm and systems
df_topic_comp = pd.concat([doc_topic_comp, pd.Series(red_list_cat), pd.Series(realm), pd.Series(systems)], axis=1)

# rename columns
df_topic_comp.columns = ['Monitor_endemics', 'Human_threats_to_habitat', 'Find_water', 'Agricultural_threats_to_forests', 'Monitor_population', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fish_use_trade', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Species_ecology', 'Population_trend', 'Conservation_actions', 'Red_list_category', 'Realm', 'Systems']

with open("../../Data/comb_global/df_topic_comp.pkl", "wb") as f:
    pickle.dump(df_topic_comp, f)
    f.close()

with open("../../Data/comb_global/df_topic_comp.pkl", "rb") as f:
    df_topic_comp = pickle.load(f)
    f.close()

#################################################################################
# change low risk conservation dependent and low risk near threatened to near threatened, low risk least concern to least concern
# take out regional assessments
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/conservation dependent', 'Near Threatened')
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/near threatened', 'Near Threatened')
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/least concern', 'Least Concern')
df_no_reg_ext = df_topic_comp[(df_topic_comp.Red_list_category != 'Regionally Extinct')]

# group by red list category and sum topic composition for each category
red_list_topic = df_no_reg_ext.drop(columns=['Realm', 'Systems']).groupby(['Red_list_category']).sum()
# normalize
red_list_topic_pct = red_list_topic/(df_no_reg_ext.drop(columns=['Realm', 'Systems']).groupby(['Red_list_category']).count())
# reorder and sort index
index_order = ['Least Concern', 'Near Threatened', 'Vulnerable', 'Endangered', 'Critically Endangered', 'Extinct in the Wild', 'Extinct']
red_list_topic_pct = red_list_topic_pct.reindex(index_order)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(red_list_topic_pct.loc[red_list_topic_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Red list category") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/red_list_cat_HM.svg")
plt.close('all')

################################################################################
# replace Marine|Marine with Marine
df_topic_comp['Systems'] = df_topic_comp['Systems'].replace('Marine|Marine', 'Marine')
# group by systems and sum topic composition for each category
systems_topic = df_topic_comp.drop(columns=['Red_list_category', 'Realm']).groupby(['Systems']).sum()
# normalize
systems_topic_pct = systems_topic/(df_topic_comp.drop(columns=['Red_list_category', 'Realm']).groupby(['Systems']).count())
# reorder and sort index
index_order = ['Terrestrial', 'Freshwater (=Inland waters)', 'Marine', 'Terrestrial|Freshwater (=Inland waters)', 'Freshwater (=Inland waters)|Marine', 'Terrestrial|Marine', 'Terrestrial|Freshwater (=Inland waters)|Marine']
systems_topic_pct = systems_topic_pct.reindex(index_order)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(systems_topic_pct.loc[systems_topic_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Systems category") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/systems_HM.svg")
plt.close('all')

################################################################################
# remove rows with na
realm_topic = df_topic_comp.dropna().drop(columns=['Red_list_category', 'Systems'])
# group docTopicSys by realm and sum topic composition for each realm
realm_topic = realm_topic.groupby(['Realm'])
# calculate percentage composition of topics per realm
realm_topic_pct = realm_topic.sum()/realm_topic.count()

# save realm_topic_pct.pkl
with open("../../Data/comb_global/realm_topic_pct.pkl", "wb") as f:
    pickle.dump(realm_topic_pct, f)
    f.close()

# load docTopicRlmPct
with open("../../Data/comb/realm_topic_pct.pkl", "rb") as f:
    realm_topic_pct = pickle.load(f)
    f.close()

# create column for realm from index
realm_topic_pct['Realm'] = realm_topic_pct.index

# experimenting with extracting by realm as condition
afro = realm_topic_pct.Realm.str.contains('Afrotropical')
anta = realm_topic_pct.Realm.str.contains('Antarctic')
aust = realm_topic_pct.Realm.str.contains('Australasian')
indo = realm_topic_pct.Realm.str.contains('Indomalayan')
near = realm_topic_pct.Realm.str.contains('Nearctic')
neot = realm_topic_pct.Realm.str.contains('Neotropical')
ocea = realm_topic_pct.Realm.str.contains('Oceanian')
pale = realm_topic_pct.Realm.str.contains('Palearctic')

afro_pct = realm_topic_pct[afro].drop(columns = 'Realm').sum()/realm_topic_pct[afro].drop(columns = 'Realm').count()
anta_pct = realm_topic_pct[anta].drop(columns = 'Realm').sum()/realm_topic_pct[anta].drop(columns = 'Realm').count()
aust_pct = realm_topic_pct[aust].drop(columns = 'Realm').sum()/realm_topic_pct[aust].drop(columns = 'Realm').count()
indo_pct = realm_topic_pct[indo].drop(columns = 'Realm').sum()/realm_topic_pct[indo].drop(columns = 'Realm').count()
near_pct = realm_topic_pct[near].drop(columns = 'Realm').sum()/realm_topic_pct[near].drop(columns = 'Realm').count()
neot_pct = realm_topic_pct[neot].drop(columns = 'Realm').sum()/realm_topic_pct[neot].drop(columns = 'Realm').count()
ocea_pct = realm_topic_pct[ocea].drop(columns = 'Realm').sum()/realm_topic_pct[ocea].drop(columns = 'Realm').count()
pale_pct = realm_topic_pct[pale].drop(columns = 'Realm').sum()/realm_topic_pct[pale].drop(columns = 'Realm').count()

indiv_realm_topic = pd.DataFrame(afro_pct)
indiv_realm_topic = pd.concat([indiv_realm_topic, anta_pct, aust_pct, indo_pct, near_pct, neot_pct, ocea_pct, pale_pct], axis = 1)
indiv_realm_topic.columns = ['Afrotropical', 'Antarctic', 'Australasian', 'Indomalayan', 'Nearctic', 'Neotropical', 'Oceanian', 'Palearctic']
indiv_realm_topic = indiv_realm_topic.transpose()

# try log10 for better visual representation
log_indiv_realm_topic = np.log10(indiv_realm_topic)

# save indiv_realm_topic
with open("../../Data/comb_global/indiv_realm_topic", "wb") as f:
    pickle.dump(indiv_realm_topic, f)
    f.close()

# open indiv_realm_topic
with open("../../Data/comb_global/indiv_realm?topic", "rb") as f:
    indiv_realm_topic = pickle.load(f)
    f.close()
    
# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(indiv_realm_topic.loc[indiv_realm_topic.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Realm category") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/realms_HM.svg")
plt.close('all')

################################################################################
topic_dict = {0: 'Monitor_endemics', 1: 'Human_threats_to_habitat', 2: 'Find_water', 3: 'Agricultural_threats_to_forests', 4: 'Monitor_population', 5: 'Range', 6: 'Common_threats', 7: 'Forest_ecosystem', 8: 'Population_structure', 9: 'Fish_use_trade', 10: 'Threat_distribution', 11: 'Forest_fragmentation', 12: 'Habitat_loss', 13: 'Area_based_protection', 14: 'Species_ecology', 15: 'Population_trend', 16: 'Conservation_actions'}

def get_topic_desig(doc_topics_list, topic_dict=topic_dict):
    '''
    Translate the numerical topics in a '.get_document_topics' list into the assigned topic designation
    '''
    descriptive_topic_list = []
    for pr in doc_topics_list:
        if pr[0] in topic_dict:
            descriptive_topic_list.append((topic_dict[pr[0]], pr[1]))
        else:
            descriptive_topic_list.append(('unassigned_topic', pr[1]))
    return descriptive_topic_list

# most relevant/likely topics for a word (aka 'term)
endemic_tops = model.get_term_topics(dic.token2id['endemic'], minimum_probability=0.001)
endemic_tops, get_topic_desig(endemic_tops)

test_tops = model.get_term_topics(dic.token2id[''], minimum_probability=0.001)
test_tops, get_topic_desig(test_tops)