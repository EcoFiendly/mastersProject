#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# from PIL import Image

# t-SNE imports
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors

# import cleaned df
with open("../../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# load trained model
with open("../../Data/comb/model_20.pkl", "rb") as f:
    model = pickle.load(f)
    f.close()

# load corpus
corpus = corpora.MmCorpus("../../Data/comb/bow_corpus.mm")

# load dictionary
dic = corpora.Dictionary.load("../../Data/comb/dic.dict")

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
doc_topic_comp = pd.concat([topics_to_df(topics, 20) for topics in topic_comp]).reset_index(drop=True).fillna(0)

with open("../../Data/comb/doc_topic_comp.pkl", "wb") as f:
    pickle.dump(doc_topic_comp, f)
    f.close()

with open("../../Data/comb/doc_topic_comp.pkl", "rb") as f:
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
red_list_cat = (df[df['rationale']!='']['redlistCategory']).values.tolist()

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
df_topic_comp.columns = ['Monitor_species', 'Area_based_protection', 'Assess_major_threat', 'Monitor_trade', 'Monitor_aquatic_habitat', 'Monitor_population_size', 'Human_threat_to_habitat', 'European_regional_assessment', 'Wide_distribution', 'Forest_habitat_loss_to_agriculture', 'Monitor_population_structure', 'Restricted_range', 'Coral_reef_utility', 'Commercial_fishing', 'Species_range_parameters', 'Conservation_action', 'Population_decline_habitat_loss', 'Forest_habitat', 'Population_information', 'Monitor_island_endemics', 'Red_list_category', 'Realm', 'Systems']

with open("../../Data/comb/df_topic_comp.pkl", "wb") as f:
    pickle.dump(df_topic_comp, f)
    f.close()

with open("../../Data/comb/df_topic_comp.pkl", "rb") as f:
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

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(red_list_topic_pct.loc[red_list_topic_pct.idxmax(axis=1).sort_values().index],
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
fig.savefig("../../Data/comb/red_list_cat_HM.svg")
plt.close('all')

################################################################################
# replace Marine|Marine with Marine
df_topic_comp['Systems'] = df_topic_comp['Systems'].replace('Marine|Marine', 'Marine')
# group by systems and sum topic composition for each category
systems_topic = df_topic_comp.drop(columns=['Red_list_category', 'Realm']).groupby(['Systems']).sum()
# normalize
systems_topic_pct = systems_topic/(df_topic_comp.drop(columns=['Red_list_category', 'Realm']).groupby(['Systems']).count())

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(systems_topic_pct.loc[systems_topic_pct.idxmax(axis=1).sort_values().index],
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
fig.savefig("../../Data/comb/systems_HM.svg")
plt.close('all')

################################################################################
# remove rows with na
realm_topic = df_topic_comp.dropna().drop(columns=['Red_list_category', 'Systems'])
# group docTopicSys by realm and sum topic composition for each realm
realm_topic = realm_topic.groupby(['Realm'])
# calculate percentage composition of topics per realm
realm_topic_pct = realm_topic.sum()/realm_topic.count()

# save docTopicRlmPct
with open("../../Data/comb/realm_topic_pct.pkl", "wb") as f:
    pickle.dump(realm_topic_pct, f)
    f.close()

# load docTopicRlmPct
with open("../../Data/comb/realm_topic_pct.pkl", "rb") as f:
    realm_topic_pct = pickle.load(f)
    f.close()

# top_topics = []
# for i in range(len(realm_topic_pct)):
#     top_topics.append(', '.join(realm_topic_pct.iloc[i].sort_values(ascending=False)[:5].index.values)) # top 5

    # topTopics.append(', '.join(docTopicRlmPct.iloc[i].sort_values(ascending=False).index.values)) # all topics

# create dataframe for each realm(s) and their top topics
realm_topic_pct['Realm'] = realm_topic_pct.index
# realm_top_topics['top_topics'] = top_topics
# sort dataframe by length of characters in realm
# realm_topic_pct = realm_topic_pct.sort_values(by='Realm', key=lambda x: x.str.len()).reset_index(drop=True)
# split realm into individual columns based on number of realms listed
# indiv_realm = pd.DataFrame(realm_topic_pct['Realms'].str.split("|",expand=True))
# indiv_realm.columns = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']
# realm_topic_pct = pd.concat([realm_topic_pct, indiv_realm], axis=1)

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
with open("../../Data/comb/indiv_realm_topic", "wb") as f:
    pickle.dump(indiv_realm_topic, f)
    f.close()

# open indiv_realm_topic
with open("../../Data/comb/indiv_realm?topic", "rb") as f:
    indiv_realm_topic = pickle.load(f)
    f.close()
    
# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(log_indiv_realm_topic.loc[log_indiv_realm_topic.idxmax(axis=1).sort_values().index],
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
fig.savefig("../../Data/comb/realms_HM.svg")
plt.close('all')

################################################################################
topic_dict = {0 : 'Monitor_species', 1 : 'Area_based_protection', 2 :  'Assess_major_threat', 3 : 'Monitor_trade', 4 : 'Monitor_aquatic_habitat', 5 : 'Monitor_population_size', 6 : 'Human_threat_to_habitat', 7 : 'European_regional_assessment', 8 : 'Wide_distribution', 9 : 'Forest_habitat_loss_to_agriculture', 10 : 'Monitor_population_structure', 11 : 'Restricted_range', 12 : 'Coral_reef_utility', 13 : 'Commercial_fishing', 14 : 'Species_range_parameters', 15 : 'Conservation_action', 16 : 'Population_decline_habitat_loss', 17 : 'Forest_habitat', 18 : 'Population_information', 19 : 'Monitor_island_endemics'}

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

# # initialize empty list for each realm
# list_afro = [] # afrotropical
# list_anta = [] # antarctic
# list_aust = [] # australasian
# list_indo = [] # indomalayan
# list_near = [] # Nearctic
# list_neot = [] # Neotropical
# list_ocea = [] # Oceanian
# list_pale = [] # Palearctic
# # loop through realm columns and generate list of topics for each realm
# for i in ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']:
#     afro_topics = realm_top_topics[realm_top_topics[i] == 'Afrotropical'].reset_index(drop=True)
#     for j in range(len(afro_topics)):
#         # topicsAfro |= set(afroTop.topTopics[j].split(', '))
#         list_afro.append(', '.join(afro_topics.top_topics[j].split(', ')))

#     anta_topics = realm_top_topics[realm_top_topics[i] == 'Antarctic'].reset_index(drop=True)
#     for j in range(len(anta_topics)):
#         list_anta.append(', '.join(anta_topics.top_topics[j].split(', ')))

#     aust_topics = realm_top_topics[realm_top_topics[i] == 'Australasian'].reset_index(drop=True)
#     for j in range(len(aust_topics)):
#         list_aust.append(', '.join(aust_topics.top_topics[j].split(', ')))

#     indo_topics = realm_top_topics[realm_top_topics[i] == 'Indomalayan'].reset_index(drop=True)
#     for j in range(len(indo_topics)):
#         list_indo.append(', '.join(indo_topics.top_topics[j].split(', ')))

#     near_topics = realm_top_topics[realm_top_topics[i] == 'Nearctic'].reset_index(drop=True)
#     for j in range(len(near_topics)):
#         list_near.append(', '.join(near_topics.top_topics[j].split(', ')))

#     neot_topics = realm_top_topics[realm_top_topics[i] == 'Neotropical'].reset_index(drop=True)
#     for j in range(len(neot_topics)):
#         list_neot.append(', '.join(neot_topics.top_topics[j].split(', ')))

#     ocea_topics = realm_top_topics[realm_top_topics[i] == 'Oceanian'].reset_index(drop=True)
#     for j in range(len(ocea_topics)):
#         list_ocea.append(', '.join(ocea_topics.top_topics[j].split(', ')))

#     pale_topics = realm_top_topics[realm_top_topics[i] == 'Palearctic'].reset_index(drop=True)
#     for j in range(len(pale_topics)):
#         list_pale.append(', '.join(pale_topics.top_topics[j].split(', ')))

# # load mask afrotropical
# afro_mask = np.array(Image.open("../../Data/Afrotropic-Ecozone-Biocountries-IM500.png"))
# # join list into string for generating wordcloud
# string_afro = (", ").join(list_afro)
# f = plt.figure()
# plt.imshow(WordCloud(scale=3, mask=afro_mask, background_color='white', max_font_size=100, relative_scaling=1, contour_color='black', contour_width=0.5).generate(string_afro))
# plt.axis('off')
# plt.savefig("../../Data/comb/afro_topic_WC.svg")
# plt.show()
# plt.close('all')

# # load mask antarctic

# # wordcloud per biogeographic realm
# string_afro = (", ").join(list_afro)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_afro))
# plt.axis('off')
# plt.title('Top topics for Afrotropical realm')
# f.savefig("../../Data/comb/afro_topic_WC.svg")
# plt.close('all')

# string_anta = (", ").join(list_anta)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_anta))
# plt.axis('off')
# plt.title('Top topics for Antarctic realm')
# f.savefig("../../Data/comb/anta_topic_WC.svg")
# plt.close('all')

# string_aust = (", ").join(list_aust)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_aust))
# plt.axis('off')
# plt.title('Top topics for Australasian realm')
# f.savefig("../../Data/comb/aust_topic_WC.svg")
# plt.close('all')

# string_indo = (", ").join(list_indo)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_indo))
# plt.axis('off')
# plt.title('Top topics for Indomalayan realm')
# f.savefig("../../Data/comb/indo_topic_WC.svg")
# plt.close('all')

# string_near = (", ").join(list_near)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_near))
# plt.axis('off')
# plt.title('Top topics for Nearctic realm')
# f.savefig("../../Data/comb/near_topic_WC.svg")
# plt.close('all')

# string_neot = (", ").join(list_neot)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_neot))
# plt.axis('off')
# plt.title('Top topics for Neotropical realm')
# f.savefig("../../Data/comb/neot_topic_WC.svg")
# plt.close('all')

# string_ocea = (", ").join(list_ocea)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750).generate(string_ocea))
# plt.axis('off')
# plt.title('Top topics for Oceanic realm')
# f.savefig("../../Data/comb/ocea_topic_WC.svg")
# plt.close('all')

# string_pale = (", ").join(list_pale)
# f = plt.figure()
# plt.imshow(WordCloud(width=1500, height=750, scale=3).generate(string_pale))
# plt.axis('off')
# plt.title('Top topics for Palearctic realm')
# f.savefig("../../Data/comb/pale_topic_WC2.svg")
# plt.close('all')



# # check which species are labeled Marine|Marine for systems
# # df[df['systems'] == 'Marine|Marine']['scientificName']
# # remove rows with na
# docTopicSys = docTopicSys.dropna()
# # group docTopicSys by systems and sum topic composition for each system
# docTopicSysSum = docTopicSys.groupby(['systems']).sum()
# # calculate percentage composition of topics per system
# docTopicSysPct = docTopicSysSum/(docTopicSys.groupby(['systems']).count())
# # rename columns to named topics
# docTopicSysPct.columns = ['Monitor_situation', 'Fishery_management', 'Forest_habitat_management', 'Monitor_species_ecology', 'Assess_present_threat', 'Trade_risk', 'Habitat_based_conservation', 'Manage_trade_threat', 'Subpopulation_assessment', 'Act_on_current_information', 'Site_based_conservation', 'Monitor_accession', 'Breeding_habitat', 'Protect_range', 'Research_population_threat']

# # save docTopicSysPct
# with open("../Data/conAct/docTopicSysPct.pkl", "wb") as f:
#     pickle.dump(docTopicSysPct, f)
#     f.close()

# # load docTopicSysPct
# with open("../Data/conAct/docTopicSysPct.pkl", "rb") as f:
#     docTopicSysPct = pickle.load(f)
#     f.close()

# plot heatmap
# fig = plt.figure(figsize=(16,6)) # figure dimensions
# sns.set_context('paper', font_scale=1.5) # context of plot and font scale
# g = sns.heatmap(docTopicSysPct.loc[docTopicSysPct.idxmax(axis=1).sort_values().index],
#     linewidths=0.5, # linewidth between cells
#     cmap='YlGnBu', # color
#     cbar_kws={'shrink':0.8} # size of color bar
# )
# g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
# g.set(xlabel="Topics", ylabel="Systems") # label both axes
# # g.set_yticklabels(g.get_yticklabels(), rotation=90)
# plt.tight_layout() # tight layout for labels to show on screen
# # docTopicSysPct.sum(axis=1) # check total percentage
# fig.show()
# fig.savefig("../Data/conAct/docTopicSysHM.png", dpi=500)
# plt.close('all')





# # find dominant topic for each doc
# domTopicDf = pd.DataFrame()
# for i, rowList in enumerate(topics):
#     row = rowList[0] if ldaModel.per_word_topics else rowList
#     row = sorted(row, key=lambda x: (x[1]), reverse = True)
#     # get dominant topic, weightage and keywords for each doc
#     for j, (topicNum, topicWght) in enumerate(row):
#         if j == 0: # dominant topic
#             domTopicDf = domTopicDf.append(pd.Series([int(topicNum), round(topicWght,4)]), ignore_index=True)
#         else:
#             break

# domTopicDf.columns = ['DominantTopic', 'TopicWeight']
# domTopicDf = pd.concat([domTopicDf, df['realm']], axis=1)
# domTopicDf = domTopicDf.dropna()


# for i in domTopicDf['realm'].unique():
#     i = 'Neotropical'
#     test = domTopicDf[domTopicDf['realm'] == i]
#     test = test.sort_values('TopicWeight', ascending=False)
#     for j in range(len(test)):
#         if test.iloc[j,1] > 0.98:
#             print(test.iloc[j,0])
#         else:
#             break


# # add realms column to docTopic
# docTopicRlm = pd.concat([docTopic, df['realm']], axis=1)
# # remove rows with na
# docTopicRlm = docTopicRlm.dropna()
# # group docTopicSys by realm and sum topic composition for each realm
# docTopicRlmSum = docTopicRlm.groupby(['realm']).sum()
# # calculate percentage composition of topics per realm
# docTopicRlmPct = docTopicRlmSum/(docTopicRlm.groupby(['realm']).count())
# # rename columns to named topics
# docTopicRlmPct.columns = ['Monitor_situation', 'Fishery_management', 'Forest_habitat_management', 'Monitor_species_ecology', 'Assess_present_threat', 'Trade_risk', 'Habitat_based_conservation', 'Manage_trade_threat', 'Subpopulation_assessment', 'Act_on_current_information', 'Site_based_conservation', 'Monitor_accession', 'Breeding_habitat', 'Protect_range', 'Research_population_threat']

# # save docTopicRlmPct
# with open("../Data/conAct/docTopicRlmPct.pkl", "wb") as f:
#     pickle.dump(docTopicRlmPct, f)
#     f.close()

# # load docTopicRlmPct
# with open("../Data/conAct/docTopicRlmPct.pkl", "rb") as f:
#     docTopicRlmPct = pickle.load(f)
#     f.close()

# topTopics = []
# for i in range(len(docTopicRlmPct)):
#     topTopics.append(', '.join(docTopicRlmPct.iloc[i].sort_values(ascending=False)[:5].index.values)) # top 5
#     # topTopics.append(', '.join(docTopicRlmPct.iloc[i].sort_values(ascending=False).index.values)) # all topics

# # create dataframe for each realm(s) and their top topics
# topTopicsRlm = pd.DataFrame(docTopicRlmPct.index)
# topTopicsRlm['topTopics'] = topTopics
# # sort dataframe by length of characters in realm
# topTopicsRlm = topTopicsRlm.sort_values(by='realm', key=lambda x: x.str.len()).reset_index(drop=True)
# # split realm into individual columns based on number of realms listed
# splitRlm = pd.DataFrame(topTopicsRlm['realm'].str.split("|",expand=True))
# splitRlm.columns = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']
# topTopicsRlm = pd.concat([topTopicsRlm, splitRlm], axis=1)

# # initialize empty list
# listAfro = []
# # loop through realm columns and generate list of topics for each realm
# for i in ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']:
#     afroTop = topTopicsRlm[topTopicsRlm[i] == 'Afrotropical'].reset_index(drop=True)
#     for j in range(len(afroTop)):
#         # topicsAfro |= set(afroTop.topTopics[j].split(', '))
#         listAfro.append(', '.join(afroTop.topTopics[j].split(', ')))
#     antaTop = topTopicsRlm[topTopicsRlm[i] == 'Antarctic'].reset_index(drop=True)
#     for j in range(len(antaTop)):
#         topicsAnta |= set(antaTop.topTopics[j].split(', '))

# # load mask
# afroMask = np.array(Image.open("../Data/Afrotropic-Ecozone-Biocountries-IM500.png"))
# # join list into string for generating wordcloud
# stringAfro = (", ").join(listAfro)
# f = plt.figure()
# plt.imshow(WordCloud(scale=3, mask=afroMask, background_color='white', max_font_size=100, relative_scaling=1, contour_color='black', contour_width=0.5).generate(stringAfro))
# plt.axis('off')
# plt.savefig("../Data/conAct/afroTopicWC.png", dpi=500)
# plt.show()
# plt.close('all')


# # plot heatmap
# fontsizePt = plt.rcParams['ytick.labelsize']
# dpi = 72.27
# matHgtPt = fontsizePt * docTopicRlmPct.shape[0]
# matHgtIn = matHgtPt/dpi
# topMgn = 0.04
# botMgn = 0.04
# figHgt = matHgtIn / (1 - topMgn - botMgn)
# fig, ax = plt.subplots(
#     figsize=(6,figHgt),
#     gridspec_kw=dict(top=1-topMgn,bottom=botMgn))
# ax = sns.heatmap(docTopicRlmPct.loc[docTopicRlmPct.idxmax(axis=1).sort_values().index],
#     linewidths=0.5,
#     cmap='RdYlGn',
#     cbar_kws={'shrink':0.5}, ax=ax)

# plt.show()

# fig = plt.figure(figsize=(12,36)) # figure dimensions
# sns.set_context('paper', font_scale=0.8) # context of plot and font scale
# g = sns.heatmap(docTopicRlmPct.loc[docTopicRlmPct.idxmax(axis=1).sort_values().index],
#     linewidths=0.5, # linewidth between cells
#     cmap='RdYlGn', # color
#     cbar_kws={'shrink':0.5}) # set size of bar
# g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
# g.set(xlabel="Topics", ylabel="Realm") # label both axes
# # g.set_yticklabels(g.get_yticklabels(), rotation=90)
# plt.tight_layout() # tight layout for labels to show on screen
# # docTopicSysPct.sum(axis=1) # check total percentage
# fig.show()
# fig.savefig("../Data/conAct/docTopicRlmHM.png", dpi=500)
# plt.close('all')