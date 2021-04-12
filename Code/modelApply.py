#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# import cleaned df
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# load trained model
with open("../Data/conAct/conActModel15.pkl", "rb") as f:
    ldaModel = pickle.load(f)
    f.close()

# load corpus
corpus = corpora.MmCorpus("../Data/conAct/conActBoWCorpus.mm")

# apply model to produce list of topic composition for each document
topics = [ldaModel[corpus[i]] for i in range(len(corpus))]

def topicsDocToDf(topicsDoc, numTopics):
    """
    Function converts the topic composition of documents to a dataframe
    """
    res = pd.DataFrame(columns=range(numTopics))
    for topicWeight in topicsDoc:
        res.loc[0, topicWeight[0]] = topicWeight[1]
    return res

# generate dataframe of document topic composition
docTopic = pd.concat([topicsDocToDf(topicsDoc, numTopics=15) for topicsDoc in topics]).reset_index(drop=True).fillna(0)
# check which species are labeled Marine|Marine for systems
# df[df['systems'] == 'Marine|Marine']['scientificName']
# add systems column to docTopic
docTopicSys = pd.concat([docTopic, df['systems']], axis=1)
# replace Marine|Marine with Marine
docTopicSys['systems'] = docTopicSys['systems'].replace('Marine|Marine', 'Marine')
# remove rows with na
docTopicSys = docTopicSys.dropna()
# group docTopicSys by systems and sum topic composition for each system
docTopicSysSum = docTopicSys.groupby(['systems']).sum()
# calculate percentage composition of topics per system
docTopicSysPct = docTopicSysSum/(docTopicSys.groupby(['systems']).count())
# rename columns to named topics
docTopicSysPct.columns = ['Monitor_situation', 'Fishery_management', 'Forest_habitat_management', 'Monitor_species_ecology', 'Assess_present_threat', 'Trade_risk', 'Habitat_based_conservation', 'Manage_trade_threat', 'Subpopulation_assessment', 'Act_on_current_information', 'Site_based_conservation', 'Monitor_accession', 'Breeding_habitat', 'Protect_range', 'Research_population_threat']

# save docTopicSysPct
with open("../Data/conAct/docTopicSysPct.pkl", "wb") as f:
    pickle.dump(docTopicSysPct, f)
    f.close()

# load docTopicSysPct
with open("../Data/conAct/docTopicSysPct.pkl", "rb") as f:
    docTopicSysPct = pickle.load(f)
    f.close()

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(docTopicSysPct.loc[docTopicSysPct.idxmax(axis=1).sort_values().index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Systems") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../Data/conAct/docTopicSysHM.png", dpi=500)
plt.close('all')





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


# add realms column to docTopic
docTopicRlm = pd.concat([docTopic, df['realm']], axis=1)
# remove rows with na
docTopicRlm = docTopicRlm.dropna()
# group docTopicSys by realm and sum topic composition for each realm
docTopicRlmSum = docTopicRlm.groupby(['realm']).sum()
# calculate percentage composition of topics per realm
docTopicRlmPct = docTopicRlmSum/(docTopicRlm.groupby(['realm']).count())
# rename columns to named topics
docTopicRlmPct.columns = ['Monitor_situation', 'Fishery_management', 'Forest_habitat_management', 'Monitor_species_ecology', 'Assess_present_threat', 'Trade_risk', 'Habitat_based_conservation', 'Manage_trade_threat', 'Subpopulation_assessment', 'Act_on_current_information', 'Site_based_conservation', 'Monitor_accession', 'Breeding_habitat', 'Protect_range', 'Research_population_threat']

# save docTopicRlmPct
with open("../Data/conAct/docTopicRlmPct.pkl", "wb") as f:
    pickle.dump(docTopicRlmPct, f)
    f.close()

# load docTopicRlmPct
with open("../Data/conAct/docTopicRlmPct.pkl", "rb") as f:
    docTopicRlmPct = pickle.load(f)
    f.close()

topTopics = []
for i in range(len(docTopicRlmPct)):
    topTopics.append(', '.join(docTopicRlmPct.iloc[i].sort_values(ascending=False)[:5].index.values)) # top 5
    # topTopics.append(', '.join(docTopicRlmPct.iloc[i].sort_values(ascending=False).index.values)) # all topics

# create dataframe for each realm(s) and their top topics
topTopicsRlm = pd.DataFrame(docTopicRlmPct.index)
topTopicsRlm['topTopics'] = topTopics
# sort dataframe by length of characters in realm
topTopicsRlm = topTopicsRlm.sort_values(by='realm', key=lambda x: x.str.len()).reset_index(drop=True)
# split realm into individual columns based on number of realms listed
splitRlm = pd.DataFrame(topTopicsRlm['realm'].str.split("|",expand=True))
splitRlm.columns = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']
topTopicsRlm = pd.concat([topTopicsRlm, splitRlm], axis=1)

# initialize empty list
listAfro = []
# loop through realm columns and generate list of topics for each realm
for i in ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven']:
    afroTop = topTopicsRlm[topTopicsRlm[i] == 'Afrotropical'].reset_index(drop=True)
    for j in range(len(afroTop)):
        # topicsAfro |= set(afroTop.topTopics[j].split(', '))
        listAfro.append(', '.join(afroTop.topTopics[j].split(', ')))
    antaTop = topTopicsRlm[topTopicsRlm[i] == 'Antarctic'].reset_index(drop=True)
    for j in range(len(antaTop)):
        topicsAnta |= set(antaTop.topTopics[j].split(', '))

# load mask
afroMask = np.array(Image.open("../Data/Afrotropic-Ecozone-Biocountries-IM500.png"))
# join list into string for generating wordcloud
stringAfro = (", ").join(listAfro)
f = plt.figure()
plt.imshow(WordCloud(scale=3, mask=afroMask, background_color='white', max_font_size=100, relative_scaling=1, contour_color='black', contour_width=0.5).generate(stringAfro))
plt.axis('off')
plt.savefig("../Data/conAct/afroTopicWC.png", dpi=500)
plt.show()
plt.close('all')


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