#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
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

# import taxonomy information
taxon_info = pd.read_csv("../../Data/taxonomy_info.csv")
# select only taxonid and class_name
taxon_info = taxon_info[['taxonid', 'class_name']]
taxon_info = taxon_info.rename(columns={"taxonid": "internalTaxonId"})

# merge df with taxon_info
df = df.merge(taxon_info[['internalTaxonId', 'class_name']])

# load trained model
with open("../../Data/comb_global/model_17.pkl", "rb") as f:
    model = pickle.load(f)
    f.close()

# load dictionary
dic = corpora.Dictionary.load("../../Data/comb_global/dic.dict")

# load bigram model
bigram_mod = Phrases.load("../../Data/comb_global/bigram_mod.pkl")

# # load corpus
# corpus = corpora.MmCorpus("../../Data/comb_global/bow_corpus.mm")

#################################################################
# alternative corpus solution
rat = df.rationale.values.tolist()
hab = df.habitat.values.tolist()
thr = df.threats.values.tolist()
pop = df.population.values.tolist()
ran = df.range.values.tolist()
use = df.useTrade.values.tolist()
con = df.conservationActions.values.tolist()

nlp = spacy.load('en_core_web_sm') # load small sized english vocab

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"situ", "appendix", "cite", "annex", "need", "book", "find"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

def bow_transform(corpus):
    """
    Function transforms corpus from text format into bag of words format
    """
    # generator of corpus
    gen = nlp.pipe(corpus, n_process = 7, batch_size = 800, disable = ["parser", "ner"])
    # tokenise corpus
    tokens = []
    for doc in gen:
        tokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP' and tok.tag_ != 'NNPS' and tok.tag_ != 'VBG' and tok.tag_ != '_SP'])
    # apply bigram model
    tokens_2 = bigramMod[tokens]
    # convert to bag of words corpus
    bow_corpus = [dic.doc2bow(doc) for doc in tokens_2]
    return bow_corpus

rat_subcorpus = bow_transform(rat)
hab_subcorpus = bow_transform(hab)
thr_subcorpus = bow_transform(thr)
pop_subcorpus = bow_transform(pop)
ran_subcorpus = bow_transform(ran)
use_subcorpus = bow_transform(use)
con_subcorpus = bow_transform(con)

################################################################

def topics_to_df(topics, num_topics):
    """
    Function converts the topic composition of documents to a dataframe
    """
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

def get_doc_topic_comp(corpus):
    """
    Function applies model to produce list of topic composition for each document
    """
    topic_comp = [model[corpus[i]] for i in range(len(corpus))]
    # generate dataframe of document topic composition
    doc_topic_comp = pd.concat([topics_to_df(topics, 17) for topics in topic_comp]).reset_index(drop = True).fillna(0)
    return doc_topic_comp

rat_topic_comp = get_doc_topic_comp(rat_subcorpus)
rat_topic_comp = rat_topic_comp.add_suffix('_rat')

with open("../../Data/comb_global/rat_topic_comp.pkl", "wb") as f:
    pickle.dump(rat_topic_comp, f)
    f.close()

hab_topic_comp = get_doc_topic_comp(hab_subcorpus)
hab_topic_comp = hab_topic_comp.add_suffix('_hab')

with open("../../Data/comb_global/hab_topic_comp.pkl", "wb") as f:
    pickle.dump(hab_topic_comp, f)
    f.close()

thr_topic_comp = get_doc_topic_comp(thr_subcorpus)
thr_topic_comp = thr_topic_comp.add_suffix('_thr')

with open("../../Data/comb_global/thr_topic_comp.pkl", "wb") as f:
    pickle.dump(thr_topic_comp, f)
    f.close()

pop_topic_comp = get_doc_topic_comp(pop_subcorpus)
pop_topic_comp = pop_topic_comp.add_suffix('_pop')

with open("../../Data/comb_global/pop_topic_comp.pkl", "wb") as f:
    pickle.dump(pop_topic_comp, f)
    f.close()

ran_topic_comp = get_doc_topic_comp(ran_subcorpus)
ran_topic_comp = ran_topic_comp.add_suffix('_ran')

with open("../../Data/comb_global/ran_topic_comp.pkl", "wb") as f:
    pickle.dump(ran_topic_comp, f)
    f.close()

use_topic_comp = get_doc_topic_comp(use_subcorpus)
use_topic_comp = use_topic_comp.add_suffix('_use')

with open("../../Data/comb_global/use_topic_comp.pkl", "wb") as f:
    pickle.dump(use_topic_comp, f)
    f.close()

con_topic_comp = get_doc_topic_comp(con_subcorpus)
con_topic_comp = con_topic_comp.add_suffix('_con')

with open("../../Data/comb_global/con_topic_comp.pkl", "wb") as f:
    pickle.dump(con_topic_comp, f)
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

# red_list_cat = []
# realm = []
# systems = []
# for i in ['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions']:
#     red_list_cat.extend(df[df[i]!='']['redlistCategory'].values.tolist())
#     realm.extend(df[df[i]!='']['realm'].values.tolist())
#     systems.extend(df[df[i]!='']['systems'].values.tolist())

# initialise dataframe for topic composition with corresponding red_list_cat, realm, systems and class_name

df_rl_r_s_c = pd.concat([df['redlistCategory'], df['realm'], df['systems'], df['class_name']], axis = 1).reset_index(drop=True)
df_rl_r_s_c = df_rl_r_s_c.rename(columns={"redlistCategory":"Red_list_category", "realm":"Realm", "systems":"Systems", "class_name":"Class"})
# topic composition of each of the 7 assessment sections for each species
text_topic_comp = pd.concat([df_rl_r_s_c, rat_topic_comp, hab_topic_comp, thr_topic_comp, pop_topic_comp, ran_topic_comp, use_topic_comp, con_topic_comp], axis = 1).reset_index(drop = True)

with open("../../Data/comb_global/text_topic_comp.pkl", "wb") as f:
    pickle.dump(text_topic_comp, f)
    f.close()

with open("../../Data/comb_global/text_topic_comp.pkl", "rb") as f:
    text_topic_comp = pickle.load(f)
    f.close()

with open("../../Data/comb_global/rat_topic_comp.pkl", "rb") as f:
    rat_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/hab_topic_comp.pkl", "rb") as f:
    hab_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/thr_topic_comp.pkl", "rb") as f:
    thr_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/pop_topic_comp.pkl", "rb") as f:
    pop_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/ran_topic_comp.pkl", "rb") as f:
    ran_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/use_topic_comp.pkl", "rb") as f:
    use_topic_comp = pickle.load(f)
    f.close()
with open("../../Data/comb_global/con_topic_comp.pkl", "rb") as f:
    con_topic_comp = pickle.load(f)
    f.close()

# aggregate topic composition
rat_topic_comp.columns = rat_topic_comp.columns.str.rstrip("_rat")
hab_topic_comp.columns = hab_topic_comp.columns.str.rstrip("_hab")
thr_topic_comp.columns = thr_topic_comp.columns.str.rstrip("_thr")
pop_topic_comp.columns = pop_topic_comp.columns.str.rstrip("_pop")
ran_topic_comp.columns = ran_topic_comp.columns.str.rstrip("_ran")
use_topic_comp.columns = use_topic_comp.columns.str.rstrip("_use")
con_topic_comp.columns = con_topic_comp.columns.str.rstrip("_con")

comb_topic_comp = rat_topic_comp + hab_topic_comp + thr_topic_comp + pop_topic_comp + ran_topic_comp + use_topic_comp + con_topic_comp

# combined topic composition of 7 text summaries for each species
def tally_prob(row):
    row = row/round(row.sum())
    return row

# normalize total topic composition for each species
comb_topic_comp_norm = comb_topic_comp.apply(lambda x: tally_prob(x), axis = 1)

# rename columns to topic names
comb_topic_comp_norm.columns = ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']

with open("../../Data/comb_global/comb_topic_comp_norm.pkl", "wb") as f:
    pickle.dump(comb_topic_comp_norm, f)
    f.close()

df_topic_comp = pd.concat([df_rl_r_s_c, comb_topic_comp_norm], axis = 1)

with open("../../Data/comb_global/df_topic_comp.pkl", "wb") as f:
    pickle.dump(df_topic_comp, f)
    f.close()

################################################################################
# Plot probability distribution of topics
################################################################################
top_prob_freq = pd.melt(comb_topic_comp_norm.reset_index(), id_vars='index', var_name='Topic', value_name="Probability")
# start here
# melted the dataframe, plot using df['Topic'] == [0-16]

sns.set(rc={"figure.figsize":(16,6)})
sns.set_style('whitegrid')
g = sns.kdeplot(data=top_prob_freq, x='Probability', hue='Topic')
g = sns.displot(data=top_prob_freq, x='Probability', hue='Topic', kind='kde')
g.set(xticks=np.arange(0,0.75,step=0.05))
g.set(xlabel = 'Topic Probability', ylabel = 'Probability Density')
plt.show()
plt.savefig("../../Data/comb_global/comb_kde.svg")
plt.close('all')

#################################################################################
# change low risk conservation dependent and low risk near threatened to near threatened, low risk least concern to least concern
# take out regional assessments
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/conservation dependent', 'Near Threatened')
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/near threatened', 'Near Threatened')
df_topic_comp['Red_list_category'] = df_topic_comp['Red_list_category'].replace('Lower Risk/least concern', 'Least Concern')
df_no_reg_ext = df_topic_comp[(df_topic_comp.Red_list_category != 'Regionally Extinct')]

## filtering then grouping
# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    df_no_reg_ext[i] = df_no_reg_ext[i].mask(df_no_reg_ext[i] < 0.05, 0)

# group by red list category and sum topic composition for each category
red_list_topic = df_no_reg_ext.drop(columns=['Realm', 'Systems', 'Class']).groupby(['Red_list_category']).sum()
# normalize
red_list_topic_pct = red_list_topic/(df_no_reg_ext.drop(columns=['Realm', 'Systems', 'Class']).groupby(['Red_list_category']).count())
# reorder and sort index
index_order = ['Least Concern', 'Near Threatened', 'Vulnerable', 'Endangered', 'Critically Endangered', 'Extinct in the Wild', 'Extinct']
red_list_topic_pct = red_list_topic_pct.reindex(index_order)

# ## grouping then filtering
# for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
#     red_list_topic_pct[i] = red_list_topic_pct[i].mask(red_list_topic_pct[i] < 0.05, 0)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(red_list_topic_pct.loc[red_list_topic_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8,}) # size of color bar
            #   'ticks':[0.000, 0.025, 0.050, 0.075, 0.100, 0.125]}) # set ticklabels
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Red list category") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
# save filtered then grouped
fig.savefig("../../Data/comb_global/red_list_cat_HM_ftg.svg")
plt.close('all')

################################################################################
# replace Marine|Marine with Marine
df_topic_comp['Systems'] = df_topic_comp['Systems'].replace('Marine|Marine', 'Marine')
systems_topic = df_topic_comp

# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    systems_topic[i] = systems_topic[i].mask(systems_topic[i] < 0.05, 0)
    
# group by systems and sum topic composition for each category
systems_topic = systems_topic.drop(columns=['Red_list_category', 'Realm', 'Class']).groupby(['Systems']).sum()
# normalize
systems_topic_pct = systems_topic/(df_topic_comp.drop(columns=['Red_list_category', 'Realm', 'Class']).groupby(['Systems']).count())
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
# save filtered then grouped
fig.savefig("../../Data/comb_global/systems_HM_ftg.svg")
plt.close('all')

################################################################################
# Grouping by number of realms spp exist across, from 1 to 8
# # remove rows with na
# realm_topic = df_topic_comp.dropna().drop(columns=['Red_list_category', 'Systems', 'Class'])

# # group realm_topics by number of realms
# # count number of realms species exist across
# def count_realms(realms):
#     return len(realms.split('|'))

# realm_topic['num_realms'] = realm_topic.Realm.apply(count_realms)
# realm_topic = realm_topic.drop(columns=['Realm']).groupby(['num_realms'])
# # calculate percentage composition of topics per realm
# realm_topic_pct = realm_topic.sum()/realm_topic.count()

# # save realm_topic_pct.pkl
# with open("../../Data/comb_global/realm_topic_pct.pkl", "wb") as f:
#     pickle.dump(realm_topic_pct, f)
#     f.close()

# # load realm_topic_pct.pkl
# with open("../../Data/comb_global/realm_topic_pct.pkl", "rb") as f:
#     realm_topic_pct = pickle.load(f)
#     f.close()

# # plot heatmap
# fig = plt.figure(figsize=(16,6)) # figure dimensions
# sns.set_context('paper', font_scale=1.5) # context of plot and font scale
# g = sns.heatmap(realm_topic_pct.loc[realm_topic_pct.idxmax(axis=1).index],
#     linewidths=0.5, # linewidth between cells
#     cmap='YlGnBu', # color
#     cbar_kws={'shrink':0.8} # size of color bar
# )
# g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
# g.set(xlabel="Topics", ylabel="Number of realm(s)") # label both axes
# # g.set_yticklabels(g.get_yticklabels(), rotation=90)
# plt.tight_layout() # tight layout for labels to show on screen
# # docTopicSysPct.sum(axis=1) # check total percentage
# fig.show()
# fig.savefig("../../Data/comb_global/realms_HM_new.svg")
# plt.close('all')

################################################################################
# grouping by the realms for all species (not feasible)
# remove rows with na
realm_topic = df_topic_comp.dropna().drop(columns=['Red_list_category', 'Systems', 'Class'])
# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    realm_topic[i] = realm_topic[i].mask(realm_topic[i] < 0.05, 0)

# group by realms
realm_topic_pct = realm_topic.groupby(['Realm']).sum()/realm_topic.groupby(['Realm']).count()
realm_topic_pct['Realm'] = realm_topic_pct.index

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

# save indiv_realm_topic
with open("../../Data/comb_global/indiv_realm_topic", "wb") as f:
    pickle.dump(indiv_realm_topic, f)
    f.close()

# # open indiv_realm_topic
# with open("../../Data/comb_global/indiv_realm?topic", "rb") as f:
#     indiv_realm_topic = pickle.load(f)
#     f.close()

# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    indiv_realm_topic[i] = indiv_realm_topic[i].mask(indiv_realm_topic[i] < 0.05, 0)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(indiv_realm_topic.loc[indiv_realm_topic.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Realms") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/realm_topics_HM_ftg.svg")
plt.close('all')

################################################################################
# compare between species that exist only in 1 realm
single_realm = ['Afrotropical', 'Antarctic', 'Australasian', 'Indomalayan', 'Nearctic', 'Neotropical', 'Oceanian', 'Palearctic']
single_realm_pct = realm_topic_pct.loc[realm_topic_pct['Realm'].isin(single_realm)]
single_realm_pct = single_realm_pct.drop(columns = 'Realm')

# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    single_realm_pct[i] = single_realm_pct[i].mask(single_realm_pct[i] < 0.05, 0)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(single_realm_pct.loc[single_realm_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Realms") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/single_realm_topics_HM_ftg.svg")
plt.close('all')

################################################################################
# group by class and sum topic composition for each category
class_topic = df_topic_comp.drop(columns=['Red_list_category', 'Realm', 'Systems']).groupby(['Class']).sum()

# filter then group
# if topic percentage less than 0.05, change to 0
for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
    class_topic[i] = class_topic[i].mask(class_topic[i] < 0.05, 0)

class_topic_pct = class_topic/(df_topic_comp.drop(columns=['Red_list_category', 'Realm', 'Systems']).groupby(['Class']).count())

# filter out classes with more than 325 species
index = df_topic_comp.drop(columns=['Red_list_category', 'Realm', 'Systems'])['Class'].value_counts()[lambda x: x>325].index.tolist()
top_quart_class_topic_pct = class_topic_pct[class_topic_pct.index.isin(index)]

# group then filter
# # if topic percentage less than 0.05, change to 0
# for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
#     top_quart_class_topic_pct[i] = top_quart_class_topic_pct[i].mask(top_quart_class_topic_pct[i] < 0.05, 0)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(top_quart_class_topic_pct.loc[top_quart_class_topic_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Class") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/top_quart_class_topics_HM_ftg.svg")
plt.close('all')

# filter out top 6 chordata classes
index = ['ACTINOPTERYGII', 'AVES', 'REPTILIA', 'AMPHIBIA', 'MAMMALIA', 'CHONDRICHTHYES']
six_class_topic_pct = class_topic_pct[class_topic_pct.index.isin(index)]

# group then filter
# # if topic percentage less than 0.05, change to 0
# for i in ['Monitor_island_endemics', 'Water_pollution_threat', 'Aquatic_ecology', 'Forest_loss', 'Monitoring_and_rewilding', 'Range', 'Common_threats', 'Forest_ecosystem', 'Population_structure', 'Fisheries_threats', 'Threat_distribution', 'Forest_fragmentation', 'Habitat_loss', 'Area_based_protection', 'Assessment_criteria', 'Population_dynamics', 'Conservation_actions']:
#     six_class_topic_pct[i] = six_class_topic_pct[i].mask(six_class_topic_pct[i] < 0.05, 0)

# plot heatmap
fig = plt.figure(figsize=(16,6)) # figure dimensions
sns.set_context('paper', font_scale=1.5) # context of plot and font scale
g = sns.heatmap(six_class_topic_pct.loc[six_class_topic_pct.idxmax(axis=1).index],
    linewidths=0.5, # linewidth between cells
    cmap='YlGnBu', # color
    cbar_kws={'shrink':0.8} # size of color bar
)
g.set_xticklabels(g.get_xticklabels(), rotation=90) # rotate xticklabels
g.set(xlabel="Topics", ylabel="Class") # label both axes
# g.set_yticklabels(g.get_yticklabels(), rotation=90)
plt.tight_layout() # tight layout for labels to show on screen
# docTopicSysPct.sum(axis=1) # check total percentage
fig.show()
fig.savefig("../../Data/comb_global/six_class_topics_HM_ftg.svg")
plt.close('all')

################################################################################
topic_dict = {0: 'Monitor_island_endemics', 1: 'Water_pollution_threat', 2: 'Aquatic_ecology', 3: 'Forest_loss', 4: 'Monitoring_and_rewilding', 5: 'Range', 6: 'Common_threats', 7: 'Forest_ecosystem', 8: 'Population_structure', 9: 'Fisheries_threats', 10: 'Threat_distribution', 11: 'Forest_fragmentation', 12: 'Habitat_loss', 13: 'Area_based_protection', 14: 'Assessment_criteria', 15: 'Population_dynamics', 16: 'Conservation_actions'}

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