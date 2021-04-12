#!/usr/bin/env python3

import os

# load files
import pickle

# tokenization
import spacy

# create dict and vectorized corpus
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

# get home directory
home = os.path.expanduser('~')

# pickle load dfClean
with open(home+"/Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# create conservation actions corpus
conActCorpus = df.conservationActions.values.tolist()
del df
conActCorpus = list(filter(None, conActCorpus))

# save conActCorpus
with open(home+"/Data/conAct/conActCorpus.pkl", "wb") as f:
    pickle.dump(conActCorpus, f)
    f.close()

nlp = spacy.load('en_core_web_sm')

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"situ", "appendix", "cite", "annex", "need", "book", "find"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

conActCorpusGen = nlp.pipe(conActCorpus, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

conActTokens = []
for doc in conActCorpusGen:
    conActTokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP' and tok.tag_ != 'NNPS' and tok.tag_ != 'VBG'])

# build bigram model
bigram = Phrases(conActTokens, min_count = len(conActTokens)//600, threshold = 140)
# trigram = Phrases(bigram[conActTokens], min_count = len(conActTokens)//500, threshold = 50000)
bigramMod = Phraser(bigram)
# trigramMod = Phraser(trigram)
# print(bigramMod[conActTokens[2]])
# print(trigramMod[conActTokens[3200]])
# # list of bigrams obtained, with their corresponding scores
# bigramsList = []
# for doc in bigram.export_phrases(conActTokens):
#     print(doc)

# bigramsList[0:100]

conActTokens2 = bigramMod[conActTokens]

# pickle conActTokens2
with open(home+"/Data/conAct/conActTokens2.pkl", "wb") as f:
    pickle.dump(conActTokens2, f)
    f.close()

# create dictionary
conActDict = corpora.Dictionary(conActTokens2)

conActDict.filter_extremes(no_below=100, no_above=0.45)

conActDict.compactify()

conActDict.save(home+"/Data/conAct/conActDict.dict")

conActBoWCorpus = [conActDict.doc2bow(doc) for doc in conActTokens2]

corpora.MmCorpus.serialize(home+"/Data/conAct/conActBoWCorpus.mm", conActBoWCorpus)