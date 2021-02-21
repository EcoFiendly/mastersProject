#!/usr/bin/env python3

import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as sw

import numpy as np
import gensim
import gensim.corpora as corpora


confirm = {"species", "specie", "need", "know", "measure", "situ", "list", "ex", "number", "include", "place", "require", "change", "extent", "find", "use", "available", "total", "new", "consider", "concern", "datum", "likely", "utilize", "action", "include", "site", "use", "place", "consider", "recommend", "estimate", "information", "trend", "report", "survey", "assess", "assessment", "occassionally", "level", "year", "currently", "occurrence", "km²", "de", "appendix", "north", "northern", "south", "southern", "east", "eastern", "west", "western", "et", "al", "asl", "occur", "criterion", "threshold", "occupancy", "threat", "qualify", "believe", "reason", "major", "significant", "presume", "unlikely", "evaluate", "suspect", "aoo", "eoo", "specific", "cite", "ii", "convention", "awareness", "known", "mexico", "austalia", "eu", "europe", "european", "africa", "madagascar", "atlantic", "brazil", "america", "rio", "ecuador", "columbia", "australia", "papua", "guinea", "china", "spain", "queensland", "india", "thailand", "unknown", "mediterranean", "republic", "central", "province", "locality", "record", "national", "indonesia", "viet", "area", "habitat", "italy", "france", "pacific", "malaysia", "asia", "la", "parque", "population", "collection", "collect", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "annex", "colombia", "sierra", "argentina", "cameroon", "mozambique", "locally", "suggest", "study", "parque", "iran", "netherland", "persian", "germany", "nam", "myanmar", "costa", "panama", "rica", "tanzania", "venezuela", "ireland", "bolivia", "widely", "continue", "recently", "japan", "solomon", "mt", "nigeria", "gabon", "turkey", "borneo", "non", "event", "describe", "canada", "united", "kenya", "wale", "peninsular", "subpopulation", "subspecie", "conservation", "research", "management", "protection", "length", "protect", "distribution", "nacional", "indian", "indo", "philippine", "peninsula", "quantify", "global", "local", "sumatra", "bay", "accord", "state", "american", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "australian", "gulf", "monitor", "arrival", "sri", "reserva", "sarawak", "common", "uncommon", "relatively", "rare", "fairly", "portugal", "greece", "russia", "country", "region", "present", "location", "distribute", "category", "experience", "identify", "range", "monitoring", "status", "legislation", "grande", "database", "del", "near", "widespread", "appear", "recent", "lanka", "million", "maximum", "implement", "establish", "extremely", "sabah", "confirm", "exist", "conduct", "additional", "abundant", "taxon", "red", "effort", "act", "nepal", "kalimantan", "río", "switzerland", "malawi", "critically", "austria", "taxonimic", "speciman", "specimen", "morocco", "southeastern", "nord", "northwestern", "understand", "northeastern", "southwestern", "assume", "bcgi", "democratic", "determine", "genus", "°", "uganda", "czech", "hungary", "romania", "angola", "los", "cordillera", "victoria", "lao", "amazon", "s", "madeira", "canary"}

# pickle load dfClean
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# create training corpus (50% of every column)
rat = df.rationale.values.tolist()
hab = df.habitat.values.tolist()
thr = df.threats.values.tolist()
pop = df.population.values.tolist()
ran = df.range.values.tolist()
use = df.useTrade.values.tolist()
con = df.conservationActions.values.tolist()
# combine into training corpus
corpus = rat + hab + thr + pop + ran + use + con
del rat,hab,thr,pop,ran,use,con
corpus = list(filter(None, corpus))

nlp = spacy.load('en_core_web_md') # load medium sized english vocab

corpusGen = nlp.pipe(corpus, n_process = 4, batch_size = 100, disable = ['parser', 'ner'])

tokens = []
for doc in corpusGen:
    tokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP'])

with open("../Data/fullTokens.pkl", "wb") as f:
    pickle.dump(tokens, f)
    f.close()

dictionary = corpora.Dictionary(tokens)
dictionary.save("../Data/fulDictionary.dict")
bowCorpus = [dictionary.doc2bow(doc) for doc in tokens]
corpora.MmCorpus.serialize('../Data/fulbowCorpus.mm', bowCorpus)

# load
bowCorpus = corpora.MmCorpus("../Data/fulbowCorpus.mm")
dictionary = corpora.Dictionary.load("../Data/fulDictionary.dict")

tfidf = gensim.models.tfidfmodel.TfidfModel(bowCorpus, smartirs='ntc')
# save model
tfidf.save("../Data/tfidfModel.model")
# work out how to use tfidf to remove useless words
tfidf = gensim.models.tfidfmodel.TfidfModel.load('../Data/tfidfModel.model')

tfidfCorpus = tfidf[bowCorpus]

tfidfWeights = {dictionary.get(id): value for doc in tfidfCorpus for id, value in doc}

tfidfWeightsSorted = sorted(tfidfWeights.items(), key = lambda w:w[1])

# low tf idf weight = stop words
tfidfWeightsSorted[:200]
tfidfWeightsSorted[-200:]

