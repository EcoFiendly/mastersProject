#!/usr/bin/env python3

import os

import pandas as pd

# load files
import pickle

# spacy for lemmatization
# spacy easier to use compared to nltk WordNetLemmatizer
import spacy

# creating dictionary and vectorized corpus
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

# get working directory
home = os.path.expanduser('~')

# pickle load dfClean
with open(home+"/Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

#with open("../../Data/dfClean.pkl", "rb") as f:
#    df = pickle.load(f)
#    f.close()

# create training corpus
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

# save combCorpus
with open(home+"/Data/comb/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
    f.close()

nlp = spacy.load('en_core_web_sm') # load small sized english vocab

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"situ", "appendix", "cite", "annex", "need", "book", "find"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

# test if stop words added (need added but not returning false)
# print([t.is_stop for t in nlp("Río")])
# can't do entire corpus at once, do columns separately before combining
# generators help decrease ram usage (keep output of nlp.pipe as a generator)
# don't need too many cores since tokenization cannot be multithreaded
corpus_gen = nlp.pipe(corpus, n_process = 8, batch_size = 800, disable = ["ner"])

del corpus

# # experimenting with token tags
# corpus[100]
# test = nlp(corpus[1010])
# for tok in test:
#     print(tok.lemma_, tok, tok.tag_, tok.dep_)

# for ent in test.ents:
#     print(ent.text, ent.label_)

# corpus_gen = nlp.pipe(corpus[1000:1010], n_process = 8, batch_size = 800, disable = ["parser", "ner"])

# tokens = []
# for doc in corpus_gen:
#     tokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.dep_ != 'det' and tok.dep_ != 'auxpass' and tok.tag_ != '_SP' and tok.tag_ != 'EX'])
# # end experiment

tokens = []
for doc in corpus_gen:
    # tokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.dep_ != 'det' and tok.dep_ != 'auxpass' and tok.tag_ != '_SP' and tok.tag_ != 'EX'])

    tokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP' and tok.tag_ != 'NNPS' and tok.tag_ != 'VBG' and tok.tag_ != '_SP'])

min_count = int(0.01*len(tokens))
# build bigram model
bigram = Phrases(tokens, min_count = min_count)
bigramMod = Phraser(bigram)
# print(bigramMod[combTokens2[1]])

tokens_2 = bigramMod[tokens]

# pickle combTokens3
with open(home+"/Data/comb/tokens_2.pkl", "wb") as f:
    pickle.dump(tokens_2, f)
    f.close()

# create a dictionary
dic = corpora.Dictionary(tokens_2)

# filter tokens
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
no_below = int(0.01*len(tokens_2))
dic.filter_extremes(no_below=no_below, no_above=0.5)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)
dic.compactify() # assign new word ids to all words, shrinking any gaps

# save dictionary
dic.save(home+"/Data/comb/dic.dict")

# # look at dictionary items
# count = 0
# for k, v in combDict.iteritems():
#     print(k, v)
#     count += 1
#     if count > 50:
#         break

# bag of words corpus
bow_corpus = [dic.doc2bow(doc) for doc in tokens_2]
# save corpus and serializing decreases ram usage (by a lot)
corpora.MmCorpus.serialize(home+"/Data/comb/bow_corpus.mm", bow_corpus)
