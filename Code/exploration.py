#!/usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import re
import string
from multiprocessing import Pool
# from pprint import pprint

# NLTK stopwords
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

# gensim
import gensim
import gensim.corpora as corpora
# from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

# spacy for lemmatization
# spacy easier to use compared to nltk WordNetLemmatizer
import spacy

# plotting tools
# import pyLDAvis
# import pyLDAvis.gensim
# import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

# import data
df = pd.read_csv("../Data/assessments.csv")

print(df.columns.values)
# ['assessmentId' 'internalTaxonId' 'scientificName' 'redlistCategory'
#  'redlistCriteria' 'yearPublished' 'assessmentDate' 'criteriaVersion'
#  'language' 'rationale' 'habitat' 'threats' 'population' 'populationTrend'
#  'range' 'useTrade' 'systems' 'conservationActions' 'realm' 'yearLastSeen'
#  'possiblyExtinct' 'possiblyExtinctInTheWild' 'scopes']

df['language'].unique()
# drop entries not in english, if not LDA will pick them up and put them in the same topic
df = df[df.language == 'English']

# check for NaN in columns
df.isnull().any()
# check for number of NaN in columns
df.column.isnull().sum()

def cleanOne(text):
    text = str(text) # convert cell values to string
    text = text.lower() # lower case all text
    text = re.sub('<.*?>', '', text) # remove html tags
    text = re.sub('(&#160;)', ' ', text) # remove html non-breaking space
    text = re.sub('\[.*?\]', '', text) # remove text in square brackets
    text = re.sub('\w+\s*(and\s){0,1}\w+\s*(et al.)*\s\d{4}', '', text) #remove in text citation
    text = re.sub('\(..\)', '', text) # remove extinction status acronym
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = " ".join(text.split()) # remove extra spaces, \n, \t
    return text

# aoo and eoo = area of occupancy and extent of occurrence

# parallelize cleaning of columns
def addFeats(df):
    df['rationale'] = df['rationale'].map(lambda x: cleanOne(x))
    df['habitat'] = df['habitat'].map(lambda x: cleanOne(x))
    df['threats'] = df['threats'].map(lambda x: cleanOne(x))
    df['population'] = df['population'].map(lambda x: cleanOne(x))
    df['range'] = df['range'].map(lambda x: cleanOne(x))
    df['useTrade'] = df['useTrade'].map(lambda x: cleanOne(x))
    df['conservationActions'] = df['conservationActions'].map(lambda x: cleanOne(x))
    return df

def parallelDf(df, func, n_cores = 7):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

dfClean = parallelDf(df, addFeats)

# saving to csv and loading causes NaN to regenerate, pickle doesn't
# pickle dfClean
with open("../Data/dfClean.pkl", "wb") as f:
    pickle.dump(dfClean, f)
# pickle load dfClean
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)

# convert to list and filter empty strings
rationale = df.rationale.values.tolist()
rationale = list(filter(None, rationale))

# lemmatize: convert words to root words
nlp = spacy.load('en_core_web_sm')

ratDocs = list(nlp.pipe(rationale, n_process = 6, disable=['parser','ner']))
ratTokens = []
for doc in ratDocs:
    ratTokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and tok.text and not tok.is_punct])

# pickle ratTokens
with open("../Data/ratTokens.pkl", "wb") as f:
    pickle.dump(ratTokens, f)
# pickle lead ratTokens
with open("../Data/ratTokens.pkl", "rb") as f:
    ratTokens = pickle.load(f)

#########################################################################
# don't need to remove replicates yet
# code to check if replicates are removed by counting the occurrences
# {i:ratTokens[0].count(i) for i in set(ratTokens[0])}
#########################################################################

# sample every alternate token
ratTokens = ratTokens[::2]

# create bag of words from rationale
ratDict = corpora.Dictionary(ratTokens)

# pickle ratDict
with open("../Data/ratDict.pkl", "wb") as f:
    pickle.dump(ratDict, f)
# pickle load ratDict
with open("../Data/ratDict.pkl", "rb") as f:
    ratDict = pickle.load(f)

count = 0
for k, v in ratDict.iteritems():
    print(k, v)
    count += 1
    if count > 20:
        break

# filter 
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
ratDict.filter_extremes(no_below=45, no_above=0.8)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)

# document term matrix
ratCorpus = [ratDict.doc2bow(doc) for doc in ratTokens]

# example of words and frequency in doc 6
ratDoc5 = ratCorpus[5]
for i in range(len(ratDoc5)):
    print('Word {} (\"{}\") appears {} time.'.format(ratDoc5[i][0],
                                                    ratDict[ratDoc5[i][0]],
                                                    ratDoc5[i][1]))

# run LDA model using Bag of Words
ldaModel = gensim.models.LdaMulticore(ratTrain, num_topics = 50, id2word = ratDict, passes = 5, workers = 6)

# show topics and corresponding weights of words in topics
for idx, topic in ldaModel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# performance evaluation by classifying sample document using LDA BoW model
ratTokens[1225]

for index, score in sorted(ldaModel[ratTokens[1225]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, ldaModel.print_topic(index, 50)))

# compute perplexity: a measure of how good the model is, lower the better
print("\nPerplexity:", ldaModel.log_perplexity(ratCorpus))

# compute coherence score
coherenceModelLDA = CoherenceModel(model=ldaModel, texts=ratTokens, dictionary=ratDict, coherence='c_v')
# work out coherence
coherenceLDA = coherenceModelLDA.get_coherence()
print("\nCoherence Score:", coherenceLDA)