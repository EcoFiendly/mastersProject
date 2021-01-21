#!/usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import re
import string
# from pprint import pprint

# NLTK stopwords
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# from gensim.models import CoherenceModel

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

df.head(1)
df.iloc[1]

print(df.columns.values)
# ['assessmentId' 'internalTaxonId' 'scientificName' 'redlistCategory'
#  'redlistCriteria' 'yearPublished' 'assessmentDate' 'criteriaVersion'
#  'language' 'rationale' 'habitat' 'threats' 'population' 'populationTrend'
#  'range' 'useTrade' 'systems' 'conservationActions' 'realm' 'yearLastSeen'
#  'possiblyExtinct' 'possiblyExtinctInTheWild' 'scopes']


df['language'].unique()

# 860 in portuguese
# 48 in french
# 1539 in spanish; castilian
# drop entries not in english, if not LDA will pick them up and put them in the same topic
df = df[df.language == 'English']

# col = ['assessmentId', 'internalTaxonId', 'scientificName', 'rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions', 'realm', 'scopes']

# extract columns of interest
# col = ['assessmentId', 'redlistCategory', 'redlistCriteria', 'yearPublished', 'assessmentDate', 'criteriaVersion', 'language', 'populationTrend', 'systems', 'yearLastSeen', 'possiblyExtinct', 'possiblyExtinctInTheWild']
# dfPrep = df.drop(columns = col)

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

# iterate over specific columns
for column in df[['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions']]:
    df[column] = df[column].map(lambda x: cleanOne(x))

# save round one of cleaning
df.to_csv("../Data/dfCleanOnce.csv", index = False)

# load df
df = pd.read_csv("../Data/dfCleanOnce.csv", nrows = 5)

# convert to list
rationale = df.rationale.values.tolist()

# might be able to just use str.split()
# tokenize and clean-up using gensim's simple_preprocess()
def sent2Words(text):
    for sentence in text:
        yield(simple_preprocess(str(sentence), deacc = True))

tokens = list(sent2Words(rationale))

# removing stopwords
tokensNoStop = []
# for row in tokens:
    # newTerm = []
    # for word in doc:
    #     if not word in stopwords.words('english'):
    #         newTerm.append(word)
    # tokensNoStop.append(newTerm)
tokensNoStop = [[word for word in row if word not in stopwords.words('english')] for row in tokens]

# lemmatize: convert words to root words
nlp = spacy.load('en_core_web_sm')

def lem(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    textOut = []
    for sent in text:
        doc = nlp(" ".join(sent))
        textOut.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return textOut

rationaleLem = lem(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# started at 11:27, finished at 11:52

# pickle rationaleLem
with open("../Data/rationaleLem.pkl", "wb") as f:
    pickle.dump(rationaleLem, f)
# pickle load rationaleLem
with open("../Data/rationaleLem.pkl", "rb") as f:
    test = pickle.load(f)

# try nlp.pipe

# create bag of words from rationale
rationaleDict = corpora.Dictionary(rationaleLem)

count = 0
for k, v in rationaleDict.iteritems():
    print(k, v)
    count += 1
    if count > 20:
        break

# filter 
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens
rationaleDict.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
# this example:
# removes tokens in dictionary that appear in less than 5 documents
# removes tokens in dictionary that appear in more than 0.5 of total corpus size
# after the above 2, keep only first 100000 most frequent tokens (or keep all if keep_n=None)

rationaleCorpus = [rationaleDict.doc2bow(doc) for doc in rationaleLem]

rationaleDoc5 = rationaleCorpus[5]
for i in range(len(rationaleDoc5)):
    print('Word {} (\"{}\") appears {} time.'.format(rationaleDoc5[i][0],
                                                    rationaleDict[rationaleDoc5[i][0]],
                                                    rationaleDoc5[i][1]))

ldaModel = gensim.models.LdaMulticore(rationaleCorpus, num_topics = 10, id2word = rationaleDict, passes = 2, workers = 6)

for idx, topic in ldaModel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
