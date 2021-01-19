#!/usr/bin/env python3

# import numpy as np
import pandas as pd
import re
import string
# from pprint import pprint

# NLTK stopwords
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

# gensim
# import gensim
# import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# from gensim.models import CoherenceModel

# spacy for lemmetization
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
col = ['redlistCategory', 'redlistCriteria', 'yearPublished', 'assessmentDate', 'criteriaVersion', 'language', 'populationTrend', 'systems', 'yearLastSeen', 'possiblyExtinct', 'possiblyExtinctInTheWild']

dfPrep = df.drop(columns = col)

dfPrep.to_csv("../Data/prep.csv")
del dfPrep
dfPrep = pd.read_csv("../Data/prep.csv")

print(dfPrep.columns.values)
dfPrep = dfPrep.drop(columns = 'Unnamed: 0')

def cleanOne(text):
    # convert cell values to string
    text = str(text)
    # lower case all text
    text = text.lower()
    # remove html tags
    text = re.sub('<.*?>', '', text)
    # remove text in square brackets
    text = re.sub('\[.*?\]', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    # remove extra spaces
    # text = re.sub(' {2,}', ' ', text)
    text = " ".join(text.split())
    return text

# removing stopwords
stopWords = stopwords.words('english')

def removeStopwords(text):
    return [[word for word in simple_preprocess(doc) if word not in stopWords] for doc in text]

# iterate over specific columns
for column in dfPrep[['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions', 'realm', 'scopes']]:
    dfPrep[column] = dfPrep[column].map(lambda x: cleanOne(x))
    # tokenize before removing stopwords
    # dfPrep[column] = dfPrep[column].map(lambda x: removeStopwords(x))

# convert to list
rationale = dfPrep.rationale.values.tolist()

# tokenize and clean-up using gensim's simple_preprocess()
def sent2Words(text):
    for sentence in text:
        yield(simple_preprocess(str(sentence), deacc = True))

tokens = list(sent2Words(rationale))

# remove stopwords from tokens


# lemmatize: convert words to root words

