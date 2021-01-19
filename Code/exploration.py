#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re
from pprint import pprint

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmetization
import spacy

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

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

# remove punctuation
