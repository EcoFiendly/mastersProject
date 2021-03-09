#!/usr/bin/env python3

# text cleaning
import numpy as np
import pickle
import pandas as pd
import re
import string
from multiprocessing import Pool

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
# df.isnull().any()
# check for number of NaN in columns
df.isnull().sum()
# np.where(pd.isnull(df.rationale)) # prints NaN cell number
# skip pop, range, usetrade, focus on habitat, conservationActions

# replace NaNs with 0
df[['rationale','habitat','threats','population','range','useTrade','conservationActions']] = df[['rationale','habitat','threats','population','range','useTrade','conservationActions']].fillna(0)

def cleanOne(text):
    text = str(text) # convert cell values to string
    text = text.lower() # lower case all text
    text = re.sub(r'<.*?>', ' ', text) # remove html tags
    text = re.sub(r'(&#160;)', ' ', text) # remove html non-breaking space
    text = re.sub(r'\[.*?\]', ' ', text) # remove text in square brackets
    text = re.sub(r'\(.*?\)', ' ', text) # remove text in parentheses
    # text = re.sub(r'\w+\s*(and\s){0,1}\w+\s*(et al.)*\s\d{4}', '', text) #remove in text citation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuation
    text = re.sub(r'\d+\s\w{1,2}\b', ' ', text) # remove measurements
    text = re.sub(r'\w*\d\w*', ' ', text) # remove words containing numbers
    text = re.sub(r'\b\S{1,3}\b', ' ', text) # remove single to 3 characters
    text = re.sub(r'\b°[nsew]\b', ' ', text) # remove longtitude and latitude units
    text = " ".join(text.split()) # remove extra spaces, \n, \t
    return text

# aoo and eoo = area of occupancy and extent of occurrence

# # iterate over specific columns
# for column in df[['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions']]:
#     df[column] = df[column].map(lambda x: cleanOne(x))

# parallelize cleaning of columns
def parClean(df):
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

dfClean = parallelDf(df, parClean)

# saving to csv and loading causes NaN to regenerate, pickle doesn't
# pickle dfClean
with open("../Data/dfClean.pkl", "wb") as f:
    pickle.dump(dfClean, f)
    f.close()
