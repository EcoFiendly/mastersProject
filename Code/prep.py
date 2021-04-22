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

# df['language'].unique()
# drop entries not in english, if not LDA will pick them up and put them in the same topic
# df = df[df.language == 'English']

# select only global assessments
# df2 = df[df.scopes.str.contains('Global')]
# df3 = df2.dropna(subset=['rationale', 'habitat', 'threats', 'population', 'range', 'useTrade', 'conservationActions'])

# check for NaN in columns
# df.isnull().any()
# check for number of NaN in columns
# df2.isnull().sum()
# np.where(pd.isnull(df.rationale)) # prints NaN cell number

def tidy_one(df):
    '''
    Selects English only entries
    Selects only global assessments
    Replace NaNs with 0
    '''
    df = df[df.language == 'English']
    df = df[df.scopes.str.contains('Global')]
    # replace NaNs with 0
    df[['rationale','habitat','threats','population','range','useTrade','conservationActions']] = df[['rationale','habitat','threats','population','range','useTrade','conservationActions']].fillna(0)
    return df

df = tidy_one(df)

def clean_one(text):
    text = str(text) # convert cell values to string
    # text = text.lower() # lower case all text
    text = re.sub(r'<.*?>', ' ', text) # remove html tags
    text = re.sub(r'(&#160;)', ' ', text) # remove html non-breaking space
    text = re.sub(r'\[.*?\]', ' ', text) # remove text in square brackets
    # text = re.sub(r'\(.*?\)', ' ', text) # remove text in parentheses
    text = re.sub(r'\(([^)]+)?(?:19|20)\d{2}?([^)]+)?\)', ' ', text) #remove in text citation
    # text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuation
    text = re.sub(r'\d+\s\w{1,2}\b', ' ', text) # remove measurements
    text = re.sub(r'\w*\d\w*', ' ', text) # remove words containing numbers
    text = re.sub(r'\b\S{1,2}\b', ' ', text) # remove single to 2 characters
    text = re.sub(r'\b°[nsew]\b', ' ', text) # remove longtitude and latitude units
    text = " ".join(text.split()) # remove extra spaces, \n, \t
    return text

# aoo and eoo = area of occupancy and extent of occurrence

def par_clean(df):
    '''
    Maps clean_one to columns of interest
    '''
    df['rationale'] = df['rationale'].map(lambda x: clean_one(x))
    df['habitat'] = df['habitat'].map(lambda x: clean_one(x))
    df['threats'] = df['threats'].map(lambda x: clean_one(x))
    df['population'] = df['population'].map(lambda x: clean_one(x))
    df['range'] = df['range'].map(lambda x: clean_one(x))
    df['useTrade'] = df['useTrade'].map(lambda x: clean_one(x))
    df['conservationActions'] = df['conservationActions'].map(lambda x: clean_one(x))
    return df

def parallel_df(df, func, n_cores = 7):
    '''
    Parallelise cleaning of columns
    '''
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

df_clean = parallel_df(df, par_clean)

# saving to csv and loading causes NaN to regenerate, pickle doesn't
# pickle dfClean
with open("../Data/df_clean.pkl", "wb") as f:
    pickle.dump(df_clean, f)
    f.close()

# # tally for chordates
# chordTally = pd.DataFrame(
#                 {'Class': pd.Series(['Actinopterygii', 'Aves', 'Reptilia', 'Amphibia', 'Mammalia', 'Chondrichthyes', 'Myxini', 'Cephalaspidomorphi', 'Sarcopterygii']),
#                  'Extinct': pd.Series([80, 159, 30, 35, 85, 0, 0, 1, 0]),
#                  'Extinct in Wild': pd.Series([10, 5, 3, 2, 2, 0, 0, 0, 0]),
#                  'Regionally Extinct': pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                  'Critically Endangered': pd.Series([594, 223, 324, 650, 221, 68, 1, 2, 1]),
#                  'Endangered': pd.Series([932, 460, 584, 1036, 539, 97, 2, 4, 1]),
#                  'Vulnerable': pd.Series([1178, 798, 541, 704, 557, 151, 6, 2, 1]),
#                  'Lower Risk: Conservation Dependent': pd.Series([1, 0, 2, 0, 0, 0, 0, 0, 0]),
#                  'Near Threatened': pd.Series([595, 1001, 446, 408, 363, 112, 2, 3, 0]),
#                  'Least Concern': pd.Series([12638, 8460, 5131, 3129, 3313, 517, 35, 22, 4]),
#                  'Data Deficient': pd.Series([4081, 52, 1175, 1202, 852, 241, 30, 3, 0]),
#                  'Total': pd.Series([20109, 11158, 8236, 7166, 5932, 1186, 76, 37, 7])
#                  })

# chordTally.set_index('Class', inplace=True)

# # % sufficiently assessed
# chordTally.iloc[0,1:9].sum()/chordTally.iloc[0,10:11] # Act 0.793
# chordTally.iloc[1,1:9].sum()/chordTally.iloc[1,10:11] # Ave 0.981
# chordTally.iloc[2,1:9].sum()/chordTally.iloc[2,10:11] # Rep 0.853
# chordTally.iloc[3,1:9].sum()/chordTally.iloc[3,10:11] # Amp 0.827
# chordTally.iloc[4,1:9].sum()/chordTally.iloc[4,10:11] # Mam 0.842
# chordTally.iloc[5,1:9].sum()/chordTally.iloc[5,10:11] # Cho 0.796
# chordTally.iloc[6,1:9].sum()/chordTally.iloc[6,10:11] # Myx 0.605
# chordTally.iloc[7,1:9].sum()/chordTally.iloc[7,10:11] # Cep 0.891
# chordTally.iloc[8,1:9].sum()/chordTally.iloc[8,10:11] # Sar 1.00

# # % threatened
# chordTally.iloc[0,1:8].sum()/chordTally.iloc[0,10:11] # Act 0.164 or 3310
# chordTally.iloc[1,1:8].sum()/chordTally.iloc[1,10:11] # Ave 0.222 or 2487
# chordTally.iloc[2,1:8].sum()/chordTally.iloc[2,10:11] # Rep 0.230 or 1900
# chordTally.iloc[3,1:8].sum()/chordTally.iloc[3,10:11] # Amp 0.390 or 2800
# chordTally.iloc[4,1:8].sum()/chordTally.iloc[4,10:11] # Mam 0.283 or 1682
# chordTally.iloc[5,1:8].sum()/chordTally.iloc[5,10:11] # Cho 0.360 or 428
# chordTally.iloc[6,1:8].sum()/chordTally.iloc[6,10:11] # Myx 0.144 or 11
# chordTally.iloc[7,1:8].sum()/chordTally.iloc[7,10:11] # Cep 0.297 or 11
# chordTally.iloc[8,1:8].sum()/chordTally.iloc[8,10:11] # Sar 0.428 or 3

