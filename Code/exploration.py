#!/usr/bin/env python3

import numpy as np
import pickle
import pandas as pd

# for transparently opening remote files
# from smart_open import open

# text cleaning
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
# import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
from wordcloud import WordCloud
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
df.isnull().sum()
# np.where(pd.isnull(df.rationale)) # prints NaN cell number

# replace NaNs with 0
df[['rationale','habitat','threats','population','range','useTrade','conservationActions']] = df[['rationale','habitat','threats','population','range','useTrade','conservationActions']].fillna(0)

def cleanOne(text):
    text = str(text) # convert cell values to string
    text = text.lower() # lower case all text
    text = re.sub('<.*?>', '', text) # remove html tags
    text = re.sub('(&#160;)', ' ', text) # remove html non-breaking space
    text = re.sub('\[.*?\]', '', text) # remove text in square brackets
    text = re.sub('\(.*?\)', '', text) # remove text in parentheses
    # text = re.sub('\w+\s*(and\s){0,1}\w+\s*(et al.)*\s\d{4}', '', text) #remove in text citation
    # text = re.sub('\(..\)', '', text) # remove extinction status acronym
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuation
    text = re.sub('\d+\s\w{1,2}\s', ' ', text) # remove measurements
    text = re.sub('\w*\d\w*', ' ', text) # remove words containing numbers
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
# pickle load dfClean
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# create corpus
rat = df.rationale.values.tolist()[::2]
hab = df.habitat.values.tolist()[::2]
thr = df.threats.values.tolist()[::2]
pop = df.population.values.tolist()[::2]
ran = df.range.values.tolist()[::2]
use = df.useTrade.values.tolist()[::2]
con = df.conservationActions.values.tolist()[::2]

testCorpus = rat + hab + thr + pop + ran + use + con
del rat,hab,thr,pop,ran,use,con
testCorpus = list(filter(None, testCorpus))

# # write to txt file
# with open("../Data/corpus.txt", "w") as f:
#     for row in corpus:
#         f.write(str(row) + '\n')
#     f.close()

# class MyCorpus: # corpus streaming - one doc at a time (ram friendly)
#     def __iter__(self):
#         for line in open("../Data/corpus.txt"):
#             # assume there's one doc per line, tokens separated by whitespace
#             yield dictionary.doc2bow(line.lower().split())

# corpus = MyCorpus()

# lemmatize: convert words to root words
nlp = spacy.load('en_core_web_md')

# can't do entire corpus at once, do columns separately before combining
# generators help decrease ram usage (keep output of nlp.pipe as a generator)
# don't need too many cores since tokenization cannot be multithreaded
testCorpusGen = nlp.pipe(testCorpus, n_process = 3, batch_size = 100, disable = ['parser','ner'])

testTokens = []
for doc in testCorpusGen:
    testTokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and tok.text and not tok.is_punct])

# pickle ratTokens
with open("../Data/testTokens.pkl", "wb") as f:
    pickle.dump(testTokens, f)
# pickle load ratTokens
with open("../Data/testTokens.pkl", "rb") as f:
    testTokens = pickle.load(f)
    f.close()

# create a dictionary
testDict = corpora.Dictionary(testTokens)

# filter tokens
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
testDict.filter_extremes(no_below=349, no_above=0.8)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)
testDict.compactify() # assign new word ids to all words, shrinking any gaps

# save dictionary
testDict.save("../Data/testDict.dict")
# load dictionary
testDict = corpora.Dictionary.load("../Data/testDict.dict")

# bag of words corpus
testCorpus = [testDict.doc2bow(doc) for doc in testTokens]
# save corpus and serializing decreases ram usage (by a lot)
corpora.MmCorpus.serialize('../Data/testCorpus.mm', testCorpus)
# load corpus
testCorpus = corpora.MmCorpus('../Data/testCorpus.mm')

#########################################################################
# don't need to remove replicates yet
# code to check if replicates are removed by counting the occurrences
# {i:ratTokens[0].count(i) for i in set(ratTokens[0])}
#########################################################################

# # pickle ratDict
# with open("../Data/ratDict.pkl", "wb") as f:
#     pickle.dump(ratDict, f)
# # pickle load ratDict
# with open("../Data/ratDict.pkl", "rb") as f:
#     ratDict = pickle.load(f)

count = 0
for k, v in testDict.iteritems():
    print(k, v)
    count += 1
    if count > 20:
        break

# training corpus
# trainCorpus = [testDict.doc2bow(doc) for doc in testTokens]

# example of words and frequency in doc 6
trainDoc5 = trainCorpus[5]
for i in range(len(trainDoc5)):
    print('Word {} (\"{}\") appears {} time.'.format(trainDoc5[i][0],
                                                     trainDict[trainDoc5[i][0]],
                                                     trainDoc5[i][1]))

# run LDA model using Bag of Words
testModel = gensim.models.LdaMulticore(testCorpus, num_topics = 14, id2word = testDict, chunksize = 70, passes = 5, workers = 6)
# save model
testModel.save('../Data/testModel.model')
# load model
testModel = gensim.models.LdaMulticore.load('../Data/testModel.model')

# show topics and corresponding weights of words in topics
for idx, topic in testModel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# performance evaluation by classifying sample document using LDA BoW model
ratTokens[1225]

for index, score in sorted(ldaModel[ratTokens[1225]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, ldaModel.print_topic(index, 50)))

# compute perplexity: a measure of how good the model is, lower the better
print("\nPerplexity:", testModel.log_perplexity(testCorpus))

# compute coherence score
testCohModLDA = CoherenceModel(model=testModel, texts=testTokens, dictionary=testDict, coherence='c_v')
# work out coherence
testCohLDA = testCohModLDA.get_coherence()
print("\nCoherence Score:", testCohLDA)

# visualize topics-keywords of LDA

for t in range(ldaModel.num_topics):
    f = plt.figure()
    plt.imshow(WordCloud().fit_words(dict(ldaModel.show_topic(t, 200))))
    plt.axis('off')
    plt.title('Topic #' + str(t))
    f.savefig("wordCloud" + str(t))

def modelOpti(corpus, dictionary, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    corpus: BoWcorpus 
    dictionary: Gensim dictionary
    limit: max number of topics

    Returns:
    modelList: list of LDA topic models
    cohVals: coherence values corresponding to LDA model with respective number of topics
    """
    cohVals = []
    modelList = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus, num_topics = num_topics, id2word = testDict, chunksize = 1860, passes = 15, workers = 7)
        modelList.append(model)
        cohLDA = CoherenceModel(model = model, corpus = corpus, dictionary = testDict, coherence = 'u_mass', processes = 7)
        cohVals.append(cohLDA.get_coherence())
    
    return modelList, cohVals

modelList, cohVals = modelOpti(testCorpus, testDict, limit = 27, start = 2, step = 2)

# plot graph for coherence values
limit=27;start=2;step=2
x = range(start,limit,step)
plt.plot(x, cohVals)
plt.show() # seems to peak at 14 topics

# compute perplexity: a measure of how good the model is, lower the better
print("\nPerplexity:", modelList[1].log_perplexity(testCorpus))

# show topics and corresponding weights of words in topics
for idx, topic in modelList[3].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# visualize topics-keywords of LDA

for t in range(modelList[0].num_topics):
    f = plt.figure()
    plt.imshow(WordCloud().fit_words(dict(modelList[0].show_topic(t, 200))))
    plt.axis('off')
    plt.title('Topic #' + str(t))
    f.savefig("wordCloud" + str(t))
