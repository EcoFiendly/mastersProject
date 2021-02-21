#!/usr/bin/env python3

import pickle

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# pickle load trainTokens
# only required if using c_v as coherence metric
# with open("/rds/general/user/yl4220/home/Data/trainTokens3.pkl", "rb") as f:
#    trainTokens3 = pickle.load(f)
#    f.close()

# load dictionary
trainDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/trainDict.dict")

# load corpus
trainCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/trainBoWCorpus.mm")

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
        model = gensim.models.LdaMulticore(corpus, num_topics = num_topics, id2word = dictionary, chunksize = 700, passes = 15, workers = 8, eval_every = None)
        modelList.append(model)
        cohLDA = CoherenceModel(model = model, corpus = corpus, dictionary = dictionary, coherence = 'u_mass', processes = 8)
        cohVals.append(cohLDA.get_coherence())
    
    return modelList, cohVals

modelList, cohVals = modelOpti(trainCorpus, trainDict, limit = 201, start = 2, step = 2)

with open("/rds/general/user/yl4220/home/Data/modelList.pkl", "wb") as f:
    pickle.dump(modelList, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/cohVals.pkl", "wb") as f:
    pickle.dump(cohVals, f)
    f.close()
