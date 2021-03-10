#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Model parameters:

# load tokens
with open("/rds/general/user/yl4220/home/Data/hab/habTokens3.pkl", "rb") as f:
   habTokens3 = pickle.load(f)
   f.close()

# load dictionary
habDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/hab/habDict.dict")

# load corpus
habCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/hab/habBoWCorpus.mm")

# read job number from cluster
iter = int(os.getenv("PBS_ARRAY_INDEX"))

# set chunksize as 5% of corpus length
chunksize = len(habCorpus)//20

model = gensim.models.LdaMulticore(corpus = habCorpus, num_topics = iter, id2word = habDict, chunksize = chunksize, passes = 20, workers = 8, eval_every = chunksize, random_state= 95)
cv = CoherenceModel(model = model, corpus = habCorpus, texts = habTokens3, coherence = 'c_v', processes = 8)
cvCoh = cv.get_coherence()

with open("/rds/general/user/yl4220/home/Data/hab/habModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/hab/habCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
