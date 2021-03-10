#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# pickle load trainTokens
# only required if using c_v as coherence metric
with open("/rds/general/user/yl4220/home/Data/hab/habTokens3.pkl", "rb") as f:
   habTokens3 = pickle.load(f)
   f.close()

# load dictionary
habDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/hab/habDict.dict")

# load corpus
habCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/hab/habBoWCorpus.mm")

# read job number from cluster
iter = int(os.getenv("PBS_ARRAY_INDEX"))

model = gensim.models.LdaMulticore(habCorpus, num_topics = iter, id2word = habDict, chunksize = 8000, passes = 15, workers = 8, eval_every = None)
cv = CoherenceModel(model = model, corpus = habCorpus, texts = habTokens3, coherence = 'c_v', processes = 8)
cvCoh = cv.get_coherence()

with open("/rds/general/user/yl4220/home/Data/hab/habModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/hab/habCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
