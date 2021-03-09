#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# pickle load combTokens
# only required if using c_v as coherence metric
with open("/rds/general/user/yl4220/home/Data/comb/combTokens3.pkl", "rb") as f:
   combTokens3 = pickle.load(f)
   f.close()

# load dictionary
combDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/comb/combDict.dict")

# load corpus
combCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/comb/combBoWCorpus.mm")

# read job number from cluster
iter = int(os.getenv("PBS_ARRAY_INDEX"))

model = gensim.models.LdaMulticore(combCorpus, num_topics = iter, id2word = combDict, chunksize = 8000, passes = 15, workers = 8, eval_every = None)
npmi = CoherenceModel(model = model, corpus = combCorpus, texts = combTokens3, coherence = 'c_npmi', processes = 8)
npmiCoh = npmi.get_coherence()
cv = CoherenceModel(model = model, corpus = combCorpus, texts = combTokens3, coherence = 'c_v', processes = 8)
try:
    cvCoh = cv.get_coherence()
except RuntimeWarning:
    cvCoh = 0

with open("/rds/general/user/yl4220/home/Data/comb/combModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/comb/combNpmi"+str(iter)+".pkl", "wb") as f:
    pickle.dump(npmiCoh, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/comb/combCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
