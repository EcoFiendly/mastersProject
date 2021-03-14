#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Model parameters:

# load tokens
with open("/rds/general/user/yl4220/home/Data/comb/combTokens3.pkl", "rb") as f:
   combTokens3 = pickle.load(f)
   f.close()

# load dictionary
combDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/comb/combDict.dict")

# load corpus
combCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/comb/combBoWCorpus.mm")

iter = int(os.getenv("PBS_ARRAY_INDEX")) # read job number from cluster
chunksize = len(combCorpus)//20 # set chunksize as 5% of corpus length
eval_every = chunksize*4 # set eval_every to chunksize*4

model = gensim.models.LdaMulticore(corpus = combCorpus, num_topics = iter, id2word = combDict, chunksize = chunksize, passes = 20, workers = 8, eval_every = eval_every, random_state = 95)
cv = CoherenceModel(model = model, corpus = combCorpus, texts = combTokens3, coherence = 'c_v', processes = 8)
cvCoh = cv.get_coherence()

with open("/rds/general/user/yl4220/home/Data/comb/combModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/comb/combCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
