#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Model parameters:

# load tokens
with open("/rds/general/user/yl4220/home/Data/rat/ratTokens3.pkl", "rb") as f:
   ratTokens3 = pickle.load(f)
   f.close()

# load dictionary
ratDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/rat/ratDict.dict")

# load corpus
ratCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/rat/ratBoWCorpus.mm")

iter = int(os.getenv("PBS_ARRAY_INDEX")) # read job number from cluster
chunksize = len(ratCorpus)//20 # set chunksize as 5% of corpus length
eval_every = chunksize*4 # set eval_every to chunksize*4

model = gensim.models.LdaMulticore(corpus = ratCorpus, num_topics = iter, id2word = ratDict, chunksize = chunksize, passes = 20, workers = 8, eval_every = eval_every, random_state = 95)
cv = CoherenceModel(model = model, corpus = ratCorpus, texts = ratTokens3, coherence = 'c_v', processes = 8)
cvCoh = cv.get_coherence()

with open("/rds/general/user/yl4220/home/Data/rat/ratModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/rat/ratCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
