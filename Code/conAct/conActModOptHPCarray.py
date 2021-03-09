#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# pickle load trainTokens
# only required if using c_v as coherence metric
with open("/rds/general/user/yl4220/home/Data/conAct/conActTokens3.pkl", "rb") as f:
   conActTokens3 = pickle.load(f)
   f.close()

# load dictionary
conActDict = corpora.Dictionary.load("/rds/general/user/yl4220/home/Data/conAct/conActDict.dict")

# load corpus
conActCorpus = corpora.MmCorpus("/rds/general/user/yl4220/home/Data/conAct/conActBoWCorpus.mm")

# read job number from cluster
iter = int(os.getenv("PBS_ARRAY_INDEX"))

model = gensim.models.LdaMulticore(conActCorpus, num_topics = iter, id2word = conActDict, chunksize = 8000, passes = 15, workers = 8, eval_every = None)
npmi = CoherenceModel(model = model, corpus = conActCorpus, texts = conActTokens3, coherence = 'c_npmi', processes = 8)
npmiCoh = npmi.get_coherence()
cv = CoherenceModel(model = model, corpus = conActCorpus, texts = conActTokens3, coherence = 'c_v', processes = 8)
try:
    cvCoh = cv.get_coherence()
except RuntimeWarning:
    cvCoh = 0

with open("/rds/general/user/yl4220/home/Data/conAct/conActModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/conAct/conActNpmi"+str(iter)+".pkl", "wb") as f:
    pickle.dump(npmiCoh, f)
    f.close()

with open("/rds/general/user/yl4220/home/Data/conAct/conActCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
