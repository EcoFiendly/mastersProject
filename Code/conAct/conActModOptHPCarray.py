#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# get home directory
home = os.path.expanduser('~')
# Model parameters:

# load tokens
with open(home+"/Data/conAct/conActTokens2.pkl", "rb") as f:
   conActTokens2 = pickle.load(f)
   f.close()

# load dictionary
conActDict = corpora.Dictionary.load(home+"/Data/conAct/conActDict.dict")

# load corpus
conActCorpus = corpora.MmCorpus(home+"/Data/conAct/conActBoWCorpus.mm")

iter = int(os.getenv("PBS_ARRAY_INDEX")) # read job number from cluster
chunksize = len(conActCorpus)//20 # set chunksize as 5% of corpus length
eval_every = chunksize*4 # set eval_every to chunksize*4

model = gensim.models.LdaMulticore(corpus = conActCorpus, num_topics = iter, id2word = conActDict, chunksize = chunksize, passes = 20, workers = 8, eta = 'auto', eval_every = eval_every, random_state = 95)
cv = CoherenceModel(model = model, corpus = conActCorpus, texts = conActTokens2, coherence = 'c_v', processes = 8)
cvCoh = cv.get_coherence()

with open(home+"/Data/conAct/conActModel"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open(home+"/Data/conAct/conActCv"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cvCoh, f)
    f.close()
