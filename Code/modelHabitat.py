#!/usr/bin/env python3

import pickle
import spacy

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

# pickle load dfClean
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

habCorpus = df.habitat.values.tolist()
habCorpus = list(filter(None, habCorpus))

nlp = spacy.load('en_core_web_md')

habCorpusGen = nlp.pipe(habCorpus, n_process = 6, batch_size = 200)

habTokens = []
for doc in habCorpusGen:
    habTokens.append([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and not tok.lemma_ == 'species'])

# ('species', 'NOUN', 'specie', 'NNS', 'nsubjpass'),
# ('species', 'NOUN', 'specie', 'NNS', 'nmod'),
# ('species', 'NOUN', 'species', 'NN', 'pobj'),
# ('species', 'NOUN', 'specie', 'NNS', 'poss'),
# ('species', 'NOUN', 'specie', 'NNS', 'dobj'),

# pickle habTokens
with open("../Data/perCol/habTokens.pkl", "wb") as f:
    pickle.dump(habTokens, f)

habDict = corpora.Dictionary(habTokens)

habDict.filter_extremes(no_below= 104, no_above=0.8)
habDict.compactify()
habDict.save("../Data/perCol/habDict.dict")
habDict = corpora.Dictionary.load("../Data/perCol/habDict.dict")

habCorpus = [habDict.doc2bow(doc) for doc in habTokens]
corpora.MmCorpus.serialize("../Data/perCol/habCorpus.mm", habCorpus)
habCorpus = corpora.MmCorpus("../Data/perCol/habCorpus.mm")

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
        model = gensim.models.LdaMulticore(corpus, num_topics = num_topics, id2word = dictionary, chunksize = 980, passes = 20, workers = 7, eval_every = None)
        modelList.append(model)
        cohLDA = CoherenceModel(model = model, corpus = corpus, dictionary = dictionary, coherence = 'u_mass', processes = 7)
        cohVals.append(cohLDA.get_coherence())
    
    return modelList, cohVals

modelList, cohVals = modelOpti(habCorpus, habDict, limit = 17, start = 2, step = 2)

limit=17;start=2;step=2
x = range(start,limit,step)
plt.plot(x, cohVals)
plt.show() # seems to peak at 14 topics

# check 0 and 4
# show topics and corresponding weights of words in topics
for idx, topic in modelList[0].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for t in range(modelList[0].num_topics):
    i = "habitat"
    j = 2
    f = plt.figure()
    plt.imshow(WordCloud().fit_words(dict(modelList[0].show_topic(t, 200))))
    plt.axis('off')
    plt.title(i + str( j) + ' Topic #' + str(t))
    f.savefig("../Results/habWC/" + i + str(j) + "WordCloud" + str(t))

with open("../Data/perCol/habModelList.pkl", "wb") as f:
    pickle.dump(modelList, f)
    f.close()

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(modelList[0], habCorpus, habDict)
pyLDAvis.show(vis)