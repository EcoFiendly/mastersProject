#!/usr/bin/env python3

import pickle
# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# pickle load trainTokens
# only required if using c_v as coherence metric
with open("../Data/trainTokens.pkl", "rb") as f:
    trainTokens2 = pickle.load(f)
    f.close()

# load dictionary
trainDict = corpora.Dictionary.load("../Data/trainDict.dict")

# load corpus
trainCorpus = corpora.MmCorpus('../Data/trainBoWCorpus.mm')

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
    # cohVals = []
    modelList = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus, num_topics = num_topics, id2word = dictionary, chunksize = 700, passes = 10, workers = 6, eval_every = None)
        modelList.append(model)
        # cohLDA = CoherenceModel(model = model, corpus = corpus, dictionary = dictionary, coherence = 'u_mass', processes = 7)
        # cohVals.append(cohLDA.get_coherence())
    
    return modelList #, cohVals

modelList = modelOpti(trainCorpus, trainDict, limit = 13, start = 10, step = 2)

# plot graph for coherence values
limit=101;start=2;step=2
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cohVals)
plt.xlabel('Number of topics')
plt.ylabel('u_mass coherence value')
f.savefig("coherencePlot")
plt.show() # seems to peak at 14 topics

# compute perplexity: a measure of how good the model is, lower the better
print("\nPerplexity:", modelList[1].log_perplexity(trainCorpus))

# show topics and corresponding weights of words in topics
for idx, topic in modelList[3].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# visualize topics-keywords of LDA

for t in range(modelList[2].num_topics):
    f = plt.figure()
    plt.imshow(WordCloud(width=1500, height=750).fit_words(dict(modelList[2].show_topic(t, 50))))
    plt.axis('off')
    plt.title('Topic #' + str(t))
    f.savefig("wordCloud2" + str(t))
    plt.show()

# with open("../Data/modelList.pkl", "wb") as f:
#     pickle.dump(modelList, f)
#     f.close()

with open("../Data/modelList.pkl", "rb") as f:
    modelList = pickle.load(f)
    f.close()

with open("../Data/cohVals.pkl", "rb") as f:
    cohVals = pickle.load(f)
    f.close()

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(modelList[0], trainCorpus, trainDict)
pyLDAvis.show(vis)