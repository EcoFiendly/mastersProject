#!/usr/bin/env python3

import pickle
import spacy

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# pickle load dfClean
with open("../Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

ratCorpus = df.rationale.values.tolist()
ratCorpus = list(filter(None, ratCorpus))

nlp = spacy.load('en_core_web_sm')

# add stopwords to spacy
nlp.Defaults.stop_words |= {"species", "specie", "need", "know", "measure", "situ", "list", "ex", "number", "include", "place", "require", "change", "extent", "find", "use", "available", "total", "new", "consider", "concern", "datum", "likely", "utilize", "action", "include", "site", "use", "place", "consider", "recommend", "estimate", "information", "trend", "report", "survey", "assess", "assessment", "occassionally", "level", "year", "currently", "occurrence", "km²", "de", "appendix", "north", "northern", "south", "southern", "east", "eastern", "west", "western", "et", "al", "asl", "occur", "criterion", "threshold", "occupancy", "threat", "qualify", "believe", "reason", "major", "significant", "presume", "unlikely", "evaluate", "suspect", "aoo", "eoo"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

ratCorpusGen = nlp.pipe(ratCorpus, n_process = 6, batch_size = 480)

ratTokens = []
for doc in ratCorpusGen:
    ratTokens.append(' '.join([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct]))

# pickle ratTokens
with open("../Data/perCol/ratTokens.pkl", "wb") as f:
    pickle.dump(ratTokens, f)
    f.close()

with open("../Data/perCol/ratTokens.pkl", "rb") as f:
    ratTokens = pickle.load(f)
    f.close()

ratCorpusGen2 = nlp.pipe(ratTokens, n_process = 6, batch_size = 480)

ratTokens2 = []
for doc in ratCorpusGen2:
    ratTokens2.append([(tok.lemma_) for tok in doc if not tok.is_stop])

# pickle ratTokens2
with open("../Data/perCol/ratTokens2.pkl", "wb") as f:
    pickle.dump(ratTokens2, f)
    f.close()

with open("../Data/perCol/ratTokens.pkl", "rb") as f:
    ratTokens = pickle.load(f)
    f.close()

ratDict = corpora.Dictionary(ratTokens2)

ratDict.filter_extremes(no_below= 115, no_above=0.8)
ratDict.compactify()
ratDict.save("../Data/perCol/ratDict.dict")
ratDict = corpora.Dictionary.load("../Data/perCol/ratDict.dict")

ratCorpus = [ratDict.doc2bow(doc) for doc in ratTokens2]
corpora.MmCorpus.serialize("../Data/perCol/ratCorpus.mm", ratCorpus)
ratCorpus = corpora.MmCorpus("../Data/perCol/ratCorpus.mm")

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
        model = gensim.models.LdaMulticore(corpus, num_topics = num_topics, id2word = dictionary, chunksize = 980, passes = 10, workers = 7, eval_every = None)
        modelList.append(model)
        # cohLDA = CoherenceModel(model = model, corpus = corpus, dictionary = dictionary, coherence = 'u_mass', processes = 7)
        # cohVals.append(cohLDA.get_coherence())
    
    return modelList #, cohVals

modelList = modelOpti(ratCorpus, ratDict, limit = 9, start = 2, step = 2)

limit=17;start=2;step=2
x = range(start,limit,step)
plt.plot(x, cohVals)
plt.show() # seems to peak at 14 topics

# check 0 and 4
# show topics and corresponding weights of words in topics
for idx, topic in modelList[0].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for t in range(modelList[0].num_topics):
    i = "rationale"
    j = 2
    f = plt.figure()
    plt.imshow(WordCloud().fit_words(dict(modelList[0].show_topic(t, 200))))
    plt.axis('off')
    plt.title('rationale ' + str(j) + ' Topic #' + str(t))
    # f.savefig("../Results/ratWC/" + i + str(j) + "WordCloud" + str(t))
    plt.show()

with open("../Data/perCol/ratModelList.pkl", "wb") as f:
    pickle.dump(modelList, f)
    f.close()
