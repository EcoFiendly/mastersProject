#!/usr/bin/env python3

import pickle
# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# pickle load trainTokens
# only required if using c_v as coherence metric
with open("../Data/trainTokens.pkl", "rb") as f:
    trainTokens = pickle.load(f)
    f.close()

# load dictionary
trainDict = corpora.Dictionary.load("../Data/trainDict.dict")

# load corpus
trainCorpus = corpora.MmCorpus('../Data/trainCorpus.mm')

# run LDA model using vectorized corpus
trainModel = gensim.models.LdaMulticore(trainCorpus, num_topics = 14, id2word = trainDict, chunksize = 70, passes = 5, workers = 6)
# save model
trainModel.save('../Data/trainModel.model')
# load model
trainModel = gensim.models.LdaMulticore.load('../Data/trainModel.model')

# show topics and corresponding weights of words in topics
for idx, topic in trainModel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# performance evaluation by classifying sample document using LDA BoW model
# trainTokens[1225]

for index, score in sorted(trainModel[trainTokens[1225]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, trainModel.print_topic(index, 50)))

# number of ways to assess model performance

# compute perplexity: a measure of how good the model is, lower the better
print("\nPerplexity:", trainModel.log_perplexity(trainCorpus))

# compute coherence score
trainCohModLDA = CoherenceModel(model=trainModel, texts=trainTokens, dictionary=trainDict, coherence='u_mass')
# work out coherence
trainCohLDA = trainCohModLDA.get_coherence()
print("\nCoherence Score:", trainCohLDA)

# visualize topics-keywords of LDA
for t in range(trainModel.num_topics):
    f = plt.figure()
    plt.imshow(WordCloud().fit_words(dict(trainModel.show_topic(t, 200))))
    plt.axis('off')
    plt.title('Topic #' + str(t))
    f.savefig("wordCloud" + str(t))