#!/usr/bin/env python3

# load files
import pickle

# spacy for lemmatization
# spacy easier to use compared to nltk WordNetLemmatizer
import spacy

# creating dictionary and vectorized corpus
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

# load trainCorpus
with open("../Data/trainCorpus.pkl", "rb") as f:
    trainCorpus = pickle.load(f)
    f.close()

nlp = spacy.load('en_core_web_sm') # load small sized english vocab

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "south", "southern", "east", "eastern", "west", "western", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "río", "mexico", "australia", "eu", "europe", "european", "africa", "madagascar", "atlantic", "brazil", "america", "american", "united", "states", "ecuador", "columbia", "papua", "guinea", "china", "spain", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "viet", "nam", "italy", "france", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "austalian", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "madeira", "canary", "chile", "specie", "species", "subspecie", "assess", "assessment", "find", "occur", "occurrence", "record", "know", "area", "unknown", "know", "known", "year", "population", "available", "protect", "protection", "conservation", "distribution", "need", "major", "collect", "collection", "survey", "red", "list", "listing", "need", "criteria", "s", "asl", "bcgi", "annex", "currently", "ex", "situ", "require", "include", "report", "extent", "location", "locality", "subpopulation", "habitat", "site", "place", "information", "research", "trend", "action", "number", "measure", "range", "new", "concern", "measure", "specific", "estimate", "individual", "datum", "national", "park", "management", "status", "likely", "province", "consider", "region", "cause", "result", "usually", "near", "remain", "monitor", "taxon", "overall", "suspect", "affect", "threat", "threaten", "state", "global", "establish", "threshold", "important", "plan", "km²", "suggest", "recently", "describe", "quantify", "de", "appendix", "ii", "monitoring", "appear"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

# test if stop words added (need added but not returning false)
print([t.is_stop for t in nlp("río")])
# can't do entire corpus at once, do columns separately before combining
# generators help decrease ram usage (keep output of nlp.pipe as a generator)
# don't need too many cores since tokenization cannot be multithreaded
trainCorpusGen = nlp.pipe(trainCorpus, n_process = 6, batch_size = 480, disable = ["parser", "ner"])

# before making bigrams (probably useless now)
trainTokens = []
for doc in trainCorpusGen:
    trainTokens.append(' '.join([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP']))

with open("../Data/trainTokens.pkl", "wb") as f:
    pickle.dump(trainTokens, f)
    f.close()

trainCorpusGen2 = nlp.pipe(trainTokens, n_process = 6, batch_size = 480, disable = ["parser", "ner"])

trainTokens2 = []
for doc in trainCorpusGen2:
    trainTokens2.append([(tok.lemma_) for tok in doc if not tok.is_stop])

# pickle trainTokens
with open("../Data/trainTokens2.pkl", "wb") as f:
    pickle.dump(trainTokens2, f)
    f.close()

bigram = Phrases(trainTokens2, min_count = 5, threshold = 10)
bigramMod = Phraser(bigram)
print(bigramMod[trainTokens2[1]])

trainTokens3 = bigramMod[trainTokens2]

# create a dictionary
trainDict = corpora.Dictionary(trainTokens3)

# filter tokens
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
trainDict.filter_extremes(no_below=140, no_above=0.8)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)
trainDict.compactify() # assign new word ids to all words, shrinking any gaps

# save dictionary
trainDict.save("../Data/trainDict.dict")

# look at dictionary items
count = 0
for k, v in trainDict.iteritems():
    print(k, v)
    count += 1
    if count > 50:
        break

# bag of words corpus
trainBoWCorpus = [trainDict.doc2bow(doc) for doc in trainTokens3]
# save corpus and serializing decreases ram usage (by a lot)
corpora.MmCorpus.serialize('../Data/trainBoWCorpus.mm', trainBoWCorpus)
