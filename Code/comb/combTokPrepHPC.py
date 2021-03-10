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

# pickle load dfClean
with open("/rds/general/user/yl4220/home/Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# create training corpus (50% of every column)
rat = df.rationale.values.tolist()
hab = df.habitat.values.tolist()
thr = df.threats.values.tolist()
pop = df.population.values.tolist()
ran = df.range.values.tolist()
use = df.useTrade.values.tolist()
con = df.conservationActions.values.tolist()
# combine into training corpus
combCorpus = rat + hab + thr + pop + ran + use + con
del rat,hab,thr,pop,ran,use,con
combCorpus = list(filter(None, combCorpus))

# save combCorpus
with open("/rds/general/user/yl4220/home/Data/comb/combCorpus.pkl", "wb") as f:
    pickle.dump(combCorpus, f)
    f.close()

nlp = spacy.load('en_core_web_sm') # load small sized english vocab

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"country", "río", "rio", "mexico", "australia", "eu", "europe", "european", "africa", "african", "madagascar", "atlantic", "brazil", "brazilian", "america", "american", "united", "states", "usa", "ecuador", "columbia", "papua", "guinea", "china", "spain", "spanish", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "indonesian", "viet", "nam", "vietnam", "italy", "france", "french", "guiana", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "austalian", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "amazonian", "madeira", "canary", "chile", "iberian", "taiwan", "zambia", "tunisia", "korea", "norway", "israel", "egypt", "slovakia", "leone", "caribbean", "zimbabwe", "cyprus", "sicily", "honduras", "fiji", "british", "bahamas", "namibia", "georgia", "puerto", "rico", "verde", "bulgaria", "alaska", "grande", "terre", "adriatic", "asian", "caucasus", "senegal", "scandinavia", "somalia", "finland", "azores", "alps", "corsica", "santa", "cruz", "bangladesh", "croatia", "albania", "andes", "sulawesi", "balkan", "poland", "ukraine", "hawaii", "hawaiian", "sweden", "slovenia", "tasmania", "andean", "chinese", "kwazulu", "natal", "java", "helena", "australian", "australia", "baja", "guatemala", "nicaragua", "ghana", "liberia", "hong", "kong", "janeiro", "palawan", "pakistan", "veracruz", "bahia", "arabian", "sudan", "paraguay", "yunnan", "singapore", "oaxaca", "uk", "england", "lebannon", "cambodia", "sardinia", "britain", "mekong", "côte", "ivoire", "bermuda", "cerro", "belize", "uruguay", "lebanon", "aegean", "syria", "galápago", "seychelle"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

# test if stop words added (need added but not returning false)
# print([t.is_stop for t in nlp("río")])
# can't do entire corpus at once, do columns separately before combining
# generators help decrease ram usage (keep output of nlp.pipe as a generator)
# don't need too many cores since tokenization cannot be multithreaded
combCorpusGen = nlp.pipe(combCorpus, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

combTokens = []
for doc in combCorpusGen:
    combTokens.append(' '.join([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP']))

combCorpusGen2 = nlp.pipe(combTokens, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

combTokens2 = []
for doc in combCorpusGen2:
    combTokens2.append([(tok.lemma_) for tok in doc if not tok.is_stop])

bigram = Phrases(combTokens2, min_count = 9, threshold = 125)
bigramMod = Phraser(bigram)
# print(bigramMod[combTokens2[1]])

combTokens3 = bigramMod[combTokens2]

# pickle combTokens3
with open("/rds/general/user/yl4220/home/Data/comb/combTokens3.pkl", "wb") as f:
    pickle.dump(combTokens3, f)
    f.close()

# create a dictionary
combDict = corpora.Dictionary(combTokens3)

# filter tokens
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
combDict.filter_extremes(no_below=349, no_above=0.45)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)
combDict.compactify() # assign new word ids to all words, shrinking any gaps

# save dictionary
combDict.save("/rds/general/user/yl4220/home/Data/combined/combDict.dict")

# # look at dictionary items
# count = 0
# for k, v in combDict.iteritems():
#     print(k, v)
#     count += 1
#     if count > 50:
#         break

# bag of words corpus
combBoWCorpus = [combDict.doc2bow(doc) for doc in combTokens3]
# save corpus and serializing decreases ram usage (by a lot)
corpora.MmCorpus.serialize("/rds/general/user/yl4220/home/Data/comb/combBoWCorpus.mm", combBoWCorpus)
