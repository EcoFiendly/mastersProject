#!/usr/bin/env python3

# load files
import pickle

# tokenization
import spacy

# create dict and vectorized corpus
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

# pickle load dfClean
with open("/rds/general/user/yl4220/home/Data/dfClean.pkl", "rb") as f:
    df = pickle.load(f)
    f.close()

# create conservation actions corpus
thrtCorpus = df.threats.values.tolist()
del df
thrtCorpus = list(filter(None, thrtCorpus))

# save thrtCorpus
with open("/rds/general/user/yl4220/home/Data/thrt/thrtCorpus.pkl", "wb") as f:
    pickle.dump(thrtCorpus, f)
    f.close()

nlp = spacy.load('en_core_web_sm')

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"country", "río", "rio", "mexico", "australia", "eu", "europe", "european", "africa", "african", "madagascar", "atlantic", "brazil", "brazilian", "america", "american", "united", "states", "usa", "ecuador", "columbia", "papua", "guinea", "china", "spain", "spanish", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "indonesian", "viet", "nam", "vietnam", "italy", "france", "french", "guiana", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "amazonian", "madeira", "canary", "chile", "iberian", "taiwan", "zambia", "tunisia", "korea", "norway", "israel", "egypt", "slovakia", "leone", "caribbean", "zimbabwe", "cyprus", "sicily", "honduras", "fiji", "british", "bahamas", "namibia", "georgia", "puerto", "rico", "verde", "bulgaria", "alaska", "grande", "terre", "adriatic", "asian", "caucasus", "senegal", "scandinavia", "somalia", "finland", "azores", "alps", "corsica", "santa", "cruz", "bangladesh", "croatia", "albania", "andes", "sulawesi", "balkan", "poland", "ukraine", "hawaii", "hawaiian", "sweden", "slovenia", "tasmania", "andean", "chinese", "kwazulu", "natal", "java", "helena", "australian", "australia", "baja", "guatemala", "nicaragua", "ghana", "liberia", "hong", "kong", "janeiro", "palawan", "pakistan", "veracruz", "bahia", "arabian", "sudan", "paraguay", "yunnan", "singapore", "oaxaca", "uk", "england", "lebannon", "cambodia", "sardinia", "britain", "mekong", "côte", "ivoire", "bermuda", "cerro", "belize", "uruguay", "lebanon", "aegean", "syria", "galápago", "seychelle"} | {"situ", "appendix", "cite", "annex", "need", "book", "find"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

thrtCorpusGen = nlp.pipe(thrtCorpus, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

thrtTokens = []
for doc in thrtCorpusGen:
    thrtTokens.append(' '.join([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP']))

thrtCorpusGen2 = nlp.pipe(thrtTokens, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

thrtTokens2 = []
for doc in thrtCorpusGen2:
    thrtTokens2.append([(tok.lemma_) for tok in doc if not tok.is_stop])

# build bigram model
bigram = Phrases(thrtTokens2, min_count = 9, threshold = 125)
bigramMod = Phraser(bigram)
# print(bigramMod[thrtTokens2[0]])
# # list of bigrams obtained, with their corresponding scores
# bigramsList = []
# for doc in bigram.export_phrases(thrtTokens2):
#     bigramsList.append(doc)

# bigramsList[0:100]

thrtTokens3 = bigramMod[thrtTokens2]

# pickle trainTokens3
with open("/rds/general/user/yl4220/home/Data/thrt/thrtTokens3.pkl", "wb") as f:
    pickle.dump(thrtTokens3, f)
    f.close()

# create dictionary
thrtDict = corpora.Dictionary(thrtTokens3)

thrtDict.filter_extremes(no_below=100, no_above=0.45)

thrtDict.compactify()

thrtDict.save("/rds/general/user/yl4220/home/Data/thrt/thrtDict.dict")

thrtBoWCorpus = [thrtDict.doc2bow(doc) for doc in thrtTokens3]

corpora.MmCorpus.serialize("/rds/general/user/yl4220/home/Data/thrt/thrtBoWCorpus.mm", thrtBoWCorpus)