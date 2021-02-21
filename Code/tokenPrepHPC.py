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
with open("/rds/general/user/yl4220/home/Data/trainCorpus.pkl", "rb") as f:
    trainCorpus = pickle.load(f)
    f.close()

nlp = spacy.load('en_core_web_sm') # load small sized english vocab

# add stopwords to spacy
nlp.Defaults.stop_words |= {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "" "west", "western", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"} | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"} | {"country", "río", "rio", "mexico", "australia", "eu", "europe", "european", "africa", "african", "madagascar", "atlantic", "brazil", "brazillian", "america", "american", "united", "states", "usa", "ecuador", "columbia", "papua", "guinea", "china", "spain", "spanish", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "indonesian", "viet", "nam", "vietnam", "italy", "france", "french", "guiana", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "austalian", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "madeira", "canary", "chile", "iberian", "taiwan", "zambia", "tunisia", "korea", "norway", "israel", "egypt", "slovakia", "leone", "caribbean", "zimbabwe", "cyprus", "sicily", "honduras", "fiji", "british", "bahamas", "namibia", "georgia", "puerto", "rico", "verde", "bulgaria", "alaska", "grande", "terre", "adriatic", "asian", "caucasus", "senegal", "scandinavia", "somalia", "finland", "azores", "alps", "corsica", "santa", "cruz", "bangladesh", "croatia", "albania", "andes", "sulawesi", "balkan", "poland", "ukraine", "hawaii", "hawaiian", "sweden", "slovenia", "tasmania", "andean", "chinese", "kwazulu", "natal", "java", "helena", "austrailian", "australia", "baja", "guatemala", "nicaragua", "ghana", "liberia", "hong", "kong", "janeiro", "palawan", "pakistan", "veracruz", "bahia", "arabian", "sudan", "paraguay", "yunnan", "singapore", "oaxaca", "uk", "england", "lebannon", "cambodia"} | {"habitat", "coastal", "coast", "reef", "river", "island", "lake", "sea", "marine", "stream", "water", "forest", "mountain", "montane", "tree", "plant", "elevation", "altitude", "grassland", "highland", "lowland", "woodland", "tropical", "basin", "rocky", "bank", "hill", "cave", "valley", "rock", "mainland", "herbarium", "plantation", "freshwater", "meadow", "muddy", "mangrove", "pool", "lagoon", "rainforest", "rainfor", "steppe", "ravine", "cliff", "offshore", "desert", "waterfall", "shrubland", "stony", "coastline", "mountainous", "thicket", "canyon", "marsh", "wooded", "savanna", "savannah", "alpine", "submontane"} | {"specie", "species", "subspecie", "assess", "assessment", "find", "occur", "occurrence", "record", "know", "area", "unknown", "know", "known", "population", "available", "protect", "protection", "conservation", "distribution", "need", "major", "collect", "collection", "survey", "red", "list", "listing", "need", "criteria", "s", "asl", "bcgi", "annex", "current", "currently", "ex", "situ", "require", "include", "report", "extent", "location", "locality", "subpopulation", "habitat", "site", "place", "information", "research", "trend", "action", "number", "measure", "range", "new", "concern", "measure", "specific", "estimate", "individual", "datum", "national", "park", "management", "status", "likely", "province", "consider", "region", "cause", "result", "usually", "near", "remain", "monitor", "taxon", "overall", "suspect", "affect", "threat", "threaten", "state", "global", "establish", "threshold", "important", "plan", "km²", "suggest", "recently", "describe", "quantify", "de", "appendix", "ii", "monitoring", "appear", "recommend", "increase", "decrease", "specimen", "speciman", "relatively", "wide", "widely", "criterion", "approach", "distribute", "infer", "probably", "recent", "present", "fairly", "common", "locally", "main", "evidence", "significant", "study", "genus", "peninsula", "cite", "potential", "taxonomy", "taxonomic", "lack", "legislation", "necessary", "critically", "endanger", "vulnerable", "ongoing", "extinct", "least", "presumably", "specifically", "possibly", "type", "little", "think", "addition", "additional", "confirm", "programme", "outside", "rarely", "associate", "mt", "mount", "awareness", "sub", "direct", "variety", "abundant", "future", "mainly", "et", "al", "unlikely", "believe", "peninsular", "follow", "la", "different", "adjacent", "reason", "evaluate", "discover", "document", "aoo", "eoo", "particular", "avoid", "benefit", "requirement", "beneficial", "generally", "refer", "continue", "class", "view", "seven", "classify", "category", "department", "numerous", "democratic", "entire", "approximately", "occupancy", "exist", "apparently", "indicate", "possible", "define", "determine", "district", "restrict", "previously", "occassionally", "typically", "reach", "total", "importance", "substantial", "situate", "regionally", "particularly", "come", "quickly", "preferred", "constitute", "estimated", "calculate", "regime", "able", "availability", "furthermore", "historically", "closely", "note", "difficult", "prove", "immediate", "run", "subject", "return", "etc", "annually", "regularly", "day", "surround", "primarily", "shelter", "bed", "interest", "natura", "reportedly", "partly", "mention", "visit", "white", "apart", "involve", "urgently", "exception", "somewhat", "nearly", "pdr", "especially", "key", "undergo", "gallery", "sp", "spp", "easily", "whilst", "nacional", "kingdom", "similarly", "like", "clarify", "sl", "gt", "considerably", "ha", "lt", "alongside", "million", "hactare", "meter", "km", "ago", "c", "del", "metre", "el", "taxa", "subsp", "author"}

for word in nlp.Defaults.stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

# test if stop words added (need added but not returning false)
# print([t.is_stop for t in nlp("río")])
# can't do entire corpus at once, do columns separately before combining
# generators help decrease ram usage (keep output of nlp.pipe as a generator)
# don't need too many cores since tokenization cannot be multithreaded
trainCorpusGen = nlp.pipe(trainCorpus, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

# before making bigrams (probably useless now)
trainTokens = []
for doc in trainCorpusGen:
    trainTokens.append(' '.join([(tok.lemma_) for tok in doc if not tok.is_stop and not tok.is_punct and tok.tag_ != 'NNP']))

trainCorpusGen2 = nlp.pipe(trainTokens, n_process = 8, batch_size = 800, disable = ["parser", "ner"])

trainTokens2 = []
for doc in trainCorpusGen2:
    trainTokens2.append([(tok.lemma_) for tok in doc if not tok.is_stop])

bigram = Phrases(trainTokens2, min_count = 5, threshold = 10)
bigramMod = Phraser(bigram)
# print(bigramMod[trainTokens2[1]])

trainTokens3 = bigramMod[trainTokens2]

# pickle trainTokens3
with open("/rds/general/user/yl4220/home/Data/trainTokens3.pkl", "wb") as f:
    pickle.dump(trainTokens2, f)
    f.close()

# create a dictionary
trainDict = corpora.Dictionary(trainTokens3)

# filter tokens
# no_below(int): keep tokens which are contained in at least int documents
# no_above(float): keep tokens which are contained in no more than float documents (fraction of total corpus size, not an absolute number)
# keep_n(int): keep only the first int most frequent tokens, keep all if None
trainDict.filter_extremes(no_below=349, no_above=0.8)
# this example:
# removes tokens in dictionary that appear in less than 45 (0.1%) sample documents
# removes tokens in dictionary that appear in more than 0.8 of total corpus size
# after the above 2, keep all of the tokens (or keep_n = int)
trainDict.compactify() # assign new word ids to all words, shrinking any gaps

# save dictionary
trainDict.save("/rds/general/user/yl4220/home/Data/trainDict.dict")

# # look at dictionary items
# count = 0
# for k, v in trainDict.iteritems():
#     print(k, v)
#     count += 1
#     if count > 50:
#         break

# bag of words corpus
trainBoWCorpus = [trainDict.doc2bow(doc) for doc in trainTokens3]
# save corpus and serializing decreases ram usage (by a lot)
corpora.MmCorpus.serialize('/rds/general/user/yl4220/home/Data/trainBoWCorpus.mm', trainBoWCorpus)
