## thought process
don't combine columns first, start with cleaning before applying lda so the individual columns can be distinguished

Two options: start with option 1 then try 2 if possible
1) train 1 model and apply to every column
  - create 1 dtm, with every single word from all the columns of focus

2) train 1 model for each column
  - create 1 dtm for every column

current dataframe
      |  rationale      |  habitat     |  threats     |  etc
doc1  |  doc1rationale  |  doc1habitat |  doc1threats |
doc2  |  doc2rationale  |  doc2habitat |  doc2threats |
etc

doc1rationale is the tokenized list of words

document term matrix
                | word1 | word2 | word3 | etc
  doc1rationale |   1   |   0   |   1   |
  doc1habitat   |   0   |   1   |   2   |
  doc1threats   |   0   |   4   |   1   |
  doc2rationale |   1   |   2   |   0   |
  etc

- assembling list of topic general words (contextually non informative)

  Confirm:

  directions = {"north", "northern", "northward", "south", "southern", "southward", "east", "eastern", "eastward", "west", "western", "westward", "northeast", "northeastern", "northwest", "northwestern", "southeast", "southeastern", "southwest", "southwestern", "centre", "central", "center", "upper", "lower", "high", "low"}

  letters = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}

  time = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"}

  geographic names = {"country", "río", "rio", "mexico", "australia", "eu", "europe", "european", "africa", "african", "madagascar", "atlantic", "brazil", "brazillian", "america", "american", "united", "states", "usa", "ecuador", "columbia", "papua", "guinea", "china", "spain", "spanish", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "indonesian", "viet", "nam", "vietnam", "italy", "france", "french", "guiana", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "austalian", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "amazonian", "madeira", "canary", "chile", "iberian", "taiwan", "zambia", "tunisia", "korea", "norway", "israel", "egypt", "slovakia", "leone", "caribbean", "zimbabwe", "cyprus", "sicily", "honduras", "fiji", "british", "bahamas", "namibia", "georgia", "puerto", "rico", "verde", "bulgaria", "alaska", "grande", "terre", "adriatic", "asian", "caucasus", "senegal", "scandinavia", "somalia", "finland", "azores", "alps", "corsica", "santa", "cruz", "bangladesh", "croatia", "albania", "andes", "sulawesi", "balkan", "poland", "ukraine", "hawaii", "hawaiian", "sweden", "slovenia", "tasmania", "andean", "chinese", "kwazulu", "natal", "java", "helena", "australian", "australia", "baja", "guatemala", "nicaragua", "ghana", "liberia", "hong", "kong", "janeiro", "palawan", "pakistan", "veracruz", "bahia", "arabian", "sudan", "paraguay", "yunnan", "singapore", "oaxaca", "uk", "england", "lebannon", "cambodia", "sardinia", "britain", "mekong", "côte", "ivoire", "bermuda", "cerro", "belize", "uruguay", "lebanon"}

  habitat = {"habitat", "coastal", "coast", "reef", "river", "island", "lake", "sea", "marine", "stream", "water", "forest", "mountain", "montane", "tree", "plant", "elevation", "altitude", "grassland", "highland", "lowland", "woodland", "tropical", "basin", "rocky", "bank", "hill", "cave", "valley", "rock", "mainland", "herbarium", "plantation", "freshwater", "meadow", "muddy", "mangrove", "pool", "lagoon", "rainforest", "rainfor", "steppe", "ravine", "cliff", "offshore", "desert", "waterfall", "shrubland", "stony", "coastline", "mountainous", "thicket", "canyon", "marsh", "wooded", "savanna", "savannah", "alpine", "subalpine", "submontane", "subtropical"}

  habitat iucn list = {"habitat", "woodland", "boreal", "subarctic", "subantarctic", "temperate", "subtropical", "tropical", "lowland", "mangrove", "swamp", "montane", "savanna", "savannah", "shrubland", "shrubby", "grassland", "tundra", "wetland", "inland", "river", "stream", "creek", "bog", "marsh", "fen", "peatland", "lake", "oases", "oasis", "alpine", "geothermal", "delta", "saline", "brackish", "alkaline", "flat", "pool", "karst", "subterranean", "aquatic", "rock", "rocky", "cave", "desert", "marine", "neritic", "pelagic", "subtidal", "reef", "pebble", "gravel", "sand", "sandy", "mud", "muddy", "macroalgal", "kelp", "substrate", "foreslope", "lagoon", "rubble", "seagrass", "submerge", "estuary", "estuaries", "oceanic", "epipelagic", "mesopelagic", "bathypelagic", "abyssopelagic", "zone", "ocean", "benthic", "demersal", "bathyl", "abyssal", "plain", "mountain", "hill", "hadal", "trench", "seamount", "vent", "rift", "seep", "intertidal", "shoreline", "beach", "sandbar", "spit", "shingle", "tidepool", "coastal", "supratidal", "dune", "terrestrial", "arable", "pastureland", "aquaculture"}

  others = {"specie", "species", "subspecie", "assess", "assessment", "find", "occur", "occurrence", "record", "know", "area", "unknown", "know", "known", "population", "available", "protect", "protection", "conservation", "distribution", "need", "major", "collect", "collection", "survey", "red", "list", "listing", "need", "criteria", "asl", "bcgi", "annex", "current", "currently", "ex", "situ", "require", "include", "report", "extent", "location", "locality", "subpopulation", "habitat", "site", "place", "information", "research", "trend", "action", "number", "measure", "range", "new", "concern", "measure", "specific", "estimate", "individual", "datum", "national", "park", "management", "status", "likely", "province", "consider", "region", "cause", "result", "usually", "near", "remain", "monitor", "taxon", "overall", "suspect", "affect", "threat", "threaten", "state", "global", "establish", "threshold", "important", "plan", "km²", "suggest", "recently", "describe", "quantify", "de", "appendix", "ii", "monitoring", "appear", "recommend", "increase", "decrease", "specimen", "speciman", "relatively", "wide", "widely", "criterion", "approach", "distribute", "infer", "probably", "recent", "present", "fairly", "common", "locally", "main", "evidence", "significant", "study", "genus", "peninsula", "cite", "potential", "taxonomy", "taxonomic", "lack", "legislation", "necessary", "critically", "endanger", "vulnerable", "ongoing", "extinct", "least", "presumably", "specifically", "possibly", "type", "little", "think", "addition", "additional", "confirm", "programme", "outside", "rarely", "associate", "mt", "mount", "awareness", "sub", "direct", "variety", "abundant", "future", "mainly", "et", "al", "unlikely", "believe", "peninsular", "follow", "la", "different", "adjacent", "reason", "evaluate", "discover", "document", "aoo", "eoo", "particular", "avoid", "benefit", "requirement", "beneficial", "generally", "refer", "continue", "class", "view", "seven", "classify", "category", "department", "numerous", "democratic", "entire", "approximately", "occupancy", "exist", "apparently", "indicate", "possible", "define", "determine", "district", "restrict", "previously", "occassionally", "typically", "reach", "total", "importance", "substantial", "situate", "regionally", "particularly", "come", "quickly", "preferred", "constitute", "estimated", "calculate", "regime", "able", "availability", "furthermore", "historically", "closely", "note", "difficult", "prove", "immediate", "run", "subject", "return", "etc", "annually", "regularly", "day", "surround", "primarily", "shelter", "bed", "interest", "natura", "reportedly", "partly", "mention", "visit", "white", "apart", "involve", "urgently", "exception", "somewhat", "nearly", "pdr", "especially", "key", "undergo", "gallery", "sp", "spp", "easily", "whilst", "nacional", "kingdom", "similarly", "like", "clarify", "sl", "gt", "considerably", "ha", "lt", "alongside", "million", "hectare", "meter", "km", "ago", "del", "metre", "el", "taxa", "subsp", "author", "localitie", "prior", "enter", "verify", "subsequently", "slightly", "percentage", "additionally", "occasional", "supplementary", "material", "municipality", "qualifie"}


    




1) Training model on conservationActions.
    Start with stopwords of only directions, time and geographic names
    Bigram min count 9, threshold 125
    Dictionary.filter_extremes(no_below=100, no_above=0.75)
    Train model from 2 to 200 topics
    Compare umass and c_v (c_v seem to work because training with full dataset)

    results:
    umass peaks at 2, cv peaks at 14
    Many words repeat across topics

2) Changed filter_extremes to no_above=0.6 for conAct model

    results:
    umass peaks at 2, cv peaks at 21

3) Changed filter_extremes to no_above=0.5 for both comb, conAct and hab, training with full corpus.

    results:
    comb peaks at 4 5* 7 11 (asterisk is max)

4) solved problem from c_v metric

    results:
    hab peaks at 13*, 15, 18

    comb peaks at 10, 14*, 24

    conAct peaks at 7, 9*, 39

5) changed filter_extremes to no_above=0.45









