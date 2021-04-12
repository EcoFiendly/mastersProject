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

  time = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "day", "night", "week", "month", "year"}

  geographic names = {"country", "río", "rio", "mexico", "australia", "eu", "europe", "european", "africa", "african", "madagascar", "atlantic", "brazil", "brazillian", "america", "american", "united", "states", "usa", "ecuador", "columbia", "papua", "guinea", "china", "spain", "spanish", "queensland", "india", "thailand", "mediterranean", "republic", "indonesia", "indonesian", "viet", "nam", "vietnam", "italy", "france", "french", "guiana", "pacific", "malaysia", "asia", "zealand", "caledonia", "congo", "florida", "cape", "san", "california", "peru", "colombia", "sierra", "argentina", "cameroon", "mozambique", "parque", "iran", "netherland", "persian", "germany", "myanmar", "costa", "rica", "panama", "tanzania", "venezuela", "ireland", "bolivia", "japan", "solomon", "nigeria", "gabon", "turkey", "borneo", "canada", "kenya", "wale", "indian", "indo", "philippine", "sumatra", "bay", "mexican", "texas", "carolina", "cuba", "algeria", "ethiopia", "austalian", "gulf", "sri", "lanka", "sarawak", "portugal", "greece", "russia", "sabah", "nepal", "kalimantan", "switzerland", "malawi", "austria", "morocco", "nord", "uganda", "czech", "hungary", "romania", "angola", "los", "angeles", "cordillera", "victoria", "lao", "amazon", "amazonian", "madeira", "canary", "chile", "iberian", "taiwan", "zambia", "tunisia", "korea", "norway", "israel", "egypt", "slovakia", "leone", "caribbean", "zimbabwe", "cyprus", "sicily", "honduras", "fiji", "british", "bahamas", "namibia", "georgia", "puerto", "rico", "verde", "bulgaria", "alaska", "grande", "terre", "adriatic", "asian", "caucasus", "senegal", "scandinavia", "somalia", "finland", "azores", "alps", "corsica", "santa", "cruz", "bangladesh", "croatia", "albania", "andes", "sulawesi", "balkan", "poland", "ukraine", "hawaii", "hawaiian", "sweden", "slovenia", "tasmania", "andean", "chinese", "kwazulu", "natal", "java", "helena", "australian", "australia", "baja", "guatemala", "nicaragua", "ghana", "liberia", "hong", "kong", "janeiro", "palawan", "pakistan", "veracruz", "bahia", "arabian", "sudan", "paraguay", "yunnan", "singapore", "oaxaca", "uk", "england", "lebannon", "cambodia", "sardinia", "britain", "mekong", "côte", "ivoire", "bermuda", "cerro", "belize", "uruguay", "lebanon", "aegean", "syria", "galápago", "seychelle"}

  habitat = {"habitat", "coastal", "coast", "reef", "river", "island", "lake", "sea", "marine", "stream", "water", "forest", "mountain", "montane", "tree", "plant", "elevation", "altitude", "grassland", "highland", "lowland", "woodland", "tropical", "basin", "rocky", "bank", "hill", "cave", "valley", "rock", "mainland", "herbarium", "plantation", "freshwater", "meadow", "muddy", "mangrove", "pool", "lagoon", "rainforest", "rainfor", "steppe", "ravine", "cliff", "offshore", "desert", "waterfall", "shrubland", "stony", "coastline", "mountainous", "thicket", "canyon", "marsh", "wooded", "savanna", "savannah", "alpine", "subalpine", "submontane", "subtropical"}

  habitat iucn list = {"habitat", "woodland", "boreal", "subarctic", "subantarctic", "temperate", "subtropical", "tropical", "lowland", "mangrove", "swamp", "montane", "savanna", "savannah", "shrubland", "shrubby", "grassland", "tundra", "wetland", "inland", "river", "stream", "creek", "bog", "marsh", "fen", "peatland", "lake", "oases", "oasis", "alpine", "geothermal", "delta", "saline", "brackish", "alkaline", "flat", "pool", "karst", "subterranean", "aquatic", "rock", "rocky", "cave", "desert", "marine", "neritic", "pelagic", "subtidal", "reef", "pebble", "gravel", "sand", "sandy", "mud", "muddy", "macroalgal", "kelp", "substrate", "foreslope", "lagoon", "rubble", "seagrass", "submerge", "estuary", "estuaries", "oceanic", "epipelagic", "mesopelagic", "bathypelagic", "abyssopelagic", "zone", "ocean", "benthic", "demersal", "bathyl", "abyssal", "plain", "mountain", "hill", "hadal", "trench", "seamount", "vent", "rift", "seep", "intertidal", "shoreline", "beach", "sandbar", "spit", "shingle", "tidepool", "coastal", "supratidal", "dune", "terrestrial", "arable", "pastureland", "aquaculture"}

  others = {"situ", "appendix", "cite", "annex"}


    




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

5) changed filter_extremes to no_above=0.45, other params: chunksize=8000, eval_every=None, passes=15

6) changed chunksize and eval_every to 5% of corpus, increased passes to 20, set model seed to 95

7) changed eval_every to 4*chunksize, added {"situ", "appendix", "cite", "annex", "need", "book", "find"} to stopwords

    Combined peaks at 25

    Topic: assess extinction status \
    Words: 0.084*"endanger" + 0.063*"list" + 0.057*"assess" + 0.049*"threaten" + 0.040*"vulnerable" + 0.031*"critically" + 0.028*"near" + 0.025*"habitat" + 0.023*"assessment" + 0.022*"location"

    Topic: threat to habitat (human activity) \
    Words: 0.045*"habitat" + 0.040*"threat" + 0.020*"fire" + 0.018*"change" + 0.017*"area" + 0.016*"activity" + 0.016*"human" + 0.014*"impact" + 0.014*"land" + 0.012*"cause"

    Topic: use/trade \
    Words: 0.223*"trade" + 0.036*"know" + 0.031*"aquarium" + 0.031*"harvest" + 0.028*"utilise" + 0.028*"wild" + 0.027*"collect" + 0.023*"ornamental" + 0.019*"market" + 0.016*"international"

    Topic: research (conservation actions) \
    Words: 0.109*"conservation" + 0.064*"measure" + 0.055*"place" + 0.053*"protect" + 0.053*"area" + 0.048*"research" + 0.044*"specific" + 0.038*"action" + 0.036*"population" + 0.034*"distribution"

    Topic: monitor (conservation actions) \
    Words: 0.041*"conservation" + 0.027*"population" + 0.024*"management" + 0.023*"habitat" + 0.021*"action" + 0.018*"protection" + 0.017*"protect" + 0.014*"monitor" + 0.014*"site" + 0.012*"include"

    Topic: threat to forests \
    Words: 0.029*"area" + 0.022*"hunt" + 0.020*"timber" + 0.015*"construction" + 0.015*"forest" + 0.014*"local" + 0.013*"increase" + 0.013*"log" + 0.012*"human" + 0.012*"wood"

    Topic: population information \
    Words: 0.157*"population" + 0.145*"information" + 0.098*"available" + 0.065*"trend" + 0.053*"size" + 0.040*"datum" + 0.023*"unknown" + 0.023*"subpopulation" + 0.021*"decline" + 0.016*"current"

    Topic: population information \
    Words: 0.126*"population" + 0.093*"criterion" + 0.087*"threshold" + 0.087*"size" + 0.083*"vulnerable" + 0.074*"approach" + 0.060*"range" + 0.060*"trend" + 0.050*"believe" + 0.032*"large"

    Topic: fish use trade \
    Words: 0.057*"fishery" + 0.042*"catch" + 0.025*"fishing" + 0.020*"fish" + 0.016*"trawl" + 0.016*"commercial" + 0.013*"bycatch" + 0.011*"increase" + 0.011*"target" + 0.011*"stock"

    Topic: population information \
    Words: 0.103*"list" + 0.064*"concern" + 0.061*"decline" + 0.052*"distribution" + 0.049*"threaten" + 0.045*"population" + 0.045*"category" + 0.044*"large" + 0.041*"wide" + 0.032*"qualify"

    Topic: population information \
    Words: 0.135*"common" + 0.076*"widespread" + 0.063*"population" + 0.053*"concern" + 0.050*"abundant" + 0.043*"range" + 0.041*"locally" + 0.029*"global" + 0.025*"stable" + 0.023*"size"

    Topic: population information \
    Words: 0.128*"individual" + 0.104*"population" + 0.071*"estimate" + 0.053*"number" + 0.045*"decline" + 0.044*"mature" + 0.021*"total" + 0.019*"density" + 0.018*"range" + 0.015*"survey"

    Topic: forests; lowland/montane \
    Words: 0.166*"forest" + 0.034*"tree" + 0.025*"lowland" + 0.016*"montane" + 0.015*"fruit" + 0.014*"secondary" + 0.012*"leave" + 0.012*"plantation" + 0.012*"primary" + 0.012*"habitat"

    Topic: threat level in space and time \
    Words: 0.207*"threat" + 0.111*"major" + 0.067*"concern" + 0.066*"know" + 0.056*"assess" + 0.041*"currently" + 0.040*"significant" + 0.035*"distribution" + 0.032*"future" + 0.031*"identify"

    Topic: monitoring \
    Words: 0.081*"know" + 0.050*"locality" + 0.049*"record" + 0.038*"collection" + 0.038*"speciman" + 0.031*"survey" + 0.031*"collect" + 0.024*"area" + 0.022*"site" + 0.019*"type"

    Topic: freshwater habitat \
    Words: 0.044*"water" + 0.026*"stream" + 0.021*"river" + 0.018*"small" + 0.018*"occur" + 0.015*"habitat" + 0.015*"fish" + 0.015*"lake" + 0.014*"inhabit" + 0.014*"feed"

    Topic: island bird ecology \
    Words: 0.173*"island" + 0.032*"nest" + 0.027*"breeding" + 0.025*"bird" + 0.025*"breed" + 0.017*"colony" + 0.016*"site" + 0.014*"introduce" + 0.012*"include" + 0.010*"predator"

    Topic: threat to coral systems \
    Words: 0.039*"threat" + 0.030*"coral" + 0.029*"habitat" + 0.028*"change" + 0.028*"pollution" + 0.027*"water" + 0.024*"increase" + 0.021*"disease" + 0.019*"climate" + 0.017*"impact"

    Topic: freshwater habitat \
    Words: 0.076*"river" + 0.049*"occur" + 0.042*"range" + 0.031*"record" + 0.027*"lake" + 0.025*"region" + 0.024*"basin" + 0.022*"distribution" + 0.021*"know" + 0.015*"subspecie"

    Topic: protect parks and forests (conservation)\
    Words: 0.114*"protect" + 0.104*"area" + 0.086*"national" + 0.081*"park" + 0.065*"reserve" + 0.059*"occur" + 0.022*"forest" + 0.020*"know" + 0.016*"conservation" + 0.016*"collection"

    Topic: research on endemics \
    Words: 0.085*"occurrence" + 0.080*"extent" + 0.075*"area" + 0.069*"endemic" + 0.050*"occupancy" + 0.047*"occur" + 0.037*"estimate" + 0.037*"know" + 0.033*"location" + 0.030*"province"

    Topic: forest habitat decline \
    Words: 0.098*"habitat" + 0.047*"decline" + 0.044*"loss" + 0.039*"forest" + 0.035*"threat" + 0.026*"threaten" + 0.025*"agriculture" + 0.024*"deforestation" + 0.023*"log" + 0.020*"continue"

    Topic: reef habitat/system \
    Words: 0.067*"reef" + 0.055*"depth" + 0.036*"range" + 0.029*"length" + 0.026*"coral" + 0.024*"occur" + 0.024*"island" + 0.023*"ocean" + 0.022*"coast" + 0.017*"generation"

    Topic: forest habitat \
    Words: 0.035*"forest" + 0.031*"grow" + 0.022*"tree" + 0.021*"occur" + 0.018*"habitat" + 0.017*"soil" + 0.016*"area" + 0.014*"grassland" + 0.014*"rock" + 0.014*"open"

    Topic: use/trade \
    Words: 0.064*"utilize" + 0.049*"record" + 0.043*"report" + 0.036*"unknown" + 0.028*"female" + 0.028*"use" + 0.023*"know" + 0.022*"size" + 0.022*"male" + 0.015*"group"

8) trained models for pop, rat, rng, thrt and use

    population peaks at 19

    Topic: specimen collection \

    Words: 0.098*"speciman" + 0.096*"know" + 0.075*"collect" + 0.068*"record" + 0.055*"collection" + 0.044*"locality" + 0.043*"datum" + 0.033*"type" + 0.023*"available" + 0.021*"museum"
    
    Topic: reef decline \
    Words: 0.109*"reef" + 0.065*"decline" + 0.059*"assume" + 0.051*"average" + 0.035*"individual" + 0.034*"generation" + 0.032*"estimate" + 0.030*"coral" + 0.025*"range" + 0.024*"rate"
    
    Topic: forests \
    Words: 0.140*"forest" + 0.034*"tree" + 0.024*"area" + 0.023*"common" + 0.021*"encounter" + 0.019*"record" + 0.019*"habitat" + 0.018*"occur" + 0.014*"lowland" + 0.011*"rarely"
    
    Topic: study island species \
    Words: 0.092*"survey" + 0.048*"island" + 0.043*"individual" + 0.040*"record" + 0.028*"abundance" + 0.021*"conduct" + 0.019*"observe" + 0.016*"density" + 0.015*"sample" + 0.015*"study"
    
    Topic: plant subpopulation \
    Words: 0.204*"subpopulation" + 0.072*"know" + 0.040*"plant" + 0.032*"recent" + 0.032*"small" + 0.032*"fragment" + 0.030*"collection" + 0.023*"severely" + 0.022*"occur" + 0.013*"isolate"
    
    Topic: reduce fishing \
    Words: 0.056*"decline" + 0.033*"fishing" + 0.031*"catch" + 0.029*"datum" + 0.029*"increase" + 0.028*"effort" + 0.024*"reduction" + 0.021*"fishery" + 0.020*"landing" + 0.020*"level"
    
    Topic: generational decline \
    Words: 0.067*"decline" + 0.041*"generation" + 0.035*"estimate" + 0.028*"rate" + 0.027*"report" + 0.025*"size" + 0.024*"reduction" + 0.024*"distribution" + 0.024*"change" + 0.020*"length"
    
    Topic: monitor rare individuals \
    Words: 0.062*"rare" + 0.044*"individual" + 0.036*"locality" + 0.034*"site" + 0.028*"small" + 0.027*"consider" + 0.026*"trend" + 0.025*"know" + 0.020*"record" + 0.018*"stable"
    
    Topic: threat of habitat decline \
    Words: 0.112*"decline" + 0.093*"habitat" + 0.051*"suspect" + 0.045*"loss" + 0.038*"ongoing" + 0.036*"decrease" + 0.024*"quality" + 0.023*"extent" + 0.023*"threat" + 0.022*"infer"
    
    Topic: increase area and density of national parks \
    Words: 0.038*"area" + 0.031*"density" + 0.029*"estimate" + 0.019*"park" + 0.018*"national" + 0.016*"range" + 0.015*"number" + 0.012*"increase" + 0.012*"animal" + 0.011*"individual"
    
    Topic: estimate number of mature individuals \
    Words: 0.229*"individual" + 0.149*"estimate" + 0.119*"mature" + 0.066*"number" + 0.047*"size" + 0.031*"pair" + 0.030*"total" + 0.024*"equate" + 0.022*"global" + 0.018*"large"
    
    Topic: common sizes \
    Words: 0.156*"common" + 0.140*"size" + 0.136*"global" + 0.129*"quantify" + 0.129*"describe" + 0.059*"fairly" + 0.058*"uncommon" + 0.042*"report" + 0.030*"locally" + 0.013*"detailed"
    
    Topic: stability \
    Words: 0.242*"abundant" + 0.173*"common" + 0.134*"locally" + 0.077*"widespread" + 0.050*"range" + 0.046*"stable" + 0.021*"consider" + 0.013*"frequent" + 0.012*"occur" + 0.012*"presume"
    
    Topic: large in number and size unknown \
    Words: 0.066*"large" + 0.057*"number" + 0.049*"size" + 0.041*"relatively" + 0.039*"unknown" + 0.039*"occurrence" + 0.036*"total" + 0.035*"probably" + 0.034*"trend" + 0.030*"represent"
    
    Topic: stable common ranges \
    Words: 0.147*"common" + 0.094*"range" + 0.069*"stable" + 0.053*"consider" + 0.043*"habitat" + 0.032*"appear" + 0.027*"decline" + 0.026*"suitable" + 0.026*"rare" + 0.023*"relatively"
    
    Topic: trend information \
    Words: 0.196*"information" + 0.159*"size" + 0.158*"trend" + 0.142*"available" + 0.083*"unknown" + 0.046*"datum" + 0.028*"current" + 0.027*"know" + 0.021*"species" + 0.016*"status"
    
    Topic: freshwater decline \
    Words: 0.080*"river" + 0.033*"decline" + 0.032*"lake" + 0.018*"remain" + 0.016*"site" + 0.015*"stream" + 0.011*"species" + 0.011*"basin" + 0.011*"province" + 0.011*"record"
    
    Topic: habitat decline \
    Words: 0.077*"habitat" + 0.065*"decline" + 0.052*"range" + 0.048*"know" + 0.043*"individual" + 0.041*"estimate" + 0.039*"land" + 0.036*"detail" + 0.033*"information" + 0.028*"provide_supplementary"
    
    Topic: fish stock decline \
    Words: 0.053*"catch" + 0.031*"fishery" + 0.024*"decline" + 0.023*"stock" + 0.021*"increase" + 0.018*"landing" + 0.018*"tonne" + 0.014*"fish" + 0.013*"commercial" + 0.013*"biomass"

    rationale peaks at 3

    Topic: vulnerable criteria threshold \
    Words: 0.097*"size" + 0.096*"criterion" + 0.088*"threshold" + 0.087*"vulnerable" + 0.074*"approach" + 0.074*"trend" + 0.066*"range" + 0.037*"believe" + 0.036*"large" + 0.029*"appear"
    
    Topic: habitat decline \
    Words: 0.034*"area" + 0.034*"habitat" + 0.030*"decline" + 0.024*"extent" + 0.021*"occurrence" + 0.019*"know" + 0.018*"threaten" + 0.016*"assess" + 0.016*"location" + 0.015*"endanger"
    
    Topic: monitor distribution \
    Words: 0.045*"major" + 0.044*"distribution" + 0.041*"assess" + 0.038*"wide" + 0.035*"list" + 0.034*"large" + 0.028*"widespread" + 0.026*"currently" + 0.025*"know" + 0.024*"range"

    rationale alternative peak at 20

    Topic: 0 \
    Words: 0.045*"subpopulation" + 0.028*"individual" + 0.023*"area" + 0.020*"plant" + 0.019*"site" + 0.019*"change" + 0.018*"estimate" + 0.015*"habitat" + 0.014*"conservation" + 0.013*"threaten"
    
    Topic: 1 \
    Words: 0.057*"area" + 0.029*"occupancy" + 0.028*"occurrence" + 0.027*"location" + 0.026*"extent" + 0.025*"estimate" + 0.024*"habitat" + 0.024*"occur" + 0.024*"protect" + 0.024*"endemic"
    
    Topic: 2 \
    Words: 0.050*"decline" + 0.025*"generation" + 0.019*"estimate" + 0.017*"past" + 0.015*"reduction" + 0.014*"increase" + 0.013*"range" + 0.012*"datum" + 0.010*"abundance" + 0.010*"base"
    
    Topic: 3 \
    Words: 0.104*"size" + 0.102*"threshold" + 0.102*"criterion" + 0.100*"approach" + 0.100*"vulnerable" + 0.071*"trend" + 0.068*"range" + 0.049*"believe" + 0.039*"large" + 0.036*"evaluate"
    
    Topic: 4 \
    Words: 0.090*"decline" + 0.067*"threaten" + 0.063*"near" + 0.039*"habitat" + 0.036*"range" + 0.030*"list" + 0.029*"vulnerable" + 0.027*"loss" + 0.022*"small" + 0.019*"meet"
    
    Topic: 5 \
    Words: 0.193*"criterion" + 0.105*"qualify" + 0.083*"assess" + 0.077*"decline" + 0.048*"list" + 0.041*"number" + 0.038*"past" + 0.035*"ongoing" + 0.031*"individual" + 0.028*"endemic"
    
    Topic: 6 \
    Words: 0.069*"reef" + 0.066*"reduction" + 0.064*"habitat" + 0.048*"degradation" + 0.043*"loss" + 0.034*"estimate" + 0.031*"coral" + 0.029*"range" + 0.024*"generation" + 0.024*"change"
    
    Topic: 7 \
    Words: 0.124*"assessment" + 0.115*"regional" + 0.089*"widespread" + 0.061*"assess" + 0.042*"major" + 0.040*"stable" + 0.027*"common" + 0.023*"know" + 0.021*"range" + 0.021*"trend"
    
    Topic: 8 \
    Words: 0.096*"significant" + 0.094*"distribution" + 0.091*"wide" + 0.091*"assess" + 0.091*"currently" + 0.089*"major" + 0.089*"identify" + 0.086*"large" + 0.085*"tree" + 0.084*"experience"
    
    Topic: 9 \
    Words: 0.065*"river" + 0.045*"lake" + 0.039*"water" + 0.030*"pollution" + 0.021*"know" + 0.021*"assess" + 0.019*"location" + 0.017*"basin" + 0.017*"impact" + 0.017*"restrict"
    
    Topic: 10 \
    Words: 0.042*"research" + 0.034*"habitat" + 0.033*"assess" + 0.030*"range" + 0.028*"distribution" + 0.026*"size" + 0.026*"trend" + 0.025*"impact" + 0.022*"recommend" + 0.017*"decline"
    
    Topic: 11 \
    Words: 0.106*"know" + 0.102*"list" + 0.098*"major" + 0.081*"distribute" + 0.074*"widely" + 0.046*"widespread" + 0.042*"common" + 0.027*"abundant" + 0.026*"occur" + 0.024*"range"
    
    Topic: 12 \
    Words: 0.065*"range" + 0.056*"area" + 0.044*"list" + 0.038*"habitat" + 0.037*"protect" + 0.035*"occur" + 0.030*"common" + 0.025*"occurrence" + 0.025*"distribution" + 0.024*"widespread"
    
    Topic: 13 \
    Words: 0.047*"fishery" + 0.031*"range" + 0.023*"catch" + 0.021*"fishing" + 0.021*"depth" + 0.018*"water" + 0.017*"occur" + 0.016*"area" + 0.015*"bycatch" + 0.014*"trawl"
    
    Topic: 14 \
    Words: 0.067*"island" + 0.033*"endanger" + 0.030*"restrict" + 0.030*"area" + 0.029*"critically" + 0.027*"small" + 0.026*"location" + 0.024*"vulnerable" + 0.021*"range" + 0.019*"know"
    
    Topic: 15 \
    Words: 0.061*"forest" + 0.051*"area" + 0.040*"know" + 0.035*"habitat" + 0.035*"occurrence" + 0.033*"extent" + 0.024*"occupancy" + 0.022*"location" + 0.020*"locality" + 0.020*"tree"
    
    Topic: 16 \
    Words: 0.059*"know" + 0.037*"record" + 0.028*"survey" + 0.025*"locality" + 0.018*"speciman" + 0.018*"collect" + 0.016*"datum" + 0.016*"status" + 0.016*"habitat" + 0.015*"information"
    
    Topic: 17 \
    Words: 0.071*"extent" + 0.064*"habitat" + 0.054*"decline" + 0.044*"occurrence" + 0.042*"quality" + 0.041*"location" + 0.041*"continue" + 0.040*"area" + 0.037*"endanger" + 0.030*"list"
    
    Topic: 18 \
    Words: 0.052*"area" + 0.049*"threaten" + 0.042*"category" + 0.036*"extent" + 0.034*"occurrence" + 0.033*"exceed" + 0.025*"criterion" + 0.024*"value" + 0.023*"habitat" + 0.023*"estimate"
    
    Topic: 19 \
    Words: 0.131*"list" + 0.074*"distribution" + 0.073*"wide" + 0.072*"large" + 0.071*"view" + 0.060*"unlikely" + 0.059*"category" + 0.058*"presume" + 0.058*"threaten" + 0.057*"decline"

    range peaks at 43

    Topic: 0 \
    Words: 0.445*"province" + 0.079*"know" + 0.054*"record" + 0.036*"occur" + 0.024*"subpopulation" + 0.024*"sichuan" + 0.020*"hainan" + 0.018*"nature_reserve" + 0.016*"guangxi" + 0.012*"laos"

    Topic: 1 \
    Words: 0.200*"range" + 0.109*"occur" + 0.068*"extend" + 0.053*"lowland" + 0.046*"elevational" + 0.045*"slope" + 0.041*"elevation" + 0.037*"level" + 0.032*"valley" + 0.022*"extreme"

    Topic: 2 \
    Words: 0.132*"department" + 0.094*"occur" + 0.051*"know" + 0.031*"slope" + 0.026*"record" + 0.026*"locality" + 0.019*"region" + 0.017*"antioquia" + 0.017*"magdalena" + 0.016*"elevation"

    Topic: 3 \
    Words: 0.201*"locality" + 0.152*"know" + 0.148*"collect" + 0.118*"type" + 0.094*"speciman" + 0.043*"collection" + 0.030*"single" + 0.018*"endemic" + 0.016*"tree" + 0.013*"near"

    Topic: 4 \
    Words: 0.565*"island" + 0.079*"endemic" + 0.031*"occur" + 0.021*"include" + 0.013*"virgin" + 0.012*"know" + 0.011*"islet" + 0.010*"offshore" + 0.010*"jamaica" + 0.010*"archipelago"

    Topic: 5 \
    Words: 0.182*"record" + 0.105*"report" + 0.041*"confirm" + 0.038*"recent" + 0.028*"survey" + 0.026*"presence" + 0.023*"extinct" + 0.021*"consider" + 0.018*"speciman" + 0.014*"likely"

    Topic: 6 \
    Words: 0.375*"forest" + 0.064*"reserve" + 0.031*"know" + 0.027*"ghat" + 0.027*"lowland" + 0.026*"record" + 0.025*"montane" + 0.019*"occur" + 0.019*"tamil_nadu" + 0.017*"report"

    Topic: 7 \
    Words: 0.136*"coastal" + 0.080*"region" + 0.074*"occur" + 0.062*"area" + 0.047*"territory" + 0.040*"plain" + 0.038*"confine" + 0.035*"range" + 0.028*"inland" + 0.022*"near"

    Topic: 8 \
    Words: 0.492*"coast" + 0.048*"occur" + 0.038*"peninsula" + 0.027*"massif" + 0.026*"include" + 0.017*"estuary" + 0.016*"island" + 0.015*"present" + 0.013*"head" + 0.011*"strait"

    Topic: 9 \
    Words: 0.185*"location" + 0.102*"know" + 0.076*"area" + 0.037*"occurrence" + 0.036*"extent" + 0.034*"currently" + 0.033*"record" + 0.033*"occupancy" + 0.031*"threat_define" + 0.025*"possible"

    Topic: 10 \
    Words: 0.306*"mountain" + 0.072*"occur" + 0.070*"mount" + 0.070*"range" + 0.045*"know" + 0.025*"record" + 0.017*"elevation" + 0.017*"locality" + 0.015*"slope" + 0.012*"widely"

    Topic: 11 \
    Words: 0.117*"population" + 0.036*"individual" + 0.031*"bird" + 0.029*"decline" + 0.026*"estimate" + 0.024*"number" + 0.023*"breed" + 0.020*"pair" + 0.016*"breeding" + 0.014*"small"

    Topic: 12 \
    Words: 0.372*"lake" + 0.074*"river" + 0.041*"stream" + 0.036*"drainage" + 0.021*"know" + 0.018*"shore" + 0.018*"system" + 0.018*"tanganyika" + 0.016*"basin" + 0.014*"occur"

    Topic: 13 \
    Words: 0.111*"wide" + 0.093*"distribution" + 0.052*"know" + 0.049*"chad" + 0.049*"niger" + 0.046*"basin" + 0.043*"nile" + 0.038*"river" + 0.036*"delta" + 0.031*"cross"

    Topic: 14 \
    Words: 0.222*"native" + 0.127*"introduce" + 0.036*"range" + 0.033*"occur" + 0.022*"widespread" + 0.022*"distribution" + 0.022*"cultivate" + 0.020*"introduction" + 0.020*"consider" + 0.019*"widely"

    Topic: 15 \
    Words: 0.223*"altitude" + 0.047*"occur" + 0.041*"siberia" + 0.027*"peninsula" + 0.027*"mainly" + 0.024*"greenland" + 0.023*"distribution" + 0.023*"iceland" + 0.022*"arctic" + 0.019*"pyrenee"

    Topic: 16 \
    Words: 0.396*"river" + 0.075*"basin" + 0.053*"system" + 0.051*"know" + 0.041*"drainage" + 0.029*"tributary" + 0.022*"creek" + 0.021*"occur" + 0.014*"stream" + 0.014*"include"

    Topic: 17 \
    Words: 0.066*"million" + 0.062*"occurrence" + 0.056*"area" + 0.055*"extent" + 0.046*"estimate" + 0.041*"occupancy" + 0.036*"occur" + 0.031*"altitudinal" + 0.027*"distribution" + 0.026*"outside"

    Topic: 18 \
    Words: 0.173*"national" + 0.170*"park" + 0.038*"record" + 0.036*"area" + 0.033*"range" + 0.032*"forest" + 0.027*"protect" + 0.018*"occur" + 0.018*"reserve" + 0.015*"know"

    Topic: 19 \
    Words: 0.647*"endemic" + 0.106*"occur" + 0.033*"area" + 0.032*"plant" + 0.030*"region" + 0.025*"plateau" + 0.019*"orchid" + 0.016*"peninsula" + 0.009*"know" + 0.007*"small"

    Topic: 20 \
    Words: 0.080*"tree" + 0.050*"taxon" + 0.038*"area" + 0.033*"measure" + 0.031*"current" + 0.029*"occur" + 0.028*"occurrence" + 0.028*"likely" + 0.026*"large" + 0.025*"extent"

    Topic: 21 \
    Words: 0.350*"island" + 0.050*"mainland" + 0.037*"record" + 0.029*"archipelago" + 0.027*"know" + 0.027*"group" + 0.023*"occur" + 0.022*"vanuatu" + 0.019*"palau" + 0.019*"samoa"

    Topic: 22 \
    Words: 0.215*"occur" + 0.175*"state" + 0.034*"mina_gerais" + 0.032*"kazakhstan" + 0.030*"guyana" + 0.029*"paulo" + 0.028*"know" + 0.027*"amazona" + 0.024*"mongolia" + 0.020*"catarina"

    Topic: 23 \
    Words: 0.157*"occurrence" + 0.150*"extent" + 0.118*"area" + 0.114*"estimate" + 0.104*"occupancy" + 0.062*"endemic" + 0.034*"know" + 0.019*"occur" + 0.018*"elevation" + 0.013*"location"

    Topic: 24 \
    Words: 0.117*"distribution" + 0.045*"range" + 0.042*"record" + 0.022*"species" + 0.022*"region" + 0.014*"limit" + 0.014*"consider" + 0.013*"include" + 0.012*"list" + 0.012*"determine"

    Topic: 25 \
    Words: 0.096*"locality" + 0.070*"describe" + 0.065*"know" + 0.062*"recently" + 0.033*"locate" + 0.032*"record" + 0.031*"discover" + 0.028*"separate" + 0.027*"unknown" + 0.025*"near"

    Topic: 26 \
    Words: 0.083*"area" + 0.075*"habitat" + 0.057*"site" + 0.047*"subpopulation" + 0.032*"suitable" + 0.028*"range" + 0.027*"small" + 0.018*"know" + 0.017*"survey" + 0.014*"occupy"

    Topic: 27 \
    Words: 0.122*"large" + 0.062*"know" + 0.058*"area" + 0.051*"exist" + 0.051*"record" + 0.050*"occurrence" + 0.047*"occupancy" + 0.046*"extent" + 0.046*"consider" + 0.040*"sample"

    Topic: 28 \
    Words: 0.221*"distribute" + 0.123*"depth" + 0.091*"widely" + 0.089*"range" + 0.063*"metre" + 0.047*"record" + 0.024*"island" + 0.023*"know" + 0.019*"ocean" + 0.017*"include"

    Topic: 29 \
    Words: 0.106*"collection" + 0.084*"herbarium" + 0.060*"speciman" + 0.059*"available" + 0.049*"datum" + 0.047*"occurrence" + 0.046*"area" + 0.046*"base" + 0.038*"extent" + 0.035*"occupancy"

    Topic: 30 \
    Words: 0.210*"subspecie" + 0.094*"occur" + 0.035*"mississippi" + 0.032*"virginia" + 0.031*"range" + 0.029*"alabama" + 0.028*"york" + 0.023*"tennessee" + 0.019*"great" + 0.018*"louisiana"

    Topic: 31 \
    Words: 0.135*"peninsular" + 0.060*"drainage" + 0.053*"record" + 0.039*"occur" + 0.038*"polygon" + 0.036*"river" + 0.036*"base" + 0.035*"minimum_convex" + 0.031*"occurrence" + 0.030*"know"

    Topic: 32 \
    Words: 0.215*"basin" + 0.086*"black" + 0.079*"present" + 0.031*"spread" + 0.028*"caspian" + 0.028*"record" + 0.027*"include" + 0.024*"drainage" + 0.021*"isle" + 0.019*"baltic"

    Topic: 33 \
    Words: 0.299*"restrict" + 0.271*"know" + 0.088*"distribution" + 0.048*"region" + 0.048*"scatter" + 0.037*"cave" + 0.032*"locality" + 0.022*"range" + 0.021*"poorly" + 0.018*"area"

    Topic: 34 \
    Words: 0.140*"county" + 0.075*"spring" + 0.048*"arizona" + 0.047*"range" + 0.045*"occur" + 0.030*"portion" + 0.029*"colorado" + 0.028*"nevada" + 0.024*"sonora" + 0.023*"population"

    Topic: 35 \
    Words: 0.112*"widespread" + 0.079*"occur" + 0.075*"distribution" + 0.075*"record" + 0.071*"democratic" + 0.042*"know" + 0.042*"global" + 0.040*"range" + 0.035*"region" + 0.021*"botswana"

    Topic: 36 \
    Words: 0.082*"island" + 0.053*"mindanao" + 0.052*"luzon" + 0.048*"forest" + 0.045*"negro" + 0.043*"natural" + 0.041*"record" + 0.034*"endemic" + 0.023*"mindoro" + 0.022*"laguna"

    Topic: 37 \
    Words: 0.152*"ocean" + 0.087*"tropical" + 0.075*"occur" + 0.064*"water" + 0.050*"depth" + 0.045*"include" + 0.043*"island" + 0.023*"subtropical" + 0.023*"reef" + 0.019*"ridge"

    Topic: 38 \
    Words: 0.100*"common" + 0.052*"present" + 0.045*"widespread" + 0.040*"rare" + 0.039*"occur" + 0.036*"region" + 0.029*"locally" + 0.026*"range" + 0.024*"absent" + 0.021*"area"

    Topic: 39 \
    Words: 0.195*"elevation" + 0.159*"endemic" + 0.108*"province" + 0.099*"grow" + 0.091*"level" + 0.076*"occur" + 0.041*"region" + 0.035*"distribute" + 0.024*"antsiranana" + 0.022*"toamasina"

    Topic: 40 \
    Words: 0.119*"°" + 0.045*"water" + 0.045*"range" + 0.027*"distribution" + 0.017*"occur" + 0.016*"region" + 0.015*"area" + 0.014*"winter" + 0.014*"catch" + 0.014*"record"

    Topic: 41 \
    Words: 0.053*"threat" + 0.052*"area" + 0.050*"location" + 0.045*"estimate" + 0.036*"base" + 0.035*"extent" + 0.033*"habitat" + 0.032*"occurrence" + 0.030*"occupancy" + 0.028*"decline"

    Topic: 42 \
    Words: 0.127*"district" + 0.124*"near" + 0.084*"province" + 0.068*"know" + 0.054*"hill" + 0.049*"highland" + 0.042*"village" + 0.038*"endemic" + 0.030*"shrub" + 0.029*"road"

    threats peak at 10

    Topic: protect forest habitats \
    Words: 0.050*"area" + 0.038*"forest" + 0.031*"habitat" + 0.016*"protect" + 0.016*"loss" + 0.014*"occur" + 0.013*"range" + 0.012*"deforestation" + 0.012*"land" + 0.011*"human"

    Topic: forest decline \
    Words: 0.089*"forest" + 0.032*"log" + 0.024*"area" + 0.023*"plantation" + 0.018*"agriculture" + 0.017*"decline" + 0.016*"habitat" + 0.015*"clear" + 0.013*"tree" + 0.013*"timber"

    Topic: fishing \
    Words: 0.059*"fishery" + 0.030*"catch" + 0.028*"fishing" + 0.022*"trawl" + 0.020*"bycatch" + 0.018*"fish" + 0.014*"target" + 0.013*"water" + 0.013*"range" + 0.012*"commercial"

    Topic: habitat loss to agriculture and deforestation \
    Words: 0.094*"habitat" + 0.053*"threaten" + 0.045*"loss" + 0.042*"agriculture" + 0.034*"log" + 0.030*"major" + 0.030*"deforestation" + 0.028*"destruction" + 0.023*"activity" + 0.021*"mining"

    Topic: climate change impact island populations \
    Words: 0.039*"change" + 0.032*"climate" + 0.030*"island" + 0.022*"decline" + 0.017*"impact" + 0.016*"population" + 0.015*"introduce" + 0.014*"habitat" + 0.014*"increase" + 0.013*"risk"

    Topic: major impact to range and habitat \
    Words: 0.126*"major" + 0.120*"know" + 0.029*"range" + 0.028*"impact" + 0.026*"habitat" + 0.023*"future" + 0.021*"significant" + 0.020*"affect" + 0.018*"appear" + 0.018*"currently"

    Topic: fire threat \
    Words: 0.038*"habitat" + 0.031*"fire" + 0.023*"plant" + 0.017*"threaten" + 0.015*"population" + 0.015*"subpopulation" + 0.014*"invasive" + 0.011*"affect" + 0.011*"graze" + 0.010*"impact"

    Topic: threats to coral reef \
    Words: 0.088*"coral" + 0.059*"disease" + 0.048*"reef" + 0.043*"increase" + 0.022*"global" + 0.020*"change" + 0.018*"habitat" + 0.015*"major" + 0.014*"climate" + 0.013*"temperature"

    Topic: habitat loss \
    Words: 0.032*"population" + 0.017*"habitat" + 0.013*"hunt" + 0.013*"decline" + 0.011*"increase" + 0.010*"loss" + 0.010*"cause" + 0.009*"area" + 0.009*"nest" + 0.009*"range"

    Topic: freshwater pollution \
    Words: 0.045*"water" + 0.034*"pollution" + 0.032*"habitat" + 0.030*"river" + 0.019*"impact" + 0.016*"lake" + 0.013*"fish" + 0.013*"development" + 0.011*"include" + 0.010*"population"

    use trade peaks at 14

    Topic: commercial fishing \
    Words: 0.071*"catch" + 0.069*"fishery" + 0.034*"commercially" + 0.033*"trawl" + 0.030*"fish" + 0.027*"bycatch" + 0.025*"market" + 0.022*"target" + 0.022*"commercial" + 0.019*"exploit"
    
    Topic:  \
    Words: 0.237*"record" + 0.095*"potential" + 0.087*"wild" + 0.077*"relative" + 0.062*"gene_donor" + 0.045*"cultivate" + 0.024*"crop" + 0.023*"tertiary" + 0.023*"taxon" + 0.018*"secondary"
    
    Topic: information available on trade \
    Words: 0.464*"trade" + 0.267*"information" + 0.081*"available" + 0.077*"know" + 0.014*"genus" + 0.012*"small" + 0.011*"number" + 0.011*"insignificant" + 0.011*"cultivate" + 0.008*"commercial"
    
    Topic: ornamental fish trade \
    Words: 0.194*"trade" + 0.140*"aquarium" + 0.101*"fish" + 0.082*"collect" + 0.053*"ornamental" + 0.052*"unknown" + 0.034*"food" + 0.031*"target" + 0.027*"occasionally" + 0.018*"know"
    
    Topic: shark fin harvest \
    Words: 0.238*"harvest" + 0.209*"consumption" + 0.196*"human" + 0.035*"meat" + 0.022*"fin" + 0.021*"palatability" + 0.021*"lipid_content" + 0.021*"myctophid_ester" + 0.020*"shark" + 0.014*"consume"
    
    Topic: illegal international trade \
    Words: 0.059*"trade" + 0.042*"export" + 0.026*"individual" + 0.023*"population" + 0.022*"wild" + 0.020*"international" + 0.018*"live" + 0.015*"increase" + 0.014*"illegal" + 0.013*"animal"
    
    Topic: shell trade \
    Words: 0.066*"trade" + 0.063*"market" + 0.051*"hunt" + 0.048*"shell" + 0.030*"price" + 0.027*"collector" + 0.026*"common" + 0.023*"range" + 0.021*"number" + 0.020*"sell"
    
    Topic: ornamental plant trade \
    Words: 0.087*"ornamental" + 0.086*"plant" + 0.036*"cultivate" + 0.035*"wild" + 0.034*"collect" + 0.033*"grow" + 0.032*"cultivation" + 0.032*"collection" + 0.026*"purpose" + 0.021*"evidence"
    
    Topic: wood use trade \
    Words: 0.085*"timber" + 0.074*"wood" + 0.056*"tree" + 0.055*"construction" + 0.027*"furniture" + 0.018*"firewood" + 0.016*"house" + 0.015*"produce" + 0.015*"charcoal" + 0.013*"harvest"
    
    Topic: commercial fishing \
    Words: 0.088*"fishery" + 0.083*"commercial" + 0.061*"local" + 0.048*"subsistence" + 0.046*"small" + 0.041*"food" + 0.035*"importance" + 0.034*"fish" + 0.028*"value" + 0.027*"minor"
    
    Topic: 10 \
    Words: 0.222*"use" + 0.190*"utilise" + 0.175*"know" + 0.094*"important" + 0.076*"eucalypt_culturally" + 0.057*"variety" + 0.055*"mean" + 0.047*"indigenous_great" + 0.038*"indigenous" + 0.028*"variety_meaning"
    
    Topic: medicinal use \
    Words: 0.305*"report" + 0.137*"use" + 0.050*"appear" + 0.040*"genus" + 0.036*"medicinal" + 0.028*"specific" + 0.026*"purpose" + 0.025*"eat" + 0.019*"seed" + 0.019*"fruit"
    
    Topic: 12 \
    Words: 0.933*"utilize" + 0.032*"know" + 0.015*"unlikely" + 0.007*"skate" + 0.002*"likely" + 0.002*"generally" + 0.002*"discard" + 0.001*"record" + 0.001*"present" + 0.000*"datum"
    
    Topic: medicinal plants \
    Words: 0.042*"leave" + 0.040*"fruit" + 0.034*"plant" + 0.030*"medicine" + 0.027*"medicinal" + 0.024*"treat" + 0.024*"edible" + 0.022*"root" + 0.020*"bark" + 0.020*"eat"

    Habitat peaks at 17

    17 Topic:

    Topic: bird ecology \
    Words: 0.039*"nest" + 0.029*"breed" + 0.023*"breeding" + 0.021*"feed" + 0.018*"small" + 0.016*"bird" + 0.016*"insect" + 0.016*"diet" + 0.015*"ground" + 0.014*"forage"

    Topic: freshwater systems \
    Words: 0.084*"water" + 0.057*"occur" + 0.040*"pond" + 0.033*"freshwater" + 0.027*"flood" + 0.027*"shallow" + 0.022*"margin" + 0.020*"grow" + 0.020*"ditch" + 0.018*"slow"

    Topic: rock habitats \
    Words: 0.048*"crevice" + 0.040*"limestone" + 0.029*"stone" + 0.023*"outcrop" + 0.023*"area" + 0.022*"boulder" + 0.022*"wall" + 0.021*"cliff" + 0.017*"usually" + 0.016*"base"

    Topic: plant hosts for insects \
    Words: 0.048*"tree" + 0.027*"plant" + 0.027*"wood" + 0.024*"host" + 0.019*"leave" + 0.018*"adult" + 0.016*"dead" + 0.015*"live" + 0.015*"larvae" + 0.013*"occur"

    Topic: plants on slopes habitats *phrasing* \
    Words: 0.052*"slope" + 0.047*"grow" + 0.034*"herb" + 0.032*"occur" + 0.021*"altitude" + 0.020*"pine" + 0.019*"shrub" + 0.019*"steep" + 0.015*"tree" + 0.014*"quercus"

    Topic: plant ecology \
    Words: 0.103*"grow" + 0.084*"tree" + 0.046*"tall" + 0.041*"shrub" + 0.035*"small" + 0.027*"soil" + 0.026*"flower" + 0.023*"occur" + 0.020*"seed" + 0.018*"humid"

    Topic: plant ecology (in calcareous habitats) \
    Words: 0.048*"grow" + 0.048*"soil" + 0.022*"plant" + 0.020*"occur" + 0.016*"calcareous" + 0.015*"shade" + 0.013*"rich" + 0.013*"moist" + 0.011*"bank" + 0.011*"damp"

    Topic: open area vegetation \
    Words: 0.065*"area" + 0.048*"open" + 0.032*"occur" + 0.029*"vegetation" + 0.024*"scrub" + 0.018*"grass" + 0.018*"semi" + 0.018*"arid" + 0.017*"soil" + 0.017*"edge"

    Topic: forest floor habitat (unsure) \
    Words: 0.025*"breed" + 0.023*"inhabit" + 0.022*"secondary" + 0.021*"litter" + 0.020*"area" + 0.020*"leaf" + 0.019*"occur" + 0.018*"primary" + 0.015*"disturb" + 0.014*"disturbance"

    Topic: small water habitats \
    Words: 0.066*"water" + 0.041*"vegetation" + 0.035*"egg" + 0.026*"breed" + 0.025*"season" + 0.017*"shallow" + 0.015*"small" + 0.014*"pond" + 0.013*"area" + 0.011*"larvae"

    Topic: coral habitats \
    Words: 0.065*"length" + 0.063*"maximum" + 0.046*"depth" + 0.035*"coral" + 0.035*"inhabit" + 0.032*"occur" + 0.029*"standard" + 0.027*"water" + 0.023*"feed" + 0.022*"shallow"

    Topic: gather more information \
    Words: 0.060*"know" + 0.048*"record" + 0.036*"information" + 0.028*"ecology" + 0.028*"collect" + 0.026*"speciman" + 0.021*"available" + 0.020*"little" + 0.019*"likely" + 0.018*"area"

    Topic: evergreen habitat \
    Words: 0.069*"occur" + 0.066*"tree" + 0.035*"evergreen" + 0.033*"elevation" + 0.029*"range" + 0.028*"secondary" + 0.026*"primary" + 0.026*"small" + 0.025*"humid" + 0.019*"record"

    Topic: ecology of species
    Words: 0.021*"range" + 0.016*"female" + 0.015*"population" + 0.013*"area" + 0.012*"male" + 0.011*"individual" + 0.010*"size" + 0.010*"group" + 0.010*"large" + 0.009*"study"

    Topic: lifecycle \ 
    Words: 0.063*"length" + 0.056*"size" + 0.055*"female" + 0.044*"generation" + 0.038*"maximum" + 0.037*"male" + 0.031*"mature" + 0.030*"reach" + 0.029*"maturity" + 0.026*"estimate"

    Topic: (fresh)water habitat \
    Words: 0.052*"water" + 0.043*"flow" + 0.039*"small" + 0.028*"clear" + 0.023*"large" + 0.021*"inhabit" + 0.021*"fast" + 0.020*"occur" + 0.015*"current" + 0.015*"moderate"

    Topic: fish ecology \
    Words: 0.032*"spawn" + 0.027*"fish" + 0.026*"water" + 0.026*"male" + 0.024*"female" + 0.023*"feed" + 0.021*"occur" + 0.017*"egg" + 0.015*"juvenile" + 0.013*"small"

9) Altered cleaning function in prep.py to not convert all text to lowercase and remove punctuations. This allowed spaCy to detect proper noun(s) (such as country names), which further shorted my custom stop word list.

    conAct peaks at 15:

    Topic: monitor_situation
    Words: 0.188*"measure" + 0.136*"place" + 0.135*"specific" + 0.083*"species" + 0.050*"know" + 0.043*"range" + 0.037*"need" + 0.030*"distribution" + 0.030*"marine" + 0.023*"occur"
    
    Topic: fishery_management 
    Words: 0.041*"fishery" + 0.033*"fishing" + 0.025*"catch" + 0.021*"measure" + 0.019*"management" + 0.017*"size" + 0.017*"limit" + 0.017*"water" + 0.014*"marine" + 0.013*"bycatch"
    
    Topic: forest_habitat_management
    Words: 0.043*"forest" + 0.043*"habitat" + 0.018*"local" + 0.018*"management" + 0.016*"land" + 0.015*"protection" + 0.013*"activity" + 0.012*"population" + 0.010*"recommend" + 0.
    010*"sustainable"
    
    Topic: monitor_species_ecology
    Words: 0.233*"occur" + 0.152*"know" + 0.058*"species" + 0.053*"collection" + 0.032*"subpopulation" + 0.018*"record" + 0.018*"legislation" + 0.017*"forest" + 0.015*"locality" + 0.014*"site"
    
    Topic: assess_present_threat
    Words: 0.137*"list" + 0.131*"assess" + 0.101*"threaten" + 0.078*"present" + 0.040*"national" + 0.030*"occur" + 0.030*"consider" + 0.027*"range" + 0.027*"database" + 0.025*"find"
    
    Topic: trade_risk
    Words: 0.021*"risk" + 0.019*"country" + 0.017*"chytrid" + 0.016*"trade" + 0.016*"activity" + 0.014*"policy" + 0.014*"human" + 0.013*"need" + 0.012*"cause" + 0.012*"develop"
    
    Topic: habitat_based_conservation
    Words: 0.037*"survey" + 0.032*"population" + 0.031*"habitat" + 0.027*"reserve" + 0.027*"range" + 0.023*"forest" + 0.020*"status" + 0.019*"need" + 0.018*"determine" + 0.015*"protection"
    
    Topic: manage_trade_threat
    Words: 0.075*"management" + 0.038*"population" + 0.033*"trade" + 0.031*"include" + 0.030*"threat" + 0.027*"trend" + 0.026*"coral" + 0.023*"recommend" + 0.022*"measure" + 0.022*"habitat"
    
    Topic: subpopulation_assessment
    Words: 0.027*"subpopulation" + 0.021*"assessment" + 0.017*"datum" + 0.016*"include" + 0.016*"population" + 0.013*"region" + 0.012*"important" + 0.011*"site" + 0.011*"monitoring" + 0.010*"take"
    
    Topic: act_on_current_information
    Words: 0.174*"action" + 0.076*"information" + 0.067*"currently" + 0.065*"available" + 0.060*"recommend" + 0.053*"place" + 0.050*"require" + 0.037*"research" + 0.037*"major" + 0.036*"additional"
    
    Topic: site_based_conservation
    Words: 0.054*"find" + 0.047*"site" + 0.042*"population" + 0.039*"protection" + 0.034*"recommend" + 0.032*"habitat" + 0.022*"collection" + 0.021*"management" + 0.021*"awareness" + 0.018*"effort"
    
    Topic: monitor_accession
    Words: 0.109*"collection" + 0.063*"report" + 0.049*"record" + 0.029*"wild" + 0.028*"accession" + 0.025*"collect" + 0.024*"conserve" + 0.020*"seed" + 0.018*"hold" + 0.017*"list"
    
    Topic: breeding_habitat
    Words: 0.034*"population" + 0.021*"habitat" + 0.015*"breeding" + 0.013*"propose" + 0.013*"site" + 0.012*"plan" + 0.011*"management" + 0.010*"programme" + 0.009*"control" + 0.009*"research"
    
    Topic: protect_range
    Words: 0.036*"include" + 0.032*"list" + 0.028*"range" + 0.027*"seed" + 0.027*"protection" + 0.026*"level" + 0.025*"country" + 0.021*"plant" + 0.020*"occur" + 0.019*"taxon"
    
    Topic: research_population_threat
    Words: 0.091*"need" + 0.084*"population" + 0.070*"research" + 0.052*"threat" + 0.050*"distribution" + 0.043*"trend" + 0.030*"ecology" + 0.027*"habitat" + 0.026*"size" + 0.025*"species"

10) Overhaul, train model on combined

    Peaks at 13

    Topic: site_based_conservation
    Words: 0.067*"conservation" + 0.038*"know" + 0.036*"protect" + 0.035*"area" + 0.033*"site" + 0.032*"record"
    
    Topic: protect_area_with_endemics
    Words: 0.120*"area" + 0.115*"occur" + 0.074*"protect" + 0.062*"know" + 0.054*"endemic" + 0.052*"species"
    
    Topic: know_major_threat
    Words: 0.201*"threat" + 0.112*"major" + 0.105*"know" + 0.044*"widespread" + 0.042*"assess" + 0.040*"currently"
    
    Topic: trade_and_fishery
    Words: 0.072*"trade" + 0.063*"population" + 0.039*"common" + 0.037*"fishery" + 0.032*"catch" + 0.030*"information"
    
    Topic: forest_habitat
    Words: 0.068*"forest" + 0.065*"find" + 0.032*"occur" + 0.027*"tree" + 0.024*"river" + 0.023*"small"
    
    Topic: population_size_vulnerable_trend
    Words: 0.199*"population" + 0.138*"size" + 0.105*"trend" + 0.055*"vulnerable" + 0.053*"criterion" + 0.045*"range"
    
    Topic: human_threat_to_habitat
    Words: 0.066*"threat" + 0.054*"habitat" + 0.028*"impact" + 0.025*"human" + 0.023*"activity" + 0.018*"plant"
    
    Topic: european_population_range
    Words: 0.093*"range" + 0.047*"common" + 0.045*"european" + 0.044*"population" + 0.043*"report" + 0.040*"widespread"
    
    Topic: wide_distribution
    Words: 0.094*"distribution" + 0.068*"wide" + 0.064*"threaten" + 0.056*"large" + 0.055*"category" + 0.044*"unlikely"
    
    Topic: forest_habitat_loss_to_agriculture
    Words: 0.097*"forest" + 0.087*"habitat" + 0.045*"area" + 0.035*"loss" + 0.027*"threat" + 0.023*"agriculture"
    
    Topic: population_information
    Words: 0.089*"population" + 0.058*"estimate" + 0.049*"decline" + 0.045*"individual" + 0.041*"subpopulation" + 0.039*"number"
    
    Topic: specific_conservation_measure
    Words: 0.069*"conservation_measure" + 0.067*"place" + 0.059*"specific" + 0.056*"need" + 0.051*"protect" + 0.048*"area"
    
    Topic: coral_reef
    Words: 0.048*"reef" + 0.041*"coral" + 0.041*"depth" + 0.033*"year" + 0.033*"utilize" + 0.032*"length"

    Next peak 20:

    Topic: monitor_species
    Words: 0.119*"know" + 0.074*"locality" + 0.063*"collection" + 0.053*"collect" + 0.047*"site" + 0.039*"specimen" + 0.035*"record" + 0.028*"type"

    Topic: area_based_protection
    Words: 0.174*"protect" + 0.159*"area" + 0.089*"occur" + 0.033*"species" + 0.029*"conservation" + 0.028*"know" + 0.026*"list" + 0.025*"range"

    Topic: assess_major_threat
    Words: 0.227*"threat" + 0.138*"major" + 0.081*"know" + 0.048*"assess" + 0.044*"widespread" + 0.044*"currently" + 0.040*"significant" + 0.038*"distribution"

    Topic: monitor_trade
    Words: 0.221*"trade" + 0.137*"information" + 0.099*"population" + 0.086*"information_available" + 0.045*"available" + 0.040*"know" + 0.035*"datum" + 0.027*"trend"

    Topic: Monitor_aquatic_habitat
    Words: 0.045*"find" + 0.044*"water" + 0.032*"river" + 0.030*"occur" + 0.028*"stream" + 0.026*"habitat" + 0.024*"area" + 0.023*"small"

    Topic: monitor_population_size
    Words: 0.176*"population" + 0.138*"size" + 0.100*"trend" + 0.079*"vulnerable" + 0.076*"criterion" + 0.069*"approach_threshold" + 0.058*"range" + 0.036*"large"

    Topic: human_threat_to_habitat
    Words: 0.083*"threat" + 0.046*"habitat" + 0.036*"impact" + 0.031*"human" + 0.026*"activity" + 0.025*"pollution" + 0.020*"development" + 0.019*"plant"

    Topic: european_regional_assessment
    Words: 0.108*"european" + 0.098*"report" + 0.077*"regional_assessment" + 0.045*"population" + 0.034*"range" + 0.032*"concern" + 0.031*"country" + 0.031*"assess"

    Topic: wide_distribution
    Words: 0.093*"distribution" + 0.086*"wide" + 0.075*"threaten" + 0.069*"large" + 0.064*"category" + 0.057*"unlikely" + 0.048*"population" + 0.046*"presume"

    Topic: forest_habitat_loss_to_agriculture
    Words: 0.088*"habitat" + 0.063*"forest" + 0.044*"area" + 0.032*"loss" + 0.032*"threat" + 0.031*"agriculture" + 0.025*"threaten" + 0.025*"land"

    Topic: Monitor_population_structure
    Words: 0.104*"population" + 0.073*"individual" + 0.071*"subpopulation" + 0.062*"estimate" + 0.057*"number" + 0.035*"mature_individual" + 0.025*"small" + 0.023*"total"

    Topic: restricted_range
    Words: 0.076*"range" + 0.065*"restrict" + 0.053*"believe" + 0.048*"population" + 0.040*"criterion" + 0.035*"list" + 0.032*"vulnerable" + 0.025*"status"

    Topic: coral_reef_utility
    Words: 0.103*"reef" + 0.094*"coral" + 0.073*"utilize" + 0.058*"year" + 0.046*"length" + 0.024*"estimate" + 0.024*"increase" + 0.023*"generation"

    Topic: commercial_fishing
    Words: 0.079*"fishery" + 0.065*"catch" + 0.042*"fish" + 0.039*"fishing" + 0.032*"commercial" + 0.021*"water" + 0.019*"target" + 0.019*"range"

    Topic: species_range_parameters
    Words: 0.093*"area" + 0.074*"extent_occurrence" + 0.062*"occupancy" + 0.062*"know" + 0.061*"location" + 0.046*"estimate" + 0.041*"specific" + 0.040*"conservation_measure"

    Topic: conservation_action
    Words: 0.076*"conservation" + 0.065*"population" + 0.052*"need" + 0.044*"research" + 0.038*"habitat" + 0.035*"action" + 0.027*"place" + 0.024*"protection"

    Topic: population_decline_habitat_loss
    Words: 0.136*"decline" + 0.106*"population" + 0.066*"habitat" + 0.034*"suspect" + 0.029*"range" + 0.028*"loss" + 0.026*"ongoing" + 0.025*"reduction"

    Topic: forest_habitat
    Words: 0.175*"forest" + 0.075*"find" + 0.066*"tree" + 0.050*"occur" + 0.041*"grow" + 0.035*"species" + 0.030*"lowland" + 0.028*"elevation"

    Topic: population_information
    Words: 0.196*"common" + 0.087*"population" + 0.076*"abundant" + 0.068*"locally" + 0.058*"range" + 0.039*"global" + 0.038*"size" + 0.038*"widespread"

    Topic: monitor_island_endemics
    Words: 0.143*"record" + 0.059*"island" + 0.055*"occur" + 0.053*"range" + 0.048*"find" + 0.036*"distribution" + 0.030*"species" + 0.028*"endemic"

### presentation notes

from james: the narrative heavily revolves around monitoring the situation
especially pronounced for marine systems, logic is that marine systems are the largest in the world and least we know about because it is difficult to study.
monitoring plants and protection legislation not evident in the literature 

species also singular

rephrase for biogeo realm
explore different data vis techniques for mapping topics on realms
  cluster topics more coarsely
  can use wordcloud with heatmap to better visualize
  "across all species that exist in that realm", "species counted in each of the realms which they occur"
  use a species as an example to explain how the analysis was carried out
    so explain the lda process with a specie

visualize clustering of topics, to see which ones exist close together (tSNE represent or PCA or any dimensionality reduction technique)
  can compare the most coherent topics to a 2 topic model
  K mean methods to find best performing model

rephrase 1c), instead of comparing the topics, phrase it in a way such that I can use the topic model to determine red list category for species with literature but not classified
