IUCN redlist is the leading source of extinction risk text (IUCN, iucn.org)

Brief description:
    founded in 1964 by IUCN which itself was founded in 1948. Highlights species facing the highest risk of extinction, which could lead to increased awareness and attention to the conservation of the species.

How I'm using the texts:
    The text summaries are conveniently categorized into seven parts, justification of the assessment (rationale), habitat and ecology information (habitat), threats information (threats), population information (population), geographic range information (range), use and trade information (useTrade) and conservation actions information (conservationActions).

    First, cleaned data, describe methods

    Prepare data for LDA model, remove stopwords and punctuations, lemmatize, apply bigrams (then trigrams), tokenize.

    Create corpora from the tokens, filter out extremes (<0.1%, >80%), compactify (assign new word ids to all words and shrink any gaps resulting from the filter).

    Create bag of words corpus.

    Two approaches here on. 
    
    First, train one model on all the 7 categories combined. Review the generated topics add revise stopwords list to include topic general words (contextually non-informative). Examples being directions, time and geographic names.

    Retrain models, interpret topics, calculate coherence using u_mass metric, visualize using word cloud (demonstrate using <5 topic model, less complicated)

    Second, train category specific model (7 models in total), steps same as above.

    Interpret results:

        Make sense of topics, relate to corpus

        explore possibility of habitat assignment differing from assessment information 
        e.g.:
        brown bear's assessment information: forest, shrubland, grassland, wetlands (inland), desert, artificial/terrestrial.

        text summary: dry asian steppes to arctic shrublands to temperate rain forests. Coastal areas, deciduous and mixed forests of mountain ranges. Boreal forests. Dry, desert-like areas, apline and sub-alpine areas.

### The value of the IUCN Red List for conservation https://doi.org/10.1016/j.tree.2005.10.010

    About how the red list is misunderstood, and explains how it is actually being used

    Paper talks about how the red list is misunderstood, and explains how it is actually being used. Discusses how red list is used to inform conservation of species. This point could become the launch pad for my narrative, eg took inspiration and then try to go beyond direct interpretation of the texts to go beyond more than just what and how species need protecting.

    "Real understand of the advancement of the red list has lagged behind its increased profile" (https://doi.org/10.1016/j.tree.2005.10.010 , https://doi.org/10.1016/S0169-5347(03)00090-9, https://doi.org/10.1016/S0169-5347(02)02614-9)

    Submissions to the Red List now require the rationale for listing, supported by data on range size, population size and trend, distribution, habitat preferences, altitude, threats and conservation actions in place or needed. Many of these parameters are coded in standardized ‘authority files’ that enable comparative analyses across taxa (IUCN, iucnredlist.org)

    A major contribution of the Red List assessments is the compilation of a rapidly increasing number of digital distribution maps of species (https://doi.org/10.1016/j.tree.2005.10.010). The Red List assessments are, therefore, vehicles for the compilation, synthesis and dissemination of a wealth of species-related data that would otherwise remain scattered and inaccessible to decision makers (https://pdfs.semanticscholar.org/1907/312c02db99c7adc1394e292045430c7ff8ff.pdf)

    Point on standard methodology for evaluation leads to a consistency in the dataset, increases the ease of analysis

## Example of going beyond the direct interpretation of texts

### Many IUCN red list species have names that evoke negative emotions https://doi.org/10.1080/10871209.2020.1753132

    About application of sentiment analysis on species common names to target stategic name changes to increase the engagement towards conservation efforts for the selected species

### iucn_sim: a new program to simulate future extinctions based on IUCN threat status https://doi.org/10.1111/ecog.05110

    Developed a software to simulate future extinctions based on IUCN conservation status information with the additional incorporation of:
     generation length information of individual species
     status transition rates estimated from IUCN assessment history

### 