import re
import sys
import ast
import json
import nltk
import spacy
import stanza
import wikipedia
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from laserembeddings import Laser

laser = Laser()

def get_spacy_model(languange):
    '''
    Method returns the pretrained language model for required language.
    Language is set in the ISO 639-1 format.
    '''

    models = {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'de': 'de_core_news_sm',
              'pt': 'pt_core_news_sm', 'es': 'es_core_news_sm', 'ca': 'ca_core_news_sm',
              'ja': 'ja_core_news_md'}
    if languange in models:
        return models[languange]
    else:
        return 'xx_ent_wiki_sm'

def num_loc_texts(texts, language):
    '''
    Method returns the preprocessed texts without numbers, numericals and locations.
    Also all data without platform trash (urls, mentions, #)
    '''

    spacy_nlp = spacy.load(get_spacy_model(language))
    nlp = stanza.Pipeline(language)

    loc_for_replace = []
    numbers_for_replace = []

    for text in texts:
        doc = spacy_nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                loc_for_replace.append(ent)
            elif ent.label_ == 'LOC':
                loc_for_replace.append(ent)

        doc = nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'NUM':
                    numbers_for_replace.append(word)

        #for token in row:
        #    if token.pos_ == 'NUM':
        #        numbers_for_replace.append(token)

    #text preprocessing
    preprocessed_texts = []
    numloc_texts = []
    count_num, count_loc = 0, 0
    for text in texts:
        # hashtags -> words, URLs -> URL and mentions -> USER
        text = re.sub('#', '', text)
        text = re.sub('((www\.[\\s]+)|(https?://[^\\s]+))', 'URL', text)
        text = re.sub('@[A-Za-z0-9_-]+', 'USER', text)
        text = re.sub('RT @[A-Za-z0-9_-]+:', 'USER', text)

        text = re.sub('\_', ' ', text) # _
        text = re.sub('\!', ' ', text) # !
        text = re.sub('\?', ' ', text) # ?
        text = re.sub('\W', ' ', text) # symbols
        text = re.sub('\_', ' ', text) # _
        text = re.sub('[\s]+', ' ', text) # spaces
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text) # remove spaces between numbers
        preprocessed_texts.append(text)

        separate_text = text.split()
        for t in separate_text:
        #numbers and numericals in text => NUMBER
            if str(numbers_for_replace[count_num]) == t:
                #text = text.replace(str(numbers_for_replace[count_num]), '*NUM*')
                text = text.replace(str(numbers_for_replace[count_num]), 'NUMBER')
                if count_num < len(numbers_for_replace) - 1:
                    count_num += 1
        #text = re.sub('[0-9]+', '*NUM*', text)
        text = re.sub('[0-9]+', 'NUMBER', text)

        # locations in text => LOC
        separate_text = text.split()
        for t in separate_text:
            if str(loc_for_replace[count_loc]) == t:
                text = text.replace(str(loc_for_replace[count_loc]), 'LOCATION')
                #text = text.replace(str(loc_for_replace[count_loc]), '*LOC*')
                if count_loc < len(loc_for_replace) - 1:
                    count_loc += 1

        numloc_texts.append(text)

    return preprocessed_texts, numloc_texts

def get_semantic_keyword(keywords):
    '''
    >> keywods (list)
    This fuction exctracts description for keywords from Wikipedia
    '''
    keywords_list = []
    for k in keywords:
        try:
            kw = wikipedia.summary(k, sentences=1, auto_suggest=False, redirect=False)
            keywords_list.append(kw)
        except:
            keywords_list.append(k)
    return keywords_list

def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

def get_text_features(texts, lang):
    '''
    Method returns list of platform features and morphological,
    syntactic, NER features of texts.
    Language is set in the ISO 639-1 format.
    '''
    texts = texts
    lang = lang
    features = []  # list for final features of all texts
    spacy_nlp = spacy.load(get_spacy_model(lang))
    nlp = stanza.Pipeline(lang)

    for text in texts:
        # Preprocessing and extraction platform features
        doc = re.sub('#', '', text)

        urls = len(re.findall(r'((www\.[\s]+)|(https?://[^\s]+))', doc))
        doc = re.sub(r'((www\.[\s]+)|(https?://[^\s]+))', 'URL', doc)

        users = len(re.findall('@[A-Za-z0-9]+', doc))
        doc = re.sub('@[A-Za-z0-9]+', 'USER', doc)

        numbers = len(re.findall('[0-9]+', doc))

        # Extraction morphological and syntactic features
        nouns, verbs, adverbs, adjectives, nsubj = 0, 0, 0, 0, 0
        mod, roots, compounds = 0, 0, 0

        doc = nlp(doc)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'NUM':
                    numbers += 1
                elif word.upos == 'NOUN':
                    nouns += 1
                elif word.upos == 'VERB':
                    verbs += 1
                elif word.upos == 'ADV':
                    adverbs += 1
                elif word.upos == 'ADJ':
                    adjectives += 1

                if word.deprel == "nsubj":
                    nsubj += 1
                elif word.deprel == "nmod":
                    mod += 1
                elif word.deprel == "nummod":
                    mod += 1
                elif word.deprel == "advmod":
                    mod += 1
                elif word.deprel == "root":
                    roots += 1
                elif word.deprel == "compound":
                    compounds += 1

        # Extraction named entities features
        persons, locations, dates, organizations = 0, 0, 0, 0
        doc = spacy_nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                persons += 1
            elif ent.label_ == 'GPE':
                locations += 1
            elif ent.label_ == 'LOC':
                locations += 1
            elif ent.label_ == 'DATE':
                dates += 1
            elif ent.label_ == 'ORG':
                organizations += 1

        doc_features = [urls, users, numbers, nouns, verbs, adverbs,
                        adjectives, nsubj, mod, roots, compounds, persons,
                        locations, dates, organizations]

        #doc_features = [urls, users, numbers, nouns, verbs, adverbs,
        #                adjectives, nsubj, mod, roots, compounds]
        doc_features = np.log(np.asarray(doc_features) + 1)  # normalizing
        features.append(doc_features)

    return features

def get_sim_features(texts, lang):
    sim_features = []

    #text preprocessing
    preprocess_texts, numloc_texts = num_loc_texts(texts, lang)

    # SIMILARITY
    # cousine similarity between text and prototypes
    text_embeddings = laser.embed_sentences(numloc_texts, lang=lang)
    #text_embeddings = laser.embed_sentences(preprocess_texts, lang='ml')
    prototypes_embeddings = laser.embed_sentences(query['prototypes'], lang=lang)
    templates_embeddings = laser.embed_sentences(query['templates'], lang=lang)
    keywords_embeddings = laser.embed_sentences(get_semantic_keyword(query['keywords']), lang=lang)
    for text_emb in text_embeddings:
        # PROTOTYPES-TEXT SIMILARITY
        prorotypes_similaruty_list = []
        for p_emb in prototypes_embeddings:
            prorotypes_similaruty_list.append(get_cosine_similarity(p_emb, text_emb))
        #Avg similarity between the doc and prototypes
        avg_text_prototype_similarity = sum(prorotypes_similaruty_list)/len(prorotypes_similaruty_list)
        #Max similarity between the doc and prototypes
        max_text_prototype_similarity = max(prorotypes_similaruty_list)

        # TEMPLATES-TEXT SIMILARITY
        templates_similaruty_list = []
        for temp_emb in templates_embeddings:
            templates_similaruty_list.append(get_cosine_similarity(temp_emb, text_emb))
        #Avg similarity between the doc and templates
        avg_text_template_similarity = sum(templates_similaruty_list)/len(templates_similaruty_list)
        #Max similarity between the doc and templates
        max_text_template_similarity = max(templates_similaruty_list)

        # KEYWORDS-TEXT SIMILARITY
        keywords_similaruty_list = []
        for key_emb in keywords_embeddings:
            keywords_similaruty_list.append(get_cosine_similarity(key_emb, text_emb))
        #Avg similarity between the doc and keywords
        avg_text_keyword_similarity = sum(keywords_similaruty_list)/len(keywords_similaruty_list)
        #Max similarity between the doc and keywords
        max_text_keyword_similarity = max(keywords_similaruty_list)

        s_features = [avg_text_prototype_similarity, max_text_prototype_similarity,
                                avg_text_template_similarity, max_text_template_similarity,
                                avg_text_keyword_similarity, max_text_keyword_similarity]


        s_features = np.log(np.asarray(s_features) + 1)  # normalizing
        sim_features.append(s_features)


    return sim_features


def get_laser_features(texts, lang):
     return laser.embed_sentences(texts, lang=lang)



def get_query(dataset, category):
    # reading queries from file

    crisis_queries = []

    with open(query_path) as f:
        for json_obj in f:
            query_dict = json.loads(json_obj)
            crisis_queries.append(query_dict)

    for q in crisis_queries:
        if q["dataset"] == dataset and q["category"] == category:
            return q






# example: features_extraction.py 'taal_out_en.csv' 'Taal' 'Danger' 'en'

#data_path = 'taal_out_en.csv'
#lang = 'en'
#query_dataset = 'Taal'
#query_category = 'Danger'

data_path = sys.argv[1]
query_dataset = sys.argv[2]
query_category = sys.argv[3]
lang = sys.argv[4]

query_path = 'data/queries.json'

data = pd.read_csv(data_path)
query = get_query(query_dataset, query_category)


# getting features
text = get_text_features(data['text'], lang)
sim = get_sim_features(data['text'], lang)
laser = get_laser_features(data['text'], lang)

# saving features in files
pd.DataFrame(text).to_csv(query_dataset + '_' + lang + '_text_features.csv')
pd.DataFrame(sim).to_csv(query_dataset + '_' + lang + '_sim_' + query_category + '_features.csv')
pd.DataFrame(laser).to_csv(query_dataset + '_' + lang + '_laser_features.csv')
