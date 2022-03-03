import re
import nltk
import operator
import pandas as pd
from laserembeddings import Laser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.models import Model, Sequential

# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# for ids extraction
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

laser = Laser()

# f1 evaluation
def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
tokenizer = AutoTokenizer.from_pretrained("t5-large")



# for executing the script necessary five four for every language in dataset prepared with use of feature_extraction.py
# and don't forget files with query-similatities features and also in it necessary to have translation of tweets from languages to English (translated texts should be stored in column 'en_texts')


# Train EN —> Test All other
# example: python3 sum.py 'fukushima' 'Casualties'
# dataset = 'fukushima'
# query = 'Weather'

dataset = sys.argv[1]
query = sys.argv[2]



langs = {'australia': ['en', 'es', 'fr', 'ja', 'id'],
         'fukushima': ['en', 'es', 'fr', 'ja', 'id'],
         'gloria': ['en', 'es', 'fr', 'ca'],
         'taal': ['en', 'es', 'fr', 'tg', 'pt'],
         'zagreb': ['en', 'es', 'fr', 'hr', 'de']}

events = {'australia': ['australia_bushfires', 'ab'],
         'fukushima': ['fukushima_earthquake', 'fe'],
         'gloria': ['gloria_storm', 'gs'],
         'taal': ['taal_eruption', 'te'],
         'zagreb': ['zagreb_earthquake', 'ze']}

# Reading data

languages = langs[dataset]

labels1 = pd.read_csv(dataset + languages[0] + '.csv')
n = labels1[labels1['labels'] == 1]

train_text = pd.read_csv(dataset + '_'+ languages[0] + '_text_features.csv').drop(columns = 'Unnamed: 0').iloc[n.index]
train_laser = pd.read_csv(dataset + '_'+ languages[0] + '_laser_features.csv').drop(columns = 'Unnamed: 0').iloc[n.index]
train_sim = pd.read_csv(dataset + '_'+ languages[0] + '_sim_' + query + '_features.csv').drop(columns = 'Unnamed: 0').iloc[n.index]
train_labels = pd.read_csv(dataset + '_'+ languages[0] + '.csv')[query].iloc[n.index]

test_text = []; test_laser =[]; test_sim = []; test_labels = []; final_texts = []; tweet_ids= []

for num in range(len(languages))[1:]:
    labels2 = pd.read_csv(dataset + '_' + languages[num] + '.csv')
    m = labels2[labels2['labels'] == 1]

    test_text.append(pd.read_csv(dataset + '_'+ languages[num] + '_text_features.csv').drop(columns = 'Unnamed: 0').iloc[m.index])
    test_laser.append(pd.read_csv(dataset + '_'+ languages[num] + '_laser_features.csv').drop(columns = 'Unnamed: 0').iloc[m.index])
    test_sim.append(pd.read_csv(dataset + '_'+ languages[num] + '_sim_' + query + '_features.csv').drop(columns = 'Unnamed: 0').iloc[m.index])
    test_labels.append(pd.read_csv(dataset + '_'+ languages[num] + '.csv')[query].iloc[m.index])
    final_texts.append(pd.read_csv(dataset + '_'+ languages[num] + '.csv')['en_texts'].iloc[m.index])
    tweet_ids.append(pd.read_csv('/data/tweets/' + events[dataset][0] + '/' events[dataset][1] + '_'+ languages[num] + '.csv')['id'].iloc[m.index])

test_text = pd.concat(test_text, ignore_index=True)
test_laser = pd.concat(test_laser, ignore_index=True)
test_sim = pd.concat(test_sim, ignore_index=True)
test_labels = pd.concat(test_labels, ignore_index=True)
final_texts = pd.concat(final_texts, ignore_index=True)
tweet_ids = pd.concat(tweet_ids, ignore_index=True)



# - - - - - TRAIN FEATURES - - - - -
X1_text = tf.reshape(train_text, [-1, 1, 15])
X1_laser = tf.reshape(train_laser, [-1, 1, 1024])
X1_sim = tf.reshape(train_sim, [-1, 1, 6])

Y1 = to_categorical(train_labels, 2)
Y1_reshaped = tf.reshape(Y1, [-1, 1, 2])

print('Train data shapes:', X1_text.shape, X1_laser.shape, X1_sim.shape, Y1_reshaped.shape)

# - - - - - TEST FEATURES - - - - -
X2_text = tf.reshape(test_text, [-1, 1, 15])
X2_laser = tf.reshape(test_laser, [-1, 1, 1024])
X2_sim = tf.reshape(test_sim, [-1, 1, 6])

Y2 = to_categorical(test_labels, 2)
Y2_reshaped = tf.reshape(Y2, [-1, 1, 2])

print('Test data shapes:', X2_text.shape, X2_laser.shape, X2_sim.shape, Y2_reshaped.shape)



# Defining the model paramaters.

inputA = Input(shape=(1, 15, ))
inputB = Input(shape=(1, 1024, ))
inputC = Input(shape=(1, 6, ))

# the first branch operates on the topics features
x = Dense(128, activation="relu")(inputA)
x = Dense(24, activation = "relu")(x)
x_model = Model(inputs = inputA, outputs = x)

# the first branch operates on the transformer embeddings
y = LSTM(1024, input_shape=(1, 1024), return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(inputB)
y = Dense(1024,activation='relu')(y)
y = Dense(256, activation="sigmoid")(y)
y = Dropout(0.5)(y)
y = Dense(128,activation='sigmoid')(y)
y_model = Model(inputs=inputB, outputs=y)

# the third branch operates on the topics features
z = Dense(128, activation="relu")(inputC)
z = Dense(24, activation = "relu")(z)
z_model = Model(inputs = inputC, outputs = z)

# combine the output of the three branches
combined = Concatenate()([x_model.output, y_model.output, z_model.output])

# apply a FC layer and then a regression prediction on the combined outputs
combo = BatchNormalization()(combined)
combo1 = Dense(2, activation="softmax")(combo)

# our model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[x_model.inputs, y_model.inputs, z_model.inputs], outputs=combo1)

#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'), f1, tf.keras.metrics.AUC(name='auc')])

model.fit([X1_text, X1_laser, X1_sim], Y1_reshaped,
          validation_data=([X2_text, X2_laser, X2_sim], Y2_reshaped), epochs=20)



def dublicated_k(entities):
    # funktion searching dublicates and drop them
    dublicates = []
    for num1 in range(len(entities)):
        for num2 in range(len(entities)):
            if entities[num1] != entities[num2]:
                if cosine_similarity(X2_laser[entities[num1]], X2_laser[entities[num2]]) > 0.9:
                    dublicates.append(num2)

    dublicates_wo_dublicates = list(set(dublicates))

    # drop dublicates from entities
    entities_wo_dublicates = [item for item in entities if item not in dublicates_wo_dublicates]

    return(entities_wo_dublicates[:100])




def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

def extract_ids(f_summary, ids, tweets, n):
    # - - - - - TEST IDs EXTRACTOR - - - - -
    summary_ngrams = get_ngrams(f_summary, n)

    sim_counter = {}
    for num in range(len(tweets)):
        count = 0
        for ngram in summary_ngrams:
            if ngram in get_ngrams(tweets[num], n):
                count += 1

        if count > 0:
            sim_counter[ids[num]] = count # add ids of tweet in l

    sorted_sims = dict(sorted(sim_counter.items(), key=operator.itemgetter(1),reverse=True))

    # http://twitter.com/anyuser/status/
    top_ids = list(sorted_sims.keys())[:3] # top-3 similar tweets

    ids_string = ''
    for id in top_ids:
        ids_string += 'http://twitter.com/anyuser/status/'+ str(id) + '\n'

    return ids_string



param_K = 200
# create dictionary for storing number of row and probability
higher_prob = {}
for num, prob in enumerate(model.predict([X2_text, X2_laser, X2_sim])):
    higher_prob[num] = prob[0][1]

# sort dictionary by probability decreasing
sorted_output = sorted(higher_prob.items(),key=operator.itemgetter(1),reverse=True) # sorting dictionary and choose the top of probable answers

# get list with top K entities
ents = [ent[0] for ent in sorted_output[:param_K]]

# diversified K-100
#ents = dublicated_k(ents)

# create string (text_4sum) with the most relevant tweets
texts_for_sum = []

ids_for_sum = [] # store ids of tweets for using them for summary evaluation interface

for ent in ents:
    #print(final_texts[ent])
    ids_for_sum.append(tweet_ids[ent])
    text = re.sub('#', '', final_texts[ent])
    text = re.sub(r'((www\.[\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub('@ ?[A-Za-z0-9]+', '', text)
    text = re.sub('^[A-Za-z0-9]+', '', text)
    text = re.sub('^ ?RT', '', text)
    text = re.sub('^ ?: ', '', text)
    texts_for_sum.append(text)

def get_cluster_number(text_dataset):
    coefficients = []
    for num in range(2, 6):
        # define the model
        clust_model = KMeans(n_clusters=num)
        # fit the model
        clust_model.fit(text_dataset)
        # assign a cluster to each example
        yhat = clust_model.predict(text_dataset)
        #print(silhouette_score(text_dataset, yhat))
        coefficients.append(silhouette_score(text_dataset, yhat))

    max_value = max(coefficients)
    max_index = coefficients.index(max_value)
    return(max_index+2)

# clusterisation for deversification of ranked messages

vectorizer = TfidfVectorizer(min_df=2)
new_texts = vectorizer.fit_transform(texts_for_sum)

new_texts = new_texts.toarray() # transform data for use with sklearn

#define number of clusters
number_of_clusters = get_cluster_number(new_texts)

# define the model
clust_model = KMeans(n_clusters=number_of_clusters, random_state=42)
# fit the model
clust_model.fit(new_texts)


# create ordering list based on frequency
nums = {}
for n in clust_model.labels_:
    if n not in nums:
        nums[n] = 1
    else:
        nums[n] += 1
nums = dict(sorted(nums.items(), key=lambda item: item[1], reverse=True))
ordered_list = list(nums.keys())



# - - - - - - - - - - - - - - - - - - - -

# Ordered summarization
summary = ''

for num in ordered_list:
    cluster_text = ''

    cluster_ids = [] # store ids for extracting original tweets
    cluster_tweets = [] # store texts of tweets for extracting original tweets

    for n, cluster in enumerate(clust_model.labels_):
        if cluster == num:
            cluster_text += texts_for_sum[n]
            cluster_ids.append(ids_for_sum[n])
            cluster_tweets.append(texts_for_sum[n])

    if len(summary) < 600:

        inputs = tokenizer("summarize: " + cluster_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = sum_model.generate(inputs["input_ids"], max_length=100, min_length=40, length_penalty=4.0, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(outputs[0])[6:-4]

        extract_ids(generated_summary, cluster_ids, cluster_tweets, 3)

        summary += generated_summary + '\n' + extract_ids(generated_summary, cluster_ids, cluster_tweets, 3)

print(summary)
