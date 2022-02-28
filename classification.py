import re
import sys
import ast
import stanza
import spacy
import numpy as np
import pandas as pd

from laserembeddings import Laser
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, BatchNormalization


langs = {'australia': ['en', 'es', 'fr', 'ja', 'id'],
         'fukushima': ['en', 'es', 'fr', 'ja', 'id'],
         'gloria': ['en', 'es', 'fr', 'ca'],
         'taal': ['en', 'es', 'fr', 'tg', 'pt'],
         'zagreb': ['en', 'es', 'fr', 'hr', 'de']}

def del_2_cat(dataset):
    mod_data = []
    for l in dataset:
        if l == 1:
            mod_data.append(int(1))
        else:
            mod_data.append(int(0))
    return mod_data

def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def get_experiment_data(crisis_name, test_lang):

    # create two train and test datasets with text features
    df_text_train = []

    for lang in langs[crisis_name]:
        if lang == test_lang:
            stat_file = crisis_name + '_' + lang + '_text_features.csv'
            df_text_test = pd.read_csv(stat_file)
        else:
            stat_file = crisis_name + '_' + lang + '_text_features.csv'
            df = pd.read_csv(stat_file)
            df_text_train.append(df)

    df_text_train = pd.concat(df_text_train)
    df_text_train = df_text_train.drop(columns=['Unnamed: 0'])

    df_text_test = df_text_test.drop(columns=['Unnamed: 0'])


    # create two train and test datasets with texts features
    df_emb_train = []

    for lang in langs[crisis_name]:
        if lang == test_lang:
            filename =  crisis_name + '_' + lang + '_laser_features.csv'
            df_emb_test = pd.read_csv(filename)
            df_label_test = df['labels']
        else:
            filename = crisis_name + '_' + lang + '_laser_features.csv'
            df = pd.read_csv(filename)
            df_emb_train.append(df)
            df_label_train.append(df['labels'])

    df_emb_train = np.concatenate(df_emb_train)


    #  read train and test labels
    df_label_train = []

    for lang in langs[crisis_name]:
        if lang == test_lang:
            filename =  crisis_name + '_' + lang + '.csv'
            df_emb_test = pd.read_csv(filename)
            df_label_test = df['labels']
        else:
            filename = crisis_name + '_' + lang + '.csv'
            df = pd.read_csv(filename)
            df_emb_train.append(df)
            df_label_train.append(df['labels'])

    df_label_train = pd.concat(df_label_train)

    return df_text_train, df_emb_train, df_label_train, df_text_test, df_emb_test, df_label_test


# example: python3 classification.py 'australia' 'en'

#dataset = 'australia'
#test_lang = 'en'

# for executing the script necessary three files for every language in dataset prepared with use of feature_extraction.py
# example for Taal volcano eruption set in English: taal_en.csv, taal_en_laser_features.csv, taal_en_text_features.to_csv

dataset = sys.argv[1]
test_lang = sys.argv[2]

train_text_feat, train_emb, train_labels, test_text_feat, test_emb, test_labels = get_experiment_data(dataset, test_lang)


TRANSFORMER_DIM = 1024 #laser

# - - - - - TRAIN FEATURES - - - - -
X1_text = tf.reshape(train_text_feat, [-1, 1, 15])
X1_laser = tf.reshape(train_emb, [-1, 1, TRANSFORMER_DIM])

Y1 = to_categorical(del_2_cat(train_labels), 2)
Y1_reshaped = tf.reshape(Y1, [-1, 1, 2])

print('Train data shapes:', X1_text.shape, X1_laser.shape, Y1_reshaped.shape)

# - - - - - DEV FEATURES - - - - -
X2_text = tf.reshape(test_text_feat, [-1, 1, 15])
X2_laser = tf.reshape(test_emb, [-1, 1, TRANSFORMER_DIM])

Y2 = to_categorical(del_2_cat(test_labels), 2)
Y2_reshaped = tf.reshape(Y2, [-1, 1, 2])

print('Dev data shapes:', X2_text.shape, X2_laser.shape, Y2_reshaped.shape)

# Defining the model paramaters.

inputA = Input(shape=(1, TRANSFORMER_DIM, ))
inputB = Input(shape=(1, 15, ))

# the first branch operates on the transformer embeddings
x = LSTM(TRANSFORMER_DIM, input_shape=(1, TRANSFORMER_DIM), return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(inputA)
x = Dense(TRANSFORMER_DIM,activation='relu')(x)
x = Dense(256, activation="sigmoid")(x)
x = Dropout(0.5)(x)
x = Dense(128,activation='sigmoid')(x)
x_model = Model(inputs=inputA, outputs=x)

# the second branch operates on the topics features
y = Dense(128, activation="relu")(inputB)
y = Dense(24, activation = "relu")(y)
y_model = Model(inputs = inputB, outputs = y)

# combine the output of the three branches
combined = Concatenate()([x_model.output, y_model.output])

# apply a FC layer and then a regression prediction on the combined outputs
z = BatchNormalization()(combined)
z1 = Dense(2, activation="softmax")(z)

# our model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[x_model.inputs, y_model.inputs], outputs=z1)

model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'), f1, tf.keras.metrics.AUC(name='auc')])

model.fit([X1_laser, X1_text], Y1_reshaped,
          validation_data=([X2_laser, X2_text], Y2_reshaped),
          batch_size=100, epochs=20)
