import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

max_features = 200000
maxlen = 80
batch_size = 32

# read train and test files
train = pd.read_csv('train.tsv', sep='\t', header=0)
test = pd.read_csv('test.tsv', sep='\t', header=0)


# combining the data into a single unit for processing of features
features_train = train['Phrase']
labels_train = train['Sentiment']
features_test = test['Phrase']
combined = features_train.append(features_test).values

train_length = len(features_train.values)
test_length = len(features_test.values)

combined_features=[]


# Tokenize, stem and remove stopwords from the combined data
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation

# stopwords
nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')))
punc=list(set(punctuation))
stop_words.extend(punc)
stop_words.extend(["'s", "'d", "'m"])
print(stop_words)

for x in combined:
    x=word_tokenize(x)
    stemmer=SnowballStemmer('english')
    x=[(stemmer.stem(i)).lower() for i in x]
    x=[i for i in x if x not in stop_words]
    combined_features.append(x)


# mapping frequencies with words
from gensim import corpora
dictionary = corpora.Dictionary(combined_features)
print(dictionary)

id=[]
for x in combined_features:
    temp = [dictionary.token2id[j] for j in x]
    id.append(temp)


# Creating MLP
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.utils import np_utils
from keras.preprocessing import sequence

# using gpu to increase computation speed
with tf.device('/gpu:0'):
    model=Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))

    # padding the input to ensure a fixed size input to the network
    x_train=sequence.pad_sequences(np.array(id[:train_length]))
    x_test=sequence.pad_sequences(np.array(id[train_length+1:]))

    # one hot encoding
    y_train=np_utils.to_categorical(labels_train)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train,batch_size=batch_size,epochs=10,validation_split=0.1)

    preds = model.predict_classes(x_test, verbose=0)

    def write_preds(preds, fname):
        pd.DataFrame({"PhraseID": test['PhraseId'],"SentenceId": test['SentenceId'], "Sentiment": preds}).to_csv(fname, index=False, header=True)

    write_preds(preds, "result-1.csv")
