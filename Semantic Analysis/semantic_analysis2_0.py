from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

file_name = "sentiment_train.txt"

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

with open(file_name, 'r') as file:
    for each_line in file:
        try:
            label, sentence = each_line.strip().split('\t')
            
            words = nltk.word_tokenize(sentence.lower())
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
        except ValueError as e:
            print(str(e))
            pass

print(maxlen)
print(len(word_freqs))

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0

with open(file_name, 'r') as file:
    for each_line in file:
        try:
            label, sentence = each_line.strip().split('\t')
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
        except ValueError as e:
            print(str(e))
            pass
    
    
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# preparing training and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,verbose=0, validation_data=(Xtest, ytest))

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE,verbose=0)
print('Test score: %.3f, accuracy: %.3f' % (score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("%.0f\t\t%d\t\t%s" % (ypred, ylabel, sent))

def make_prediction_2(sentence):
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X_new = np.array([seqs])
    X_new = sequence.pad_sequences(X_new, maxlen=MAX_SENTENCE_LENGTH)
    return model.predict(X_new[0].reshape(1,40))[0][0]
    
def make_prediction(sentence_arr):
    i = 0
    X_new = np.empty((len(sentence_arr), ), dtype=list)
    for sentence in sentence_arr:
        words =  nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X_new[i] = seqs
        i += 1
    X_new = sequence.pad_sequences(X_new, maxlen=MAX_SENTENCE_LENGTH)
    result = []
    for each in X_new:
            result.append(model.predict(each.reshape(1, 40))[0][0])
    return result

#print(make_prediction(["Hey, I love you","You are a monster"]))

plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()
