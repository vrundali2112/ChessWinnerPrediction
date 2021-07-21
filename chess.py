import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

data = pd.read_csv('games.csv')

data = data[(data["victory_status"] != "draw")]
data = data.drop(["id","turns","rated","opening_name","opening_ply","increment_code","created_at","last_move_at","black_rating","black_id","white_rating","white_id","victory_status","opening_eco"], axis=1)

#GETTING ALL UNIQUE MOVES
unique_moves = set()
len_unique_moves = len(unique_moves)

for move_list in data["moves"]:
    for move in move_list.split(' '):
        unique_moves.add(move)

max_vocab = len(unique_moves)

moves = np.array(data['moves'])
labels = np.array(data["winner"].map(lambda x: 1 if x=="white" else 0))

#GETTING MAXIMIUM LENGTH OF ITEM IN UNIQUE_MOVES
max_len = 0
for move in moves:
    total = 0
    for item in move.split(' '):
        total +=1
    if total > max_len:
        max_len = total

#TOKENIZATION WITH TENSORFLOW
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(moves)
sequences = tokenizer.texts_to_sequences(moves)
word_index = tokenizer.word_index
model_inputs = pad_sequences(sequences, maxlen=max_len)

#TRAINING AND TESTING DATA
train_inputs, test_inputs, train_labels, test_labels = train_test_split(model_inputs, labels, train_size=0.7, random_state=25)

embedding_dim = 300
inputs = tf.keras.Input(shape=max_len)

embedding = tf.keras.layers.Embedding(input_dim=max_vocab,output_dim=embedding_dim,input_length=max_len)(inputs)
gru = tf.keras.layers.GRU(units=embedding_dim)(embedding)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(gru) 
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',tf.keras.metrics.AUC(name='auc')]
)

batch_size = 32
epochs = 3

model.fit(
    train_inputs,
    train_labels,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],
    verbose=1
)

model.evaluate(test_inputs, test_labels, verbose = 1)

model.save("my_model")
model.save_weights("weights.h5")


