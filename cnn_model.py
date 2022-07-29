import pandas as pd
import numpy as np
from datacleaning import save_dict, load_dict
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict

def main():
    train = pd.read_pickle('cleaned_train.pkl')
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.oov_token = '<oovToken>'
    tokenizer.fit_on_texts(train.Text)
    vocab = tokenizer.word_index
    vocabCount = len(vocab) + 1

    x_Train = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(train.Text.to_numpy()), padding='pre')
    y_Train = train.Emotion.to_numpy()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocabCount+1, output_dim=64, input_length=36))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_Train, y_Train, epochs=5, shuffle=True)

if __name__ == "__main__":
    main()