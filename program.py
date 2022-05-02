import csv
from unittest.mock import sentinel
import nltk
import pickle
import numpy as np
import pandas as pd
from tokenize import tokenize
from textblob import TextBlob
from collections import Counter
from nltk.stem import PorterStemmer
from collections import defaultdict

NEUTRAL = 0.4

def main():
    tf_prog = pd.read_pickle('tf_prog.pkl')
    pdict_prog = load_dict("pdict_prog")
    exit = 0
    while exit != 1:
        sentence = input("Enter the sentence you want to classify: ")
        if sentence == "exit":
            exit = 1
            break
        sentence = list(tokenize(sentence))
        emotion = classify(sentence, pdict_prog, tf_prog)
        emo_index = predict(sentence, pdict_prog, tf_prog)
        print_emotion(emo_index, emotion)
    return 0

#============================== PREDICTION MODULE ==============================
# Take emotion sentiment from each word of a sentence and average 
# it to represent the sentiment for the sentence
def classify(sentence, dict, weight):
    sentence_len = len(sentence)
    count = Counter(sentence)
    emotion = [1]*6
    counter = 0
    if sentence_len == 0:
        return
    for word in list(set(sentence)):
        if word in dict.keys():
            counter+=1
            for i in range(6):
                if weight == 0:
                    emotion[i] += dict[word][i]
                else:
                    emotion[i] += dict[word][i]*(count[word]/sentence_len*weight[word][i])
    return emotion
    
# Build a test trial with accuracy
def predict(sentence, dict, weight):
    emotion = classify(sentence, dict, weight)
    emo_index = emotion.index(max(emotion)) 
    return emo_index

# Print sentence sentiment
def print_emotion(emo_index, emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
    print("The emotion for this sentence is:", emo_l[emo_index])
    print("Sad: ", round((emotion[0]-1)*10000, 4), "Joy: ", round((emotion[1]-1)*10000, 4), "Love: ", round((emotion[2]-1)*10000, 4), "Anger: ", round((emotion[3]-1)*10000, 4), "Fear: ", round((emotion[4]-1)*10000, 4), "Surprise: ", round((emotion[5]-1)*10000, 4))

#============================== USER INPUT MANIPULATION ==============================
# Implement for initdf, delete words and make them list of strings
def tokenize(text):
    ps = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    new = []
    for word in tokens:
        word = word.lower()
        word = ps.stem(word)
        if word not in nltk.corpus.stopwords.words("english"):
            new.append(word)
    return new

#============================== DICTIONARY MANIPULATION ==============================
# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

if __name__ == "__main__":
    main()
