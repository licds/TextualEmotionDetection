import csv
from lib2to3.pgen2.pgen import DFAState
from os import stat
from tokenize import tokenize
import nltk
import string
from numpy import number
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from textblob import TextBlob
from nltk.stem import PorterStemmer
import pickle
import math
from collections import Counter

NEUTRAL = 0.4

def main():
    df = pd.read_pickle('cleaned_train.pkl')
    df2 = pd.read_pickle('cleaned_test.pkl')
    dict = load_dict("dict")
    pdict = bayes(df, dict)
    word_IDF = pd.read_pickle('word_IDF.pkl')
    tf = pd.read_pickle('tf.pkl')

    print(test(df2, pdict, word_IDF))
    print(test(df2, pdict, tf))
    print(test(df2, pdict, 0))

# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

# Take emotion sentiment from each word of a sentence and average 
# it to represent the sentiment for the sentence
def classify(sentence, dict, weight):
    sentence_len = len(sentence)
    count = Counter(sentence)
    emotion = [0, 0, 0, 0, 0, 0]
    if sentence_len == 0:
        return
    for word in sentence:
        if word in dict.keys():
            for i in range(6):
                if weight == 0:
                    emotion[i] += dict[word][i]
                else:
                    emotion[i] += dict[word][i]*(count[word]/sentence_len*weight[word][i])
    #for i in range(len(emotion)):
     #   emotion[i] = emotion[i]/size
    return emotion

# Print sentence sentiment
def print_emotion(emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
    for i in range(len(emotion)):
        if emotion[i] >= NEUTRAL:
            print(emo_l[i])
            print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
            return
    #print("Neutral")
    print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
    
# Build a test trial with accuracy
def test(df, dict, weight):
    accuracy = 0
    index = 0
    for text in df['Text']:
        emotion = classify(text, dict, weight)
        emo_index = emotion.index(max(emotion)) # Take most probable emotion
        if emo_index == df.iloc[index, 1]:
            accuracy += 1
        index += 1
    accuracy = accuracy/len(df)
    return accuracy

def bayes(df, dict):
    #size = len(df)
    words = df.Text.str.len().sum()
    emo_words = []
    emo_p = []

    for i in range(6):
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
        emo_p.append(emo_words[i]/words)
    
    pdict = defaultdict(list)
    for word in dict.keys():
        n_word = len(dict[word])
        word_p = n_word/words
        for i in range(6):
            emo_word=dict[word].count(i)
            pdict[word].append(emo_word/emo_words[0]*emo_p[0]/word_p)
    return pdict

if __name__ == "__main__":
    main()
