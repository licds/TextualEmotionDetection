import csv
from lib2to3.pgen2.pgen import DFAState
from tokenize import tokenize
import nltk
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from textblob import TextBlob
from nltk.stem import PorterStemmer
#from textblob.np_extractors import ConllExtractor
import pickle


#save a dictionary into a file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

#load a dictionary from a file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

NEUTRAL = 0.4

def main():
    df = pd.read_csv('cleaned_train.csv')
    df2 = pd.read_csv('cleaned_test.csv')
    dict = load_dict("dict")
    pdict = bayes(df, dict)
    print(test(df2, pdict))

def classify(sentence, dict):
    emotion = [0, 0, 0, 0, 0, 0]
    size = len(sentence)
    if size == 0:
        return
    for word in sentence:
        if word in dict.keys():
            emotion[0] += dict[word][0]
            emotion[1] += dict[word][1]
            emotion[2] += dict[word][2]
            emotion[3] += dict[word][3]
            emotion[4] += dict[word][4]
            emotion[5] += dict[word][5]
    for i in range(len(emotion)):
        emotion[i] = emotion[i]/size
    return emotion

def print_emotion(emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
    for i in range(len(emotion)):
        if emotion[i] >= NEUTRAL:
            print(emo_l[i])
            print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
            return
    print("Neutral")
    print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
    
def test(df, dict):
    accuracy = 0
    index = 0
    for text in df['Text']:
        emotion = classify(text, dict)
        emo_index = emotion.index(max(emotion))
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

def initdf(csv_file):
    f=open(csv_file)
    rows=list(csv.reader(f))
    rows.pop(0)
    for row in rows:
        row[0]=list(tokenize(row[0]))
    df = pd.DataFrame(rows, columns=['Text', 'Emotion'])
    df['Emotion'] = df['Emotion'].astype(int)
    return df

def tokenize(text):
    #blob = TextBlob(text, np_extractor=ConllExtractor()).noun_phrases
    #print(blob)
    ps =PorterStemmer()
    tokens=nltk.word_tokenize(text)

    new=[]
    for word in tokens:
        word=word.lower()
        word=ps.stem(word)
        if word not in nltk.corpus.stopwords.words("english"):
            new.append(word)
    return new

def first_convert(df):
    dict1 = defaultdict(list)
    emotion = list(df.loc[:,"Emotion"])
    for row in range(0,len(df.index)):
        temp = df.iloc[row,0]
        for word in temp:
            dict1[word].append(emotion[row])
    return dict1

if __name__ == "__main__":
    main()
