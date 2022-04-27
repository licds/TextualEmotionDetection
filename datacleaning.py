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

def main():
    df = initdf("Data/training.csv")
    df.to_pickle("cleaned_train.pkl")  
    df2 = initdf("Data/test.csv")
    df2.to_pickle("cleaned_test.pkl")
    dict = first_convert(df)
    save_dict(dict, "dict")

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
