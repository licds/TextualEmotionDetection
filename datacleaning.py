import csv
from tokenize import tokenize
import nltk
import pandas as pd
from collections import defaultdict
from textblob import TextBlob
from nltk.stem import PorterStemmer
#from textblob.np_extractors import ConllExtractor
import pickle

def main():
    df = initdf("Data/training.csv")
    df.to_pickle("cleaned_train.pkl")  
    df2 = initdf("Data/test.csv")
    df2.to_pickle("cleaned_test.pkl")
    df3 = initdf("Data/validation.csv")
    df3.to_pickle("cleaned_valid.pkl")
    #dict = first_convert(df)
    dict_valid = first_convert(df3)
    #save_dict(dict, "dict")
    save_dict(dict_valid, "dict_valid")

# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

# Basic csv file cleaning for train and test
def initdf(csv_file):
    f = open(csv_file)
    rows = list(csv.reader(f))
    rows.pop(0)
    for row in rows:
        row[0] = list(tokenize(row[0]))
    df = pd.DataFrame(rows, columns=['Text', 'Emotion'])
    df['Emotion'] = df['Emotion'].astype(int)
    return df

# Implement for initdf
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

def first_convert(df):
    dict1 = defaultdict(list)
    emotion = list(df.loc[:,"Emotion"])
    for row in range(0,len(df.index)):
        temp = df.iloc[row,0]
        for word in temp:
            dict1[word].append(emotion[row])
    return dict1


def interactive_test():
    sentence = input("Enter the sentence you want to classify!!!:")

if __name__ == "__main__":
    main()
