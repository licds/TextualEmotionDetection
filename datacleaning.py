import csv
from tokenize import tokenize
import nltk
import pandas as pd
from collections import defaultdict
from textblob import TextBlob
from nltk.stem import PorterStemmer
import pickle

def main():
    #========================== ONlY UNCOMMENT IF YOU WANT TO MODIFY DATASET ==========================
    #df = initdf("Data/training.csv")
    #df.to_pickle("cleaned_train.pkl")  
    #df2 = initdf("Data/test.csv")
    #df2.to_pickle("cleaned_test.pkl") 
    #df3 = initdf("Data/validation.csv")
    #df3.to_pickle("cleaned_valid.pkl")

    #========================== UNCOMMENT THESE IF YOU WANT TO MODIFY DICT ==========================
    #df_train = pd.read_pickle('cleaned_train.pkl')
    #df_test = pd.read_pickle('cleaned_test.pkl')
    #df_valid = pd.read_pickle('cleaned_valid.pkl')
    #dict_train = dict_init(df_train)
    #dict_test = dict_init(df_test)
    #dict_valid = dict_init(df_valid)
    #save_dict(dict_train, "dict_train")
    #save_dict(dict_test, "dict_test")
    #save_dict(dict_valid, "dict_valid")


#============================== DATASET MANIPULATION ==============================
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
# Init a dictionary that holds the occurence of word in terms of emotion
def dict_init(df):
    dict1 = defaultdict(list)
    emotion = list(df.loc[:,"Emotion"])
    for row in range(0,len(df.index)):
        temp = df.iloc[row,0]
        for word in temp:
            dict1[word].append(emotion[row])
    return dict1

# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

#============================== OTHERS ==============================
def interactive_test():
    sentence = input("Enter the sentence you want to classify!!!:")

if __name__ == "__main__":
    main()
