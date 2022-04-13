import csv
import nltk
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    f = open("Data/test.csv")

    csvreader=csv.reader(f)
    rows=[]
    n=0
    for row in csvreader:
        n+=1
        if n==1:
            continue
        rows.append(row)
    for row in rows:
        l=tokenize(row[0])
        row[0]=l
    
    df = todataframe(rows)
    
def tokenize(document):
    l=[]
    tokens = nltk.word_tokenize(document)
    for word in tokens:
        word=word.lower()
        word=WordNetLemmatizer().lemmatize(word)
        if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words("english")) :
            l.append(word)
    return l

def todataframe(rawdf):
    df = pd.DataFrame(rawdf, columns =["Text","Emotion"])
    return df
if __name__ == "__main__":
    main()
