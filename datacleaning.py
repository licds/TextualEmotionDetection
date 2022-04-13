import csv
import nltk
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

def main():
    f=open("Data/test.csv")

    rows=list(csv.reader(f))
    rows.pop(0)
    for row in rows:
        row[0]=list(tokenize(row[0]))
    
    df = pd.DataFrame(rows, columns=['Text', 'Emotion'])
    df['Emotion'] = df['Emotion'].astype(int)

def tokenize(text):
    #blob = TextBlob(text, np_extractor=ConllExtractor()).noun_phrases
    #print(blob)
    tokens=nltk.word_tokenize(text)
    new=[]
    for word in tokens:
        word=word.lower()
        word=WordNetLemmatizer().lemmatize(word)
        if word not in nltk.corpus.stopwords.words("english"):
            new.append(word)
    return new

if __name__ == "__main__":
    main()
