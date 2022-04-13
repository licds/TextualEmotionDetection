import csv
import nltk
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

def main():
    f = open("Data/test.csv")

    rows=list(csv.reader(f))

    for row in rows:
        row[0]=tokenize(row[0])
    
    df = todataframe(rows)
    print(df)
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word=word.lower()
        word=WordNetLemmatizer().lemmatize(word)
        if (word in string.punctuation) and (word in nltk.corpus.stopwords.words("english")) :
            tokens.remove(word)
    return tokens

def todataframe(rawdf):
    df = pd.DataFrame(rawdf, columns =["Text","Emotion"])
    return df
if __name__ == "__main__":
    main()
