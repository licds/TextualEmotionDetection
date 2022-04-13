import csv
from lib2to3.pgen2.pgen import DFAState
import nltk
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from textblob import TextBlob
#from textblob.np_extractors import ConllExtractor

def main():
    df = initdf("Data/test.csv")
    dict = first_convert(df)
    pdict = bayes(df, dict)
    print(pdict['cry'])
    #print(pdict['suspicious'])
    """
    state = True
    while state != False:
        sentence = input("Enter a sentence: ")
        if sentence == "f":
            state = False
            break
        testimonial = TextBlob(sentence)
        print(testimonial.sentiment)
    """

def bayes(df, dict):
    size = len(df)
    words = df.Text.str.len().sum()
    emo_words = [0,0,0,0,0,0]
    emo_words[0] = (df[df['Emotion']==0]).Text.str.len().sum()
    emo_words[1] = (df[df['Emotion']==1]).Text.str.len().sum()
    emo_words[2] = (df[df['Emotion']==2]).Text.str.len().sum()
    emo_words[3] = (df[df['Emotion']==3]).Text.str.len().sum()
    emo_words[4] = (df[df['Emotion']==4]).Text.str.len().sum()
    emo_words[5] = (df[df['Emotion']==5]).Text.str.len().sum()
    
    emo_p = [0,0,0,0,0,0]
    emo_p[0] = emo_words[0]/words
    emo_p[1] = emo_words[1]/words
    emo_p[2] = emo_words[2]/words
    emo_p[3] = emo_words[3]/words
    emo_p[4] = emo_words[4]/words
    emo_p[5] = emo_words[5]/words
    
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
    tokens=nltk.word_tokenize(text)
    new=[]
    for word in tokens:
        word=word.lower()
        word=WordNetLemmatizer().lemmatize(word)
        if word not in nltk.corpus.stopwords.words("english"):
            new.append(word)
    return new

def first_convert(df):
    dict1 = defaultdict(list)
    emotion = list(df.loc[:,"Emotion"])
    for row in range(1,len(df.index)):
        temp = df.iloc[row,0]
        for word in temp:
            dict1[word].append(emotion[row])
    return dict1

"""
def weight(dict):
    weighted = defaultdict(list)
    for key in dict.keys():
        weighted[key] = sum(dict[key])/len(dict[key])
    return weighted

def frequency(dict):
    freq = defaultdict(list)
    for key in dict.keys():
        freq[key] = nltk.FreqDist(dict[key]).max()
    return freq

def emotion_sentiment(emotion):
    emotioncp = round(emotion)
    if emotioncp == 0:
        emotion_type = 'Sadness'
    elif emotioncp == 1:
        emotion_type = 'Joy'
    elif emotioncp == 2:
        emotion_type = 'Love'
    elif emotioncp == 3:
        emotion_type = 'Anger'
    else:
        emotion_type = 'Fear'
    print("Emotion detected:", emotion_type, "     ", "The actual weight is: ", emotioncp)
    return
"""

if __name__ == "__main__":
    main()
