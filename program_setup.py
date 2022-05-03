import csv
import nltk
import pickle
import numpy as np
import pandas as pd
from tokenize import tokenize
from collections import Counter
from nltk.stem import PorterStemmer
from collections import defaultdict

NEUTRAL = 0.3

def main():
    #========================== ONlY UNCOMMENT IF YOU WANT TO MODIFY DATASET ==========================
    #df = initdf("Data/program.csv")
    #df.to_pickle("cleaned_program.pkl") 

    #========================== UNCOMMENT THESE IF YOU WANT TO MODIFY DICT ==========================
    #df_prog = pd.read_pickle('cleaned_program.pkl')
    #dict_prog = dict_init(df_prog)
    #save_dict(dict_prog, "dict_prog")

    #============ TF-IDF ============
    #df_prog = pd.read_pickle('cleaned_program.pkl')
    #dict_prog = load_dict("dict_prog")
    #tf_prog = tf_idf(df_prog, dict_prog)
    #save_dict(tf_prog, "tf_prog")

    #============ TEST ============
    #df_train = pd.read_pickle('cleaned_train.pkl')
    #df_test = pd.read_pickle('cleaned_test.pkl')
    #df_valid = pd.read_pickle("cleaned_valid.pkl")
    #df_prog = pd.read_pickle("cleaned_program.pkl")
    #dict_prog = load_dict("dict_prog")
    #pdict_prog = bayes(df_prog, dict_prog)
    #save_dict(pdict_prog, "pdict_prog")
    #tf_prog = pd.read_pickle('tf_prog.pkl')
    #print("Accuracy with program - test:", test(df_test, pdict_prog, tf_prog))
    #print("Accuracy with program - train:", test(df_train, pdict_prog, tf_prog))
    #print("Accuracy with program - validation:", test(df_valid, pdict_prog, tf_prog))

    #============ ACTUAL PROGRAM ============
    df_prog = pd.read_pickle("cleaned_program.pkl")
    tf_prog = pd.read_pickle('tf_prog.pkl')
    pdict_prog = load_dict("pdict_prog")
    
    return 0

#============================== WORDS STATISTICS ==============================
def get_unique_word(df):
    unique_set = set()
    for sent in df['Text']:
        for word in sent:
            unique_set.add(word)
    return len(unique_set)
        
def get_words(df):
    emo_words = []
    for i in range(6):
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
    return sum(emo_words)

#============================== TEST MODULE ==============================
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
def test(df, dict, weight):
    accuracy = 0
    index = 0
    for text in df['Text']:
        emotion = classify(text, dict, weight)
        emo_index = emotion.index(max(emotion)) 
        # Take most probable emotion
        if emo_index == df.iloc[index, 1]:
            accuracy += 1
        index += 1
    accuracy = accuracy/len(df)
    return accuracy

def bayes(df,dict):
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
            pdict[word].append((emo_word/emo_words[i])*emo_p[i]/word_p)
    return pdict

#============================== TF-IDF WEIGHT CALCULATION ==============================
def tf_idf(df,dict):
    emo_words = []
    for i in range(6):
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
    tf_dict= defaultdict(list)
    avg_occurence_per_word = df['Text'].str.len().sum()/len(dict)
    for word in dict:
        f_x = len(dict[word])
        for emo in range(6):
            tf = dict[word].count(emo)/emo_words[emo]
            # 30 is optimal
            tf_dict[word].append(tf*np.log(1+(2*avg_occurence_per_word/f_x)))
            # how many times the word has appear in f_x
            # the denominator turns out to be a constant so the weight has an inverse relationship of its freq
            # the more a word appear, the less import it is.(eg.a noun water(that has no emotion value in a sentence would have less weight
            # evaluate it ))
    return tf_dict

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

def predict(sentence, dict, weight):
    emotion = classify(sentence, dict, weight)
    if sentence == []:
        return 6
    if max(emotion) > NEUTRAL/10000+1:
        emo_index = emotion.index(max(emotion)) 
    else:
        emo_index = 6
    return emo_index

def print_emotion(emo_index, emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise", "Neutral"]
    print("The emotion for this sentence is:", emo_l[emo_index])
    print("Sad: ", round((emotion[0]-1)*10000, 4), "Joy: ", round((emotion[1]-1)*10000, 4), "Love: ", round((emotion[2]-1)*10000, 4), "Anger: ", round((emotion[3]-1)*10000, 4), "Fear: ", round((emotion[4]-1)*10000, 4), "Surprise: ", round((emotion[5]-1)*10000, 4))


#============================== OTHERS ==============================
def interactive_test():
    sentence = input("Enter the sentence you want to classify!!!:")

if __name__ == "__main__":
    main()
