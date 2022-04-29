import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

def main():
    df = pd.read_pickle('cleaned_train1.pkl')
    dict = first_convert(df)
    save_dict(dict, "dict")
    word_IDF = wordIDF(df, dict)
    save_dict(word_IDF, "word_IDF")
    one_IDF = one_tf_idf(df, dict)
    save_dict(one_IDF, "one_IDF")
    tf = tf_idf(df, dict)
    print(tf['sad'])
    print(tf['im'])
    save_dict(tf, "tf")
    n = 0
    for i in df['Text']:
        n += len(i)
    words = len(dict)
    print(n, words, n/words)

# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

def wordIDF(df, dict):
    # number_sentence = len(df['Text'])
    emotion_count = []
    word_freq_in_emo_dict = defaultdict(list)
    word_IDF = defaultdict(list)
    for i in range(6):
        emotion_count.append((df["Emotion"]==i).sum())
    for word in dict.keys():
        for i in range(6):
            emo_freq=dict[word].count(i)
            word_freq_in_emo_dict[word].append(emo_freq)
            word_IDF[word].append(np.log(emotion_count[i]/((word_freq_in_emo_dict[word])[i]+1)))
    return word_IDF

def one_tf_idf(df, dict):
    sentences = len(df['Text'])
    word_freq = defaultdict(int)
    word_IDF = defaultdict(float)
    for word in dict:
        word_freq[word] = len(dict[word])
        word_IDF[word] = np.log(sentences/word_freq[word])
    return word_IDF

def tf_idf(df, dict):
    emo_words = []
    for i in range(6):
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
    tf_dict= defaultdict(list)
    #for x frequency in each emotion class
    for word in dict:
        f_x = len(dict[word])
        for emo in range(6):
            tf = dict[word].count(emo)/emo_words[emo]
            # 30 is optimal
            tf_dict[word].append(tf*np.log(1+(30/f_x)))
    return tf_dict

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