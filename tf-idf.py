import pandas as pd
from collections import defaultdict
import pickle
import math

def main():
    df = pd.read_pickle('cleaned_train.pkl')
    dict = load_dict("dict")
    word_IDF = wordIDF(df, dict)
    save_dict(word_IDF, "word_IDF")
    

    tf = tf_idf(df, dict)
    save_dict(tf, "tf")

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
            word_IDF[word].append(math.log(emotion_count[i]/((word_freq_in_emo_dict[word])[i]+1)))
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
            # 10 is optimal
            tf_dict[word].append(tf*math.log(1+(10/f_x)))
    return tf_dict

if __name__ == "__main__":
    main()