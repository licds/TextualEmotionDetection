import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

def main():
    df_train = pd.read_pickle('cleaned_train.pkl')
    df_test = pd.read_pickle('cleaned_test.pkl')
    df_valid = pd.read_pickle('cleaned_valid.pkl')
    dict_train = load_dict("dict_train")
    dict_test = load_dict("dict_test")
    dict_valid = load_dict("dict_valid")
    tf_train = tf_idf(df_train, dict_train)
    tf_test = tf_idf(df_test, dict_test)
    tf_valid = tf_idf(df_valid, dict_valid)
    save_dict(tf_train, "tf_train")
    save_dict(tf_test, "tf_test")
    save_dict(tf_valid, "tf_valid")

#============================== TF-IDF WEIGHT CALCULATION ==============================
def tf_idf (df,dict):
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

#============================== DICTIONARY MANIPULATION ==============================
# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

if __name__ == "__main__":
    main()