import pandas as pd
from collections import defaultdict
import pickle
import math

def main():
    df = pd.read_pickle('cleaned_valid.pkl')
    #dict = load_dict("dict")
    dict = load_dict("dict_valid")
    #word_IDF = wordIDF(df, dict)
    #save_dict(word_IDF, "word_IDF")
    
    #valid dataset
    tf = tf_idf(df, dict)
    #print(tf["lie"])
    save_dict(tf, "tf3")

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
            word_IDF[word].append(math.log(emotion_count[i]/((word_freq_in_emo_dict[word])[i])+1))
    return word_IDF

#def tf_idf(df, dict):
   # emo_words = []
    #for i in range(6):
        #emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
    #tf_dict= defaultdict(list)
    
    #for x frequency in each emotion class
    #for word in dict:
       #f_x = len(dict[word])
        #unique_set = set(dict[word])
        #for emo in unique_set:
               #tf = /len(df.loc[df['Text'].str.contains(word), emo])
            # 10 is optimal
            #print(sum(emo_words))
            #tf_dict[word].append(tf*math.log(1+((df.shape[0])/f_x)))
    #return tf_dict

def tf_idf2(df,dict):
    emo_sentence =[]
    for i in range(6): 
        emo_sentence.append(len(df[df['Emotion'] == i]))
    print(emo_sentence)
    tf_idf_dict = defaultdict(list)
    for word in dict:
        for i in range(6):
            #tf = dict[word].count(i)/len(df['Text'].str.contains(word))   
            #print(math.log(1+(30/len(dict[word]))))
           #1) only taking the whole document as ts sum(emo_Sentence) 
           tf_idf_dict[word].append(math.log(1+(sum(emo_sentence)/len(dict[word]))))
            #tf_idf_dict[word].append(math.log(1+(emo_sentence[i]/len(dict[word]))))
    return tf_idf_dict

def tf_idf (df,dict):
    emo_words = []
    for i in range(6):
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
    tf_dict= defaultdict(list)
    for word in dict:
        f_x = len(dict[word])
        for emo in range(6):
            tf = dict[word].count(emo)/emo_words[emo]
            # 10 is optimal
            tf_dict[word].append(tf*math.log(1+(10.78/f_x)))
            #how many times the word has appear in f_x
            # the denominator turns out to be a constant so the weight has an inverse relationship of its freq
            # the more a word appear, the less import it is.(eg.a noun water(that has no emotion value in a sentence would have less weight
            # evaluate it ))
    return tf_dict
if __name__ == "__main__":
    main()