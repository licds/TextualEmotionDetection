import pandas as pd
from collections import defaultdict
import pickle
from collections import Counter

NEUTRAL = 0.4

def main():
    df_train = pd.read_pickle('cleaned_train.pkl')
    df_test = pd.read_pickle('cleaned_test.pkl')
    df_valid = pd.read_pickle("cleaned_valid.pkl")
    dict_train = load_dict("dict_train")
    dict_test = load_dict("dict_test")
    dict_valid = load_dict("dict_valid")
    pdict_train = bayes(df_train, dict_train)
    pdict_test = bayes(df_test, dict_test)
    pdict_valid = bayes(df_valid, dict_valid)
    tf_test = pd.read_pickle('tf_test.pkl')
    tf_train = pd.read_pickle('tf_train.pkl')
    tf_valid = pd.read_pickle('tf_valid.pkl')
    print("Accuracy with train - test:", test(df_test, pdict_train, tf_train))
    print("Accuracy with test - train:", test(df_train, pdict_test, tf_test))
    print("Accuracy with train - validation:", test(df_valid, pdict_train, tf_train))
    print("Accuracy with validation - train:", test(df_train, pdict_valid, tf_valid))
    print("Accuracy with test - validation:", test(df_valid, pdict_test, tf_test))
    print("Accuracy with validation - test:", test(df_test, pdict_valid, tf_valid))
    
    #average frequency of a unique word
    #print(get_unique_word(df3))
    #print(get_words(df3))

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
    for emo in emotion:
        emo = emo/sum(emotion)
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

# Print sentence sentiment
def print_emotion(emo_index, emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
    print(emo_l[emo_index])
    print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])

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
