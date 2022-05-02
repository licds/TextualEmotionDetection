import pandas as pd
from collections import defaultdict
import pickle
from collections import Counter

NEUTRAL = 0.4

def main():
    #df = pd.read_pickle('cleaned_train.pkl')
    df2 = pd.read_pickle('cleaned_test.pkl')
    df3 = pd.read_pickle("cleaned_valid.pkl")
    dict = load_dict("dict_valid")
    pdict = bayes(df3, dict)
    #word_IDF = pd.read_pickle('word_IDF.pkl')
    tf = pd.read_pickle('tf3.pkl')
    df3 = pd.read_pickle("cleaned_valid.pkl")

    #print("Yi's df-idf:",test(df2, pdict, word_IDF))
    #print("df-idf:",test(df2, pdict, tf))
    print("df-idf:",test(df2,pdict,tf))
    
    #average frequency of a unique word


    #print("Original:", test(df2, pdict, 0))
    print(get_unique_word(df3))
    print(get_words(df3))  
    
    #avg = 0
    #for i in range(16000):
        #avg += len(df['Text'][i])
    #print(avg/16000)


# Save a dictionary into pickle file
def save_dict(dict, filename):
    with open(filename+".pkl", "wb") as tf:
        pickle.dump(dict,tf)

# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict


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

# Take emotion sentiment from each word of a sentence and average 
# it to represent the sentiment for the sentence
def classify(sentence, dict, weight):
    sentence_len = len(sentence)
    #print(sentence)
    count = Counter(sentence)
    #print(count)
    #emotion = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    emotion = [1,1,1,1,1,1]
    counter = 0
    #emotion =[]
    if sentence_len == 0:
        return
    for word in sentence:
        if word in dict.keys():
            counter+=1
            for i in range(6):
                if weight == 0:
                    emotion[i] += dict[word][i]
                else:
                    emotion[i] += dict[word][i]*(count[word]/sentence_len * weight[word][i])
                    #print(emotion)
    #print(emotion)
    #for i in range(len(emotion)):
     #   emotion[i] = emotion[i]/size
    return emotion

# Print sentence sentiment
def print_emotion(emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
    for i in range(len(emotion)):
        if emotion[i] >= NEUTRAL:
            print(emo_l[i])
            print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
            return
    #print("Neutral")
    print("Sad: ", emotion[0], "Joy: ", emotion[1], "Love: ", emotion[2], "Anger: ", emotion[3], "Fear: ", emotion[4], "Surprise: ", emotion[5])
    
# Build a test trial with accuracy
def test(df, dict, weight):
    accuracy = 0
    index = 0
    for text in df['Text']:
        #print(text)
        emotion = classify(text, dict, weight)
        emo_index = emotion.index(max(emotion)) 
        # Take most probable emotion
        if emo_index == df.iloc[index, 1]:
            accuracy += 1
        index += 1
    accuracy = accuracy/len(df)
    return accuracy

'''def bayes(df, dict):
    #size = len(df)
    words = df.Text.str.len().sum()
    emo_words = []
    emo_p = []
    emo_sen =[]
    for i in range(6):
        emo_sen.append(len(df[df['Emotion']==i]))
        emo_words.append((df[df['Emotion']==i]).Text.str.len().sum())
        #print(emo_words)
        #print((df[df['Emotion']==i]).Text.str.len().sum())
        emo_p.append(emo_words[i]/words)
    #print(emo_p)
    pdict = defaultdict(list)
    #print(dict)
    for word in dict.keys():
        n_word = len(dict[word])
        word_p = n_word/words
        for i in range(6):
            emo_word=dict[word].count(i)
            #print(emo_word)
            #print(emo_words[0])
            pdict[word].append((emo_word/emo_sen[i])*(emo_sen[i]/16000)/word_p)
            
            #pdict[word].append(emo_word*(emo_sen[i]/16000)/word_p)
        #print(emo_p)
    #print(emo_words)
    #print(emo_p)
    #print(dict["smirk"])
    #print(emo_words[0])
    
    #print(emo_p[0])
    return pdict
'''
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
if __name__ == "__main__":
    main()
