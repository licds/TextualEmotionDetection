import csv
import nltk
import string
import pandas as pd

""" Use weighted and frequency to label each word, and count the freq and weight in a sentence.
weighted = weight(dict)
freq = frequency(dict)
test = tokenize(sentence)
        emotion = 0
        weighted_emotion = 0
        count = 0
        for word in test:
            if word in freq.keys():
                if freq[word] != 0:
                    emotion += freq[word]
                    weighted_emotion += weighted[word]
                    count += 1
        print(emotion_sentiment(emotion/count))
        print(emotion_sentiment(weighted_emotion/count))

# use them as percentage of categories, then compute at the last stage
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