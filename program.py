import nltk
import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pickle
import numpy as np
import pandas as pd
from tokenize import tokenize
from collections import Counter
from nltk.stem import PorterStemmer

NEUTRAL = 0.8

#============================== DICTIONARY MANIPULATION ==============================
# Load a dictionary from a pickle file
def load_dict(filename):
    with open(filename+".pkl", "rb") as tf:
        new_dict = pickle.load(tf)
    return new_dict

tf_prog = pd.read_pickle('tf_prog.pkl')
pdict_prog = load_dict("pdict_prog")

app = dash.Dash()

app.layout = html.Div(children=[
                    html.H1(children='Emotion Detection Simple App'), 
                    html.Div(children='''Enter the sentence you want to classify: '''), 
                    dcc.Input(id='input', value='', type='text'), 
                    html.Div(id='output')
                    #html.Div(id='output-graph')
                    ])
@app.callback(Output(
                    component_id='output', 
                    component_property='children'),
                    #component_id='output-graph', 
                    #component_property='children'), 
                    [Input(component_id='input', component_property='value')]
                    )

#============================== PREDICTION MODULE ==============================
# Take emotion sentiment from each word of a sentence and average 
# it to represent the sentiment for the sentence
def classify(sentence):
    dict = pdict_prog
    weight = tf_prog
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
def predict(sentence, dict, weight):
    emotion = classify(sentence, dict, weight)
    if sentence == []:
        return 6
    if max(emotion) > NEUTRAL/10000+1:
        emo_index = emotion.index(max(emotion)) 
    else:
        emo_index = 6
    return emo_index

# Print sentence sentiment
def print_emotion(emo_index, emotion):
    emo_l = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise", "Neutral"]
    print("The emotion for this sentence is:", emo_l[emo_index])
    print("Sad: ", round((emotion[0]-1)*10000, 4), "Joy: ", round((emotion[1]-1)*10000, 4), "Love: ", round((emotion[2]-1)*10000, 4), "Anger: ", round((emotion[3]-1)*10000, 4), "Fear: ", round((emotion[4]-1)*10000, 4), "Surprise: ", round((emotion[5]-1)*10000, 4))

#============================== USER INPUT MANIPULATION ==============================
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

def update_value(input_data):
    try:
        sentence = list(tokenize(input_data))
        emotion = classify(sentence, pdict_prog, tf_prog)
        for i in emotion:
            i = round((i-1)*10000, 4)
        emo_index = predict(sentence, pdict_prog, tf_prog)
        emotion_types = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise", "Neutral"]
        return emotion_types[emo_index]
    except:
        return "Oops, errors!"
    '''
    d = {'Emotion': ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"], "Emotion_Value": emotion}
    df = pd.Dataframe(d)
    df.reset_index(inplace=True)
    df.set_index("Emotion", inplace=True)
    
    return dcc.Graph(
        id='output-graph',
        figure={
            'data': [
                {'x': df.index, 'y': df.Emotion_Value, 'type': 'bar', 'name': input_data},
            ],
            'layout': {
                'title': input_data
            }
        }
    )
    '''

if __name__ == "__main__":
    app.run_server(debug=True, port=3008)
