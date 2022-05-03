import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pickle
import pandas as pd
from tokenize import tokenize
from datacleaning import tokenize
from program_setup import predict
from model import classify


NEUTRAL = 0.3

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
                    html.H1('Emotion Detection Simple App'), 
                    html.H2(children='''Enter the sentence you want to classify: '''), 
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

def update_value(input_data):
    try:
        sentence = list(tokenize(input_data))
        emotion = classify(sentence, pdict_prog, tf_prog)
        if len(sentence) == 0:
            input_data = "Neutral"
            return input_data
        for i in emotion:
            i = round((i-1)*10000, 4)
        emo_index = predict(sentence, pdict_prog, tf_prog)
        emotion_types = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise", "Neutral"]
        input_data = emotion_types[emo_index]
        return input_data
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
