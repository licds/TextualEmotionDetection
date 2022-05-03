import pandas as pd
from datacleaning import dict_init, save_dict, load_dict, tokenize
from model import test, bayes, classify
from weight import tf_idf

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
    #df_prog = pd.read_pickle("cleaned_program.pkl")
    #tf_prog = pd.read_pickle('tf_prog.pkl')
    #pdict_prog = load_dict("pdict_prog")
    return 0

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

if __name__ == "__main__":
    main()
