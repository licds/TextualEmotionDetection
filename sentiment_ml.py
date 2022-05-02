from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def main():
    df= pd.read_pickle("cleaned_train.pkl")
    df_test = pd.read_pkl("cleaned_test.pkl")
main()

    
    

