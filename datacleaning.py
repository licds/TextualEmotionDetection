import csv
import nltk
import string

def main():
    f = open("test.csv")

    csvreader=csv.reader(f)
    rows=[]
    n=0
    for row in csvreader:
        n+=1
        if n==1:
            continue
        rows.append(row)
    for row in rows:
        l=tokenize(row[0])
        row[0]=l
    print(rows)
    

def tokenize(document):
    l=[]
    tokens = nltk.word_tokenize(document)
    for word in tokens:
        word=word.lower()
        if (word not in string.punctuation) and (word not in nltk.corpus.stopwords.words("english")) :
            l.append(word)
    return l

if __name__ == "__main__":
    main()
