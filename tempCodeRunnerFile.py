# -*- coding: utf-8 -*-


import nltk
nltk.download('stopwords')

import pandas as pd
import dill
import re

classifier = dill.load(open('model_data.sav', 'rb'))
tfvectorizer = dill.load(open('tfidf_data.sav', 'rb'))
vectorizer = dill.load(open('bow_data.sav', 'rb'))

import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
def clean(s):
    s=str(s)
    s=s.lower()
    html=re.compile('<.*?>')
    cleaned = re.sub(html,' ',s)
    fil=[]
    for i in cleaned.split():
        if i!='c++':
            cleaned=re.sub('[^A-Za-z]', '', i)
            fil.append(cleaned)
        else:
            fil.append(i)
    return fil


stop=set(stopwords.words('english'))
sno=SnowballStemmer('english')


def stem(s):
    fil=[]
    for _ in s:
        if _ not in stop:
            s=(sno.stem(_).encode('utf8'))
            fil.append(s)
    s=b' '.join(fil)
    return s

from tkinter import *
from PIL import ImageTk, Image
import os
root = Tk()


root.title("Autonomous Tagging Of Stackoverflow Questions")

img = ImageTk.PhotoImage(Image.open("download (1).png"))
panel = Label(root, image = img)
panel.image = img
panel.grid(row = 0, column = 0)


img1 = ImageTk.PhotoImage(Image.open("download.png"))
label1 = Label(root,image = img1)
label1.image = img1
label1.grid(row = 1, column = 0)


label2 = Label(root, text = "Enter Your Question",font = "Arial 20", fg = 'black')
label2.grid(row = 2, column = 0)


ques = StringVar()
quesEntered = Entry(root, width = 40, textvariable = ques,font="Arial 18")
quesEntered.grid(column = 0, row = 3, padx = 3, pady = 3)


ans = StringVar()
def pred():
    t = ques.get()
    l=[]
    l.append(stem(clean(t)))
    x=tfvectorizer.transform(l)
    t=classifier.predict(x)
    k=vectorizer.inverse_transform(t)
    res = re.sub('[^A-Za-z#+-]+', ' ', str(k[0]))
    ans.set(res)
    label3 = Entry(root,textvariable = ans, font = "Arial 20 bold")
    label3.grid(column =0, row = 5, padx = 3, pady = 3)


button = Button(root,text = "Predict Now", font = "Arial 18", command = pred)
button.grid(column= 0, row = 4, padx = 3, pady = 3)



root.mainloop()



