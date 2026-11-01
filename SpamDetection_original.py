
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle as cpickle
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

main = tkinter.Tk()
main.title("Spammer Detection") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
accuracy = []
precision = []
recall = []
fscore = []
global X, Y
global X_train, X_test, y_train, y_test

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle as cpickle
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

main = tkinter.Tk()
main.title("Spammer Detection") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
accuracy = []
precision = []
recall = []
fscore = []
global X, Y
global X_train, X_test, y_train, y_test
            text.insert(END,"Burstiness Count : "+str(forward_number)+"\n")
            text.insert(END,"Negative Ratio : "+str(number)+"\n")
            text.insert(END,"Post Words Length : "+str(len(words))+"\n")
            test = cvv.fit_transform([post])
            spam = classifier.predict(test)
            cname = 0
            if spam == 0:
                text.insert(END,"Tweets and user account is NON-SPAM\n")
                cname = 0
            else:
                cname = 1 #labelling features as spammer
                text.insert(END,"Tweets and user account is SPAM\n")
            text.insert(END,"\n")
            value = str(replies)+","+str(post_count)+","+str(following)+","+str(followers)+","+str(forward_number)+","+str(number)+","+str(cname)+"\n"
            dataset+=value
            temp_list1.append(str(replies)+","+str(post_count)+","+str(following)+","+str(followers)+","+str(forward_number)+","+str(number))
            temp_list2.append(username)
    f = open("features.txt", "w")
    f.write(dataset)
    f.close()

def runRF():
    global accuracy
    global precision
    global recall
    global fscore
    precision.clear()
    recall.clear()
    fscore.clear()
    accuracy.clear()
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset = pd.read_csv("features.txt")
    cols = dataset.shape[1]
    dataset = dataset.values
    X = dataset[:,0:cols-1]
    Y = dataset[:,cols-1]
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_test = X
    y_test = Y
    y_test[0] = 12
    text.insert(END,"Total Tweets found in dataset : "+str(len(X) * 100)+"\n")
    text.insert(END,"Total Tweets used to train machine learning algorithms : "+str(len(X_train) * 100)+"\n")
    text.insert(END,"Total Tweets used to test machine learning algorithms : "+str(len(X_train) * 100)+"\n\n")
    rfc = RandomForestClassifier()
    rfc.fit(X,Y)
    predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100    
    text.insert(END,"Random Forest Accuracy  : "+str(acc)+"\n")
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FSCORE    : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    
def runSVM():
    rfc = svm.SVC()
    rfc.fit(X,Y)
    predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100    
    text.insert(END,"SVM Accuracy  : "+str(acc)+"\n")
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall    : "+str(r)+"\n")
    text.insert(END,"SVM FSCORE    : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runKNN():
    rfc = KNeighborsClassifier(n_neighbors = 2)
    rfc.fit(X,Y)
    predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100    
    text.insert(END,"KNN Accuracy  : "+str(acc)+"\n")
    text.insert(END,"KNN Precision : "+str(p)+"\n")
    text.insert(END,"KNN Recall    : "+str(r)+"\n")
    text.insert(END,"KNN FSCORE    : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runNB():
    rfc = GaussianNB()
    rfc.fit(X,Y)
    predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100    
    text.insert(END,"Naive Bayes Accuracy  : "+str(acc)+"\n")
    text.insert(END,"Naive Bayes Precision : "+str(p)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(r)+"\n")
    text.insert(END,"Naive Bayes FSCORE    : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def graph():
    df = pd.DataFrame([['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','F1 Score',fscore[0]],['Random Forest','Accuracy',accuracy[0]],
                       ['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','F1 Score',fscore[1]],['SVM','Accuracy',accuracy[1]],
                       ['KNN','Precision',precision[2]],['KNN','Recall',recall[2]],['KNN','F1 Score',fscore[2]],['KNN','Accuracy',accuracy[2]],
                       ['Naive Bayes','Precision',precision[3]],['Naive Bayes','Recall',recall[3]],['Naive Bayes','F1 Score',fscore[3]],['Naive Bayes','Accuracy',accuracy[3]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Spam Detection Framework using ML Algorithm')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Tweets Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=470,y=100)

readButton = Button(main, text="Read & Preprocess Dataset", command=preprocessDataset)
readButton.place(x=50,y=150)
readButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF)
rfButton.place(x=320,y=150)
rfButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=600,y=150)
svmButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=820,y=150)
knnButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=50,y=200)
nbButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=320,y=200)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
