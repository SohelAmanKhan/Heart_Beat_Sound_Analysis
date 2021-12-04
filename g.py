import librosa
import matplotlib as mt
import soundfile
import os, glob
import pickle
import dtale
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
emo2=0
x=[]
y=[]
for file in glob.glob("C:\\Users\\LENOVO\\Desktop\\heart\\*.wav"):
    file_name=os.path.basename(file)
    emo2=str(file_name)
    emo2=emo2[:6]
    if emo2=="normal":
        pass
    else:
        emo2="abnormal"
    SO,s_rate =librosa.load(file)
    feature=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=SO, sr=s_rate, n_mfcc=40).T, axis=0)
    feature=np.hstack((feature, mfccs))
    x.append(feature)
    y.append(emo2)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2)
names = ["Linear_SVM:","Gradient_Boosting:", "Decision_Tree:","Random_Forest:", "Naive_Bayes:"]

classifiers = [
    SVC(kernel="linear"),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    GaussianNB()]
scores = []
score=[]
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    scores.append([name,score])
print(scores)
