import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.data", header=None)

X= df.iloc[:, 0:4] #(: 0 dan 3 e kadar (tum sutunları alır)
Y= df.iloc[:, 4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y) #105x4 , 45x4, 105x1, 45x1, stratify y sorabilir

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test) #45x1

C = list(y_pred)
D = list(Y_test)

F=[]

for i in range(len(C)):
    if C[i] == D[i]:
        F.append(1)
    else:
        F.append(0)

Acc = np.sum(F)/len(C)
print(Acc * 100)

cm = confusion_matrix(Y_test,y_pred)
print(cm) #yandan bakınca toplam 15 olacak

cm1 = np.array([             #sınavda hoca soracak bu matris ne anlatıyor diye sözel olarak orada anlatacağız yani buradaki değerler ne ifade ediyor,
    [cm[0,0], cm[0,1] + cm[0,2]],
    [cm[1,0] + cm[2,0], cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]] #1.sınıfın diğerlerine göre durumu nedir, her matriste ilk sınnıf için hesaplıyoruz a e ve ı için
])                                                              # specifity hesaplamayacağız ama diğerlerini hesaplayacağız
print(cm1)                                                      #specifity neden hesaplanmaz niye? sınav sorusu

cm2 = np.array([
    [cm[1,1], cm[1,0] + cm[1,2]],
    [cm[0,1] + cm[2,1], cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]]
])
print(cm2)

cm3 = np.array([
    [cm[2,2], cm[2,1] + cm[2,0]],
    [cm[0,2]+ cm[1,2], cm[0,0] + cm[1,1] + cm[0,2] + cm[0,1]]
])
print(cm3)

sensitivity1 = cm1[0,0] / (cm1[0,0] + cm1[1,0])
print(sensitivity1)
sensitivity2 = cm2[0,0] / (cm2[0,0] + cm2[1,0])
print(sensitivity2)
sensitivity3 = cm3[0,0] / (cm3[0,0] + cm3[1,0])
print(sensitivity3)

print("-----------")

precision1= cm1[0,0] / (cm1[0,0] + cm1[0,1])
print(precision1)
precision2= cm2[0,0] / (cm2[0,0] + cm2[0,1])
print(precision2)
precision3= cm3[0,0] / (cm3[0,0] + cm3[0,1])
print(precision3)

print("-----------")
F1_Score1 = 2 * precision1 * sensitivity1 / (precision1 + sensitivity1)
print(F1_Score1)
F1_Score2 = 2 * precision2 * sensitivity2 / (precision2 + sensitivity2)
print(F1_Score2)
F1_Score3 = 2 * precision3 * sensitivity3 / (precision3 + sensitivity3)
print(F1_Score3)