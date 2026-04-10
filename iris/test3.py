import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df =pd.read_csv('iris.data',header=None)

print(df.head())

X=df.iloc[:,0:4].values
Y=df.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,stratify=Y) #Hold -out cross validtaion yöntemi ile veriseti üzerini ikiye bölmeye yarar bu fonksiyon

print("train matrisi")
print(X_train.shape)
print("test matrisi")

print(X_test.shape)


knn = KNeighborsClassifier(n_neighbors=1) # burada k değerini veriyor ve hangi mesafe metriği üzerinden olacağını  belirler.
knn.fit(X_train,y_train) # verilen değerlerden öğrenme sağlar fit

y_pred = knn.predict(X_test) #

print(y_pred)
print("********")
print(y_test)

C=list(y_pred)
D=list(y_test)

print(C)
print(D)


'''
F = []

for x, y in zip(C, D): #kütüphane olarak kullandım verileri böyle yaptım {1:0} yanii dedim ki key valueye eşit ise yap bunu 
    if x == y:
        F.append(1)
    else:
        F.append(0)     
'''

F=[]

for i in range(len(C)):
    if C[i] ==D[i]:
        F.append(1)
    else:
        F.append(0)



print(F)
print(len(F))

AccuraryRate=(F.count(1)/len(F))*100

print(f'Doğruluk oranı (accurary rate): %',AccuraryRate)

cm = confusion_matrix(y_test,y_pred)


print(cm)







c1 = np.array([
    [cm[0][0], cm[0][1] + cm[0][2]],
    [cm[1][0] + cm[2][0], cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]]
])

c2 = np.array([
    [cm[1][1], cm[1][0] + cm[1][2]],
    [cm[0][1] + cm[2][1], cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]]
])

c3 = np.array([
    [cm[2][2], cm[2][0] + cm[2][1]],
    [cm[0][2] + cm[1][2], cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]]
])

print("c1:\n", c1)
print("c2:\n", c2)
print("c3:\n", c3)

 #tp /tp+fn
print("c1 matirisinin sensivity değeri :")
sens1=(c1[0,0]) / (c1[1,0]+c1[0,0])
print(sens1)

print("c2 matrisinin sensivity değeri :")
sens2=(c2[0,0]) / (c2[1,0]+c2[0,0])
print(sens2)

print("c3 matrisininn sensivity değeri :")
sens3=(c3[0,0]) / (c3[1,0]+c3[0,0])
print(sens3)


print("********")
#tp /fp+tp
print("c1 matrisinin Prec değeri :")

prec1=(c1[0,0]) / (c1[0,0]+c1[0,1])
print(prec1)
print("c2 matrisinin Prec :")
prec2=(c2[0,0]) / (c2[0,0]+c2[0,1])
print(prec2)
print("c3 matrisinin Prec :")
prec3=(c3[0,0]) / (c3[0,0]+c3[0,0])
print(prec3)

print("********")

print("c1 matrisi için F1 değeri : ")
f1c1=(2*(prec1*sens1)) / (prec1+sens1)
print(f1c1)

print("c2 matrisi için f1 değeri :")
f1c2 =(2*(prec2*sens2)) / (prec2+sens2)
print(f1c2)

print("c3 matrisi için f1 değeri :")
f1c3=(2*(prec3*sens3)) / (prec3+sens3)
print(f1c3)



g1 = df.iloc[:, 0].values
g2 = df.iloc[:, 1].values

for i in range(len(g1)):
    if i <= 50:
        plt.plot(g1[i], g2[i], '*b')
    elif i <= 100:
        plt.plot(g1[i], g2[i], '>r')
    else:
        plt.plot(g1[i], g2[i], 'og')


plt.grid(True)
plt.show()