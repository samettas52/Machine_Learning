import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X = iris.data
Y = iris.target


sinif_0 = X[0:50]
sinif_1 = X[50:100]
sinif_2 = X[100:150]


plt.scatter(sinif_0[:, 0], sinif_0[:, 1], color='red', label='Grup 1')
plt.scatter(sinif_1[:, 0], sinif_1[:, 1], color='green', label='Grup 2')
plt.scatter(sinif_2[:, 0], sinif_2[:, 1], color='blue', label='Grup 3')

plt.legend()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

y_test_list = list(y_test)
y_pred_list = list(y_pred)

cm = [[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]

for t, p in zip(y_test_list, y_pred_list):
    cm[t][p] += 1

A = cm[0][0]
B = cm[0][1]
C = cm[0][2]
D = cm[1][0]
E = cm[1][1]
F = cm[1][2]
G = cm[2][0]
H = cm[2][1]
I = cm[2][2]

TP_1 = A
FP_1 = D + G
FN_1 = B + C
TN_1 = E + F + H + I

prec_1 = TP_1 / (TP_1 + FP_1)
rec_1 = TP_1 / (TP_1 + FN_1)
f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1)

TP_2 = E
FP_2 = B + H
FN_2 = D + F
TN_2 = A + C + G + I

prec_2 = TP_2 / (TP_2 + FP_2)
rec_2 = TP_2 / (TP_2 + FN_2)
f1_2 = 2 * (prec_2 * rec_2) / (prec_2 + rec_2)

TP_3 = I
FP_3 = C + F
FN_3 = G + H
TN_3 = A + B + D + E

prec_3 = TP_3 / (TP_3 + FP_3)
rec_3 = TP_3 / (TP_3 + FN_3)
f1_3 = 2 * (prec_3 * rec_3) / (prec_3 + rec_3)

acc = (A + E + I) / (A + B + C + D + E + F + G + H + I)
macro_prec = (prec_1 + prec_2 + prec_3) / 3
macro_rec = (rec_1 + rec_2 + rec_3) / 3
macro_f1 = (f1_1 + f1_2 + f1_3) / 3

print(f"[{A}, {B}, {C}]")
print(f"[{D}, {E}, {F}]")
print(f"[{G}, {H}, {I}]\n")

print(f"Sinif 1 - TP:{TP_1} FP:{FP_1} FN:{FN_1} TN:{TN_1} | Prec:{prec_1:.2f} Rec:{rec_1:.2f} F1:{f1_1:.2f}")
print(f"Sinif 2 - TP:{TP_2} FP:{FP_2} FN:{FN_2} TN:{TN_2} | Prec:{prec_2:.2f} Rec:{rec_2:.2f} F1:{f1_2:.2f}")
print(f"Sinif 3 - TP:{TP_3} FP:{FP_3} FN:{FN_3} TN:{TN_3} | Prec:{prec_3:.2f} Rec:{rec_3:.2f} F1:{f1_3:.2f}\n")

print(f"Accuracy: {acc:.2f}")
print(f"Macro Precision: {macro_prec:.2f}")
print(f"Macro Recall: {macro_rec:.2f}")
print(f"Macro F1: {macro_f1:.2f}")