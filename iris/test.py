import numpy as np
x1=np.array([2,3,3,4,5,6,7,9,11,10])
x2=np.array([4,6,4,10,2,3,9,7,7,2])
y=[0,1,1,0,0,1,1,0,0,0]
y1=["kötü","iyi","iyi","kötü","kötü","iyi","iyi","kötü","kötü","kötü"]
test=[8,4]
mesafe = np.zeros(10)
for i in range(10):
    mesafe[i]=np.sqrt((test[0] - x1[i]) ** 2 + (test[1] - x2[i]) ** 2)
a = np.sort(mesafe)
b = np.argsort(mesafe)
print("k=1 için " ,y1[b[0]])
#k=3 için
k=3
iyi = 0
kotu = 0
for i in range (k):
    if y1[b[i]]=="iyi":
        iyi +=1
    else :
        kotu +=1
if iyi >> kotu :
  print("k=3 için " , "iyi")
else:
  print("k=3 için " , "kotu")