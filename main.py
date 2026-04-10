from traceback import print_tb
import numpy as np

x1 = np.array([2,3,3,4,5,6,7,9,11,10])
x2 = np.array([4,6,4,10,8,3,9,7,7,2])
y  =np.array( [0,1,1,0,0,1,1,0,0,0])
Calculate = np.zeros(10)

for i in range(0,len(x1)):

    Calculate[i] = np.sqrt((x1[i]-8)**2+(x2[i]-4)**2)

Distance=np.argsort(Calculate)
# YIndex=Distance[0]
# if y[YIndex] == 0:
#     print("Kötü")
# else:
#     print("iyi")

#look for k=3 value
#I think we need to add  a Counter
#Our teacher only showed us the  for loop  and  if-else structure so i wrote the code using only both.
k = 2
GoodCounter = 0
BadCounter = 0
for i in range(0,k):
 YIndex = Distance[i]
 if y[YIndex] == 0:
  BadCounter += 1
 else:
  GoodCounter += 1
if GoodCounter > BadCounter:
    print("iyi")
else:
    print("kötü")

