import csv
import math
import os
import random
import numpy as np
from random import shuffle

# load data

def Loading(filename): 
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
    for i in range(len(data)):
        if data[i][1] == 'M': 
            data[i][1] = -1.
        else :
            data[i][1] = 1.
        for j in range(len(data[1])):
            data[i][j] = float(data[i][j])
    for i in range(len(data[0])):
        if i < 2: 
            i = 2
        feature = Normalize(Rescalling([x[i] for x in data]))
        for j in range(len(feature)):
            data[j][i] = feature[j]
    for i in range(len(data)):
        vector = ScaleToUnit(data[i][2:len(data[i])])
        for j in range(len(data[i])-2):
            data[i][j+2] = vector[j]
    return data
    print ('Data loading and generalizing finished! ')

def split(data, ratio):
    trainingSet = []
    testSet = []
    for x in range(len(data)):
        if random.random() < ratio: 
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])
    return trainingSet, testSet

# feature scaling, done by each feature
def Normalize(array):
    if len(array) == 0:
        return 0
    mean = sum(array) / len(array) * 1.0
    var = Dot(Minus(array, mean), Minus(array, mean))
    sd = math.sqrt(var / len(array))
    return Divide(Minus(array, mean), sd)   

def Rescalling(array):
    Min=min(array)*1.0
    Max=max(array)*1.0
    return Divide(Minus(array, Min), (Max-Min))

# Done by each row
def ScaleToUnit(array):
    norm = math.sqrt(Dot(array, array))
    return Divide(array, norm)


# vector, scalar computation
def Minus(a, B):
    A = a[:]
    if(type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] - B
        return A
    else: 
        for i in range(len(A)):
            A[i] = A[i] - B[i]
        return A

def ABSMinus(a, B):
    A = a[:]
    if(type(B) != list):
        for i in range(len(A)):
            A[i] = abs(a[i] - B)
        return A
    else:
        for i in range(len(A)):
            A[i] = abs(a[i] - B[i])
        return A

def Multiply(a, B):
    A = a[:]
    if(type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] * B
        return A
    else: 
        for i in range(len(A)):
            A[i] = A[i] * B[i]
        return A   
def Adding(a, B):
    A = a[:]
    if(type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] + B
        return A
    else: 
        for i in range(len(A)):
            A[i] = A[i] + B[i]
        return A    
def Divide(a, B):
    A = a[:]
    if(type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] / B
        return A
    else: 
        for i in range(len(A)):
            A[i] = A[i] / B[i]
        return A   
def Dot(A, B):
    if(type(A) != list):
        if(type(B) != list):
            return A * B
        else: 
            for i in range(len(B)):
                B[i] = B[i] * A
            return B
    if((type(B) != list) and (type(A) == list)):
        for i in range(len(A)):
            A[i] = A[i] * B
        return A    
    if(len(A) != len(B)): 
        return 0
    else:
        sum = 0
        for i in range(len(A)):
            sum += A[i] * B[i]
        return sum

'''    
def HingeLoss(W, b, C): 
    loss = 0; 
    for i in range(l): 
        loss += max(0, 1 - y[i] * (Dot(W, x[i]) + b)) / l
    return loss * C + Dot(W, W)
'''
# Gram matrix
def Gram(x):
    g = [[0.] * len(x)] * len(x)
    for i in range(len(x)):
        for j in range(len(x)):
            g[i][j] = Dot(x[i],x[j])
    return g
            
def Pred(a, b, x, xxx, index):
    p = 0.
    for i in range(len(x)):
        p += a[i] * y[i] * Dot(x[i], xxx[index])
    return p - b
    
            
def Update(x, y, a, b, i, j):
    pred_i = Pred(a, b, x, x, i)
    pred_j = Pred(a, b, x, x, j)

    e_i = pred_i - y[i]
    # print repr(e_i)
    e_j = pred_j - y[j]
    # print repr(e_j)
    lower = 1000.
    upper = -1000.
    if y[i] * y[j] == 1:
        lower = max(0, a[i] + a[j] - C)
        upper = min(C, a[i] + a[j])
    if y[i] * y[j] == -1:
        lower = max(0, a[i] - a[j])
        upper = min(C, a[i] - a[j] + C)
    a_i_old = a[i]
    if (g[i][i] - 2 * g[i][j] + g[j][j]) != 0:
        a_i_new = a_i_old - y[i] * (e_i - e_j) / (g[i][i] - 2. * g[i][j] + g[j][j])
    else:
        a_i_new = a_i_old - y[i] * (e_i - e_j) / (g[i][i] - 2 * g[i][j] + g[j][j] + 0.01)

    # print repr(a_i_new)
    if a_i_new < lower:
        a_i_new = lower
    if a_i_new > upper:
        a_i_new = upper
    # print repr(a_i_new)
    delta_a_i = a_i_new - a_i_old
    delta_a_j = - y[j] * y[i] * delta_a_i
    a[j] += delta_a_j
    a[i] = a_i_new
    b_i = b + e_i + y[j] * delta_a_j * g[j][i] + y[i] * delta_a_i * g[i][i]
    b_j = b + e_j + y[j] * delta_a_j * g[j][j] + y[i] * delta_a_i * g[j][i]
    bb = 0.
    if 0 < a[i] < C: 
        bb = b_i
    if 0 < a[j] < C: 
        bb = b_j
    if (a[i] == 0) or (a[i] == C) or (a[j] == 0) or (a[j] == C):
        bb = (b_i + b_j) / 2.
    return a, bb
'''
def KKTVerify(x, y, a, b, i): 
    pred_i = Pred(a, b, x, x, i)
    if ((y[i] * pred_i <= 1) and (a[i] < C)) or ((y[i] * pred_i >= 1) and (a[i] > 0)) or ((y[i] * pred_i == 1) and ((a[i] == 0) or (a[i] == C))):
        return 0
    else: 
        return 1
'''
def KKTVerify(x, y, a, b, i):
    pred_i = Pred(a, b, x, x, i)
    if ((y[i] * pred_i <= 1) and (a[i] == C)) or ((y[i] * pred_i >= 1) and (a[i] == 0)) or ((y[i] * pred_i == 1) and (0 < a[i] < C)):
        return 1
    else:
        return 0

def MarginKKTVerify(x, y, a, b, i):
    pred_i = Pred(a, b, x, x, i)
    if (y[i] * pred_i != 1) and (0 < a[i] <C):
        return 0
    else:
        return 1

def SelectI(x, y, a, b, pred):
    index = 0
    gap = 0
    for i in range(len(y)):
        if KKTVerify(x, y, a, b, i) == 0:
            gap_i = Gap(x, y, a, b, pred, i)
            if gap_i > gap:
                gap = gap_i
                index = i
    return index

def SelectJ(x, y, a, b, pred):
    e = Minus(pred, y)
    e_diff = ABSMinus(e, e[i])
    J = LargestAbsErrorIndex(e_diff, x, y, a, b, i)
    if J != 0:
        return
    else:
        return 1


def Gap_no_pred(x, y, a, b, i):
    pred_i = Pred(a, b, x, y, i)
    gap_i = a[i] * (y[i] * pred_i - 1 - y[i] * b) + C * max(0, 1 - y[i] * pred_i)
    return gap_i

def Gap(x, y, a, b, pred, i):
    gap_i = a[i] * (y[i] * pred[i] - 1 - y[i] * b) + C * max(0, 1 - y[i] * pred[i])
    return gap_i

def Firstloop(x, y, a, b):
    counter = 0
    for i in range(len(y)):
        if KKTVerify(x, y, a, b, i) == 0:
            pred = [0.] * len(y)
            for k in range(len(x)):
                pred[k] = Pred(a, b, x, x, k)
            j = FindJ(x, y, a, b, pred, i)
            # print "Before Updates: (" + repr(a[j]) + ", " + repr(j) + "). "
            a, b = Update(x, y, a, b, i, j)
            print "Updates: (" + repr(i) + ", " + repr(j) + "). "
            counter += 1
    return a, b, counter



def FindIJ(x, y, a, b): 
    pred = [1.0] * len(y)
    e = [1.0] * len(y)
    marker = [0.] * len(y)
    maxerror = 0.
    ISet = []
    for i in range(len(x)):
        if KKTVerify(x, y, a, b, i) == 0:
            pred[i] = Pred(a, b, x, x, i)
            e[i] = pred[i] - y[i]
            if 0 <= a[i] <= C:
                marker[i] = 1
                if e[i] > maxerror:
                    maxerror = e[i]
                    ISet.append(i)
    if (sum(marker) == 0) or (maxerror < 0.001):
        print repr(maxerror)
        return 0, 0
    else:
        # I = np.random.choice(ISet)
        I = ISet[0]
        J = FindJ(x, y, a, b, pred, I)
        return I, J

def LargestAbsErrorIndex(e_diff, x, y, a, b, i):
    abserrordiff = 0
    index = 0
    for k in range(len(e_diff)):
        if ((k != i) and (KKTVerify(x, y, a, b, k) == 1)):
            # print "k = " + repr(k)
            if abserrordiff <= e_diff[k]:
                abserrordiff = e_diff[k]
                index = k
    return index

def FindJ(x, y, a, b, pred, i):
    e = Minus(pred, y)
    e_diff = ABSMinus(e, e[i])
    J = LargestAbsErrorIndex(e_diff, x, y, a, b, i)
    if J != 0:
        return J
    else:
        return 1
	
def CalcWB(x, y, a): 
    w = [0.] * len(x[0])
    for i in range(len(y)):
        w = Adding(w, Multiply(x[i], y[i] * a[i]))
    b = Dot(w, x[1]) - y[1]
    return w, b
        

'''
def training(x, y, a, b, iterations):
    a, b, count = Firstloop(x, y, a, b)
    while (count - len(y)) < iterations:
        I, J = FindIJ(x, y, a, b)
        if I + J != 0:
            print "Before Update: a[i] a[j] are: " + repr(a[I]) + ", " + repr(a[J]) + ". "
            a, b = Update(x, y, a, b, I, J)
            print "After Update: a[i] a[j] are: " + repr(a[I]) + ", " + repr(a[J]) + ". "
            count += 1
            if count % 10 == 0:
                print "Updates: (" + repr(I) + ", " + repr(J) + "). "
        else:
            break
    w, B = CalcWB(x, y, a)
    return a, b, w, B, count
'''

def training(x, y, a, b, iterations):
    run = 0
    count = 0
    while run < iterations:
        a, b, tempcount = Firstloop(x, y, a, b)
        count += tempcount
        run += 1

    return a, b, count



    
# loading the csv file
path = os.getcwd() + '/wdbc.data'
# print repr(path)
data = Loading(path)
train, test = split(data, 0.8)

train = train[0:100]

l = len(train)
d = len(train[1])

y = [i[1] for i in train]
x = [i[2:d] for i in train]
g = Gram(x)

# print repr(l * 1. / len(data))
# print repr(l)
C = 300
b0 = 0
W0 = [0.] * d
a0 = [600.01] * len(y)
# for i in range(len(a)):
#     a[i] = random.random()
b0 = 0.1

ppp = [0.] * l
qqq = ppp[:]





for i in range(l):
    ppp[i] = Pred(a0, b0, x, x, i)
    if np.sign(ppp[i]) != np.sign(y[i]): 
        qqq[i] = "False"
    else:
        qqq[i] = "True!"
print repr(qqq)
z = Adding(y, 1)
print repr(sum(z) / 2 / l)
a_10, b_10, count = training(x, y, a0, b0, 200)
for i in range(len(y)):
    if KKTVerify(x, y, a_10, b_10, i) == 1:
        print "I am OK with KKT, I am: " + repr(i)
# a_10, b_10 = a, b
number = 0.
number2 = 0.
for i in range(l): 
    if np.sign(Pred(a_10, b_10, x, x, i)) == np.sign(y[i]): 
        # print repr(Pred(a_50, b_50, i))
        number += 1.
    if np.sign(Pred(a_10, b_10, x, x, i)) != np.sign(y[i]):
        number2 += 1.
print repr(number / l)
print repr(number2 / l)
print repr(count)
print repr(a_10)
# print repr(b_10)




yy = [i[1] for i in test]
# for i in range(len(yy)):
#     yy[i] = random.random() - 1
xx = [i[2:d] for i in test]

ll = len(yy)
dd = len(test[1])
number = 0.
number2 = 0.
for i in range(ll): 
    if np.sign(Pred(a_10, b_10, x, xx, i)) == np.sign(yy[i]): 
        # print repr(Pred(a_50, b_50, i))
        number += 1.
    if np.sign(Pred(a_10, b_10, x, xx, i)) != np.sign(yy[i]):
        number2 += 1.
print "Testing Accuracy is: " + repr(number / ll)
print repr(number2 / ll)


