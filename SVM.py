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
            
def Pred(a, b,  xxx, index):
    p = 0.
    for i in range(len(x)):
        p += a[i] * y[i] * Dot(x[i], xxx[index])
    return p + b
    
            
def Update(a, b, i, j):
    pred_i = Pred(a, b, x, i)
    pred_j = Pred(a, b, x, j)

    e_i = pred_i - y[i]
    # print repr(e_i)
    e_j = pred_j - y[j]
    # print repr(e_j)
    lower = 1000
    upper = -1000
    if y[i] * y[j] == 1:
        lower = max(0, a[i] + a[j] - C)
        upper = min(C, a[i] + a[j])
    if y[i] * y[j] == -1:
        lower = max(C, a[i] - a[j])
        upper = min(C, a[i] - a[j] + C)
    a_i_new = a[i] - y[i] * (e_i - e_j) / (g[i][i] - 2 * g[i][j] + g[j][j])
    # print repr(a_i_new)
    if a_i_new <= lower:
        a_i_new = lower
    if a_i_new > upper:
        a_i_new = upper
    # print repr(a_i_new)
    delta_a_i = a_i_new - a[i]
    delta_a_j = y[j] * y[i] * (a[i] - a_i_new)
    a[j] += y[i] / y[j] * (a[i] - a_i_new)
    a[i] = a_i_new
    b1 = b - e_i - y[i] * delta_a_i * g[i][i] - y[j] * delta_a_j * g[i][j]
    b2 = b - e_j - y[i] * delta_a_i * g[i][j] - y[j] * delta_a_j * g[j][j]
    if 0 < a[i] < C: 
        b = b1
    if 0 < a[j] < C: 
        b = b2
    else: 
        b = (b1 + b2) / 2.
    return a, b


def ChooseIandJ(a, b):
    pred = [0.] * len(y)
    e1 = [0.] * len(y)
    for i in range(len(y)):
        pred[i] = Pred(a, b, x, i)
        e1[i] = pred[i] - y[i]
    e2 = e1[:]
    e3 = e1[:]
    for i in range(len(e1)):
        if (((a[i] > 0) and (e1[i] <= 0)) or ((a[i] <= 0) and (e1[i] > 0))):
            e1[i] = 0
        if (((a[i] < C) and (e3[i] >= 0)) or ((a[i] >= C) and (e3[i] < 0))):
            e3[i] = 0
        if ((((a[i] < 0) or (a[i] > C)) and (e2[i] != 0)) or ((0 < a[i] < C) and (e2[i] == 0))):
            e2[i] = 0
    e = Adding(Multiply(e1, e1), Adding(Multiply(e2, e2), Multiply(e3, e3)))
    # print repr(len(e))
    I = np.argmax(e)
    if I == len(y):
        I = I - 1
    J = np.random.randint(len(y)-1)
    while (J == I):
        J = np.random.randint(len(y)-1)
    # print "(" + repr(I) + ", " + repr(J) + ")"
    return I, J


def training(a, b, C, iterations): 
    alpha = [0.] * len(y)
    beta = 0.
    count = 0
    while (count < iterations): 
        I, J = ChooseIandJ(a, b)
        Update(a, b, I, J)
        '''
        for i in range(l): 
            for j in range(l):
                if ppp[j] != ppp[i]:
                    Update(a, b, i, j)
            # print "i = " + repr(i)
        '''
        count += 1
    return a, b


    
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
C = 2
b = 0
W = [1] * d
a = [10.1] * len(y)
for i in range(len(a)):
    a[i] = random.random()
b = 0.1


# print repr(HingeLoss(W, b, C))
# print repr(ChooseIandJ(a, b))

z = Adding(y, 1)
# print repr(sum(z) / 2 / l)
a_10, b_10 = training(a, b, C, 500)
# a_10, b_10 = a, b
number = 0.
number2 = 0.
for i in range(l): 
    if np.sign(Pred(a_10, b_10, x, i)) == np.sign(y[i]): 
        # print repr(Pred(a_50, b_50, i))
        number += 1.
    if np.sign(Pred(a_10, b_10, x, i)) != np.sign(y[i]):
        number2 += 1.
print "Training Accuracy is: " + repr(number / l)
# print repr(number2 / l)
# print repr(a_10)
# print repr(b_10)
'''
Update(a, b, 1, 16)
print repr(a)
'''


yy = [i[1] for i in test]
# for i in range(len(yy)):
#     yy[i] = random.random() - 1
xx = [i[2:d] for i in test]

ll = len(yy)
dd = len(test[1])
number = 0.
number2 = 0.
for i in range(ll): 
    if np.sign(Pred(a_10, b_10, xx, i)) == np.sign(yy[i]): 
        # print repr(Pred(a_50, b_50, i))
        number += 1.
    if np.sign(Pred(a_10, b_10, xx, i)) != np.sign(yy[i]):
        number2 += 1.
print "Testing Accuracy is: " + repr(number / ll)
# print repr(number2 / ll)