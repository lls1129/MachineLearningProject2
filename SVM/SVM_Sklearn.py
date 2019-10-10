# python 3.7


import csv
import math
import os
import random
import numpy as np
from random import shuffle
import time


# load data


def Loading(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
    for i in range(len(data)):
        if data[i][1] == 'M':
            data[i][1] = -1.
        else:
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
        for j in range(len(data[i]) - 2):
            data[i][j + 2] = vector[j]
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
    Min = min(array) * 1.0
    Max = max(array) * 1.0
    return Divide(Minus(array, Min), (Max - Min))


# Done by each row
def ScaleToUnit(array):
    norm = math.sqrt(Dot(array, array))
    return Divide(array, norm)


# vector, scalar computation
def Minus(a, B):
    A = a[:]
    if (type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] - B
        return A
    else:
        for i in range(len(A)):
            A[i] = A[i] - B[i]
        return A


def ABSMinus(a, B):
    A = a[:]
    if (type(B) != list):
        for i in range(len(A)):
            A[i] = abs(a[i] - B)
        return A
    else:
        for i in range(len(A)):
            A[i] = abs(a[i] - B[i])
        return A


def Multiply(a, B):
    A = a[:]
    if (type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] * B
        return A
    else:
        for i in range(len(A)):
            A[i] = A[i] * B[i]
        return A


def Adding(a, B):
    A = a[:]
    if (type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] + B
        return A
    else:
        for i in range(len(A)):
            A[i] = A[i] + B[i]
        return A


def Divide(a, B):
    A = a[:]
    if (type(B) != list):
        for i in range(len(A)):
            A[i] = A[i] / B
        return A
    else:
        for i in range(len(A)):
            A[i] = A[i] / B[i]
        return A


def Dot(A, B):
    if (type(A) != list):
        if (type(B) != list):
            return A * B
        else:
            for i in range(len(B)):
                B[i] = B[i] * A
            return B
    if ((type(B) != list) and (type(A) == list)):
        for i in range(len(A)):
            A[i] = A[i] * B
        return A
    if (len(A) != len(B)):
        return 0
    else:
        sum = 0
        for i in range(len(A)):
            sum += A[i] * B[i]
        return sum




# loading the csv file
path = os.getcwd() + '/wdbc.data'
# print repr(path)
data = Loading(path)
train, test = split(data, 0.8)

# train = train[0:100]

l = len(train)
d = len(train[1])

y = [i[1] for i in train]
x = [i[2:d] for i in train]
# print (y)

# print repr(l * 1. / len(data))
# print repr(l)
C = 2
threshold = 0.001
b0 = 0
W0 = [0.] * d
a0 = [00.01] * len(y)
# for i in range(len(a)):
#     a[i] = random.random()
b0 = 0.1

z = Adding(y, 1)
print (sum(z) / 2 / l)

yy = [i[1] for i in test]
# for i in range(len(yy)):
#     yy[i] = random.random() - 1
xx = [i[2:d] for i in test]

ll = len(yy)
dd = len(test[1])
from sklearn import svm
c = 1
cc = []
trainingscore = []
testingscore = []
runningtime = []
while c < 10:
    Model = svm.SVC(C=c, gamma='auto')
    start = time.clock()
    Model.fit(x, y)
    end = time.clock()
    wall = time.time()
    V = Model.support_vectors_
    # print (V.shape)
    # print (V)
    print ("Training accuracy is: ")
    print (Model.score(x, y))
    trainingscore.append(Model.score(x, y))
    print ("Testing accuracy is: ")
    testingscore.append(Model.score(xx, yy))
    print (Model.score(xx, yy))
    print ("Running time: ")
    runningtime.append(end - start)
    print (end - start)
    print ("Wall clock: ")
    print (time.asctime(time.localtime(wall)))
    cc.append(c)
    c += 1

print (cc)
print (trainingscore)
print (testingscore)
print (runningtime)