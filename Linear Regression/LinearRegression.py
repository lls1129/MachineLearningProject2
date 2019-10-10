import csv
import math
import os
import time

# load data

def load(filename): 
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        return data[1:len(data)]
    print 'Data loading finished! '


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
def Cost(A, x, W, b): 
    n = len(A)
    sum = 0
    for i in range(n):
        sum += pow((A[i] - Dot(W, x) - b), 2)
        # sum += pow((A[i] - Predict(x, W, b)), 2)
    return sum / n
'''

def Cost_SingleFeature(A, x, w, b):
    n = len(A)
    sum = 0
    for i in range(n):
        sum += pow((A[i] - w * x[i] -b), 2)
    return sum / n

def Predict(x, W, b):
    Pred = [0] * len(x)
    for i in range(len(x)):
        Pred[i] = Dot(W, x[i]) + b
    return Pred

def UpdateWeights(w, b):
    partial = [2 * x2_mean * w + 2 * b * x_mean - 2 * xy_mean, 2 * x_mean * w + 2 * b - 2 * y_mean]
    w -= Dot(partial, H_Inverse[0])
    b -= Dot(partial, H_Inverse[1])
    return [w, b]

def GradientDescent(w, b, pre_cost, delta): 
    if delta < 0.0001: 
        return [w, b, pre_cost, delta]
    else: 
        [w, b] = UpdateWeights(w, b)
        cost = Cost_SingleFeature(y, x, w, b)
        delta = abs(cost - pre_cost)
        return GradientDescent(w, b, cost, delta)
        




# loading the csv file
path = os.getcwd() + '/kc_house_data.csv'
# print repr(path)
data = load(path)

# y is the price of house
y = [i[2] for i in data]
# x is the sqft_living
x = [i[5] for i in data]
l = len(x)
# The table loaded is stored as str as default, so change to float. 
for i in range(l):
    x[i] = float(x[i])
    y[i] = float(y[i])

x_mean = sum(x)/l
x2_mean = Dot(x,x)/l
y_mean = sum(y)/l
xy_mean = Dot(x,y)/l
H_Deter = 4 * x2_mean - 4 * x_mean * x_mean
# Hessian Matrix
H_Inverse = [[2 / H_Deter, -2 * x_mean / H_Deter], [-2 * x_mean / H_Deter, 2 * x2_mean / H_Deter]]

# Generate initial weights randomly
w0 = y_mean / x_mean
b0 = 0
delta0 = 100000
cost0 = 100000
# run time record
start = time.clock()
# Regression starts: 
[w, b, cost, delta] = GradientDescent(w0, b0, cost0, delta0)
# Ended
end = time.clock()
# print repr([w, b, cost, delta])
print "The result of linear regression: "
print "w = " + repr(w) + ", and b = " + repr(b)
print "The minimized cost converges to " + repr(cost) + ", and the delta is " + repr(delta) + ". "
print "The running time is " + repr(end - start) + " seconds. "
print "Walltime is: " + repr(time.asctime(time.localtime(time.time())))