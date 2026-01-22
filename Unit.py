import numpy as np
import copy
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

class InputUnit:
    def __init__(self,data):
        self.data = data        #one column of matrix X
        self.n = data.shape[0]  #dataset size
        self.k = 0              #layer number
        self.z = 0              #unit output

    def plug(self, unit, where = 'following'):
        unit.plug(self, where = where)

    def forward(self,i):
        self.z = self.data[i]
        return self.z

class Loss:
    #Constructor
    def __init__(self,y,k):
        self.preceding = [] #list of preceding neurons
        self.npr = 0        #length of list preceding
        self.y = y          #array of class labels of the training data
        self.k = k          #layer index
        self.l = 0
        self.delta = np.zeros((1,))

    def plug(self, unit, where = 'preceding'):
        unit.plug(self,where)

    def forward(self,i):
        if self.npr != 1:
            raise Exception("The Loss layer has more then one preceding")
        input = self.preceding[0].forward(i)
        if self.y[i] == 0:
            self.l = -np.log(1-input)
        else:
            self.l = -np.log(input)


        return self.l
    
    def backprop(self,i):
        if self.y[i] == 0:
            self.delta[0] = 1 / (1-self.preceding[0].z)
        else:
            self.delta[0] = - 1 / (self.preceding[0].z)

        
        return self.delta
        



class NeuralUnit:
    #Constructor
    def __init__(self,k,u,rng = np.random.default_rng()):
        self.u = u          #unit number
        self.preceding = [] #list of preceding neurons
        self.npr = 0        #length of list preceding
        self.following = [] #list of following neurons
        self.nfo = 0        #length of list following
        self.k = k          #layer number
        self.w = 0          #unit weights
        self.b = 0          #unit intercept
        self.z = 0          #unit output
        self.rng = rng

    def __str__(self):
        return f"Unit {self.u} of layer {self.k}"

    def reset_params(self):
        self.w =  self.rng.standard_normal(self.npr)
        self.b = self.rng.standard_normal()
        self.delta = np.zeros(self.w.shape)
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = 0

    def plug(self,unit, where = 'preceding'):
        if isinstance(unit,NeuralUnit):
            if where == 'following':
                self.preceding.append(unit)
                unit.following.append(self)
                self.npr += 1
                unit.nfo += 1
                # print(f"{self} is after {unit}")
            elif where == 'preceding':
                self.following.append(unit)
                unit.preceding.append(self)
                self.nfo += 1
                unit.npr += 1
                # print(f"{self} is before {unit}")
        elif isinstance(unit,InputUnit):
            if where == 'following':
                raise Exception("Not possible to add neurone befor the input layer")
            elif where == 'preceding':
                self.preceding.append(unit)
                self.npr += 1
        elif isinstance(unit,Loss):
            if where == 'following':
                raise Exception("Not possible to add neurone after the Loss layer")
            elif where == 'preceding':
                self.following.append(unit)
                self.nfo += 1
                unit.preceding.append(self)
                unit.npr += 1

    def forward(self,i):
        input = np.zeros(self.npr)
        for j,unit in enumerate(self.preceding):
            input[j] = unit.forward(i)
        
        self.z = sigmoid(self.w.T@input + self.b)
        return self.z
    
    def backprop(self,i,deltas):
        
        for v,unit in enumerate(self.preceding):
            self.delta[v] = self.z * (1-self.z) * self.w[v] * deltas[self.u]


        for v,unit in enumerate(self.preceding):
            self.w_grad[v] = self.z * (1-self.z) * unit.z * deltas[self.u]

        self.b_grad = self.z * (1-self.z) * deltas[self.u]

        