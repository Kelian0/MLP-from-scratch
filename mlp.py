import numpy as np
from Unit import InputUnit, NeuralUnit, Loss
import random

class MLP:
    #Constructor
    def __init__(self,X,y,archi,seed = 5):
        self.archi = archi
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.K = len(archi) #number of layers (including input layer but omitting loss layer)
        self.rng = np.random.default_rng(seed=seed)
        #creating network

        net = []
        for layer_index in range(self.K):
            layer = []            
            for i in range(archi[layer_index]):
                if layer_index == 0:
                    input_unit = InputUnit(data=X[:,i])
                    layer.append(input_unit) #add InputUnit
                else:
                    neural_unit = NeuralUnit(layer_index,u=i,rng=self.rng)
                    for unit in net[layer_index-1]:
                        unit.plug(neural_unit,where='preceding')
                    neural_unit.reset_params()
                    layer.append(neural_unit)

            net.append(layer)

        layer = [] 
        loss_unit = Loss(y,self.K +1 )
        layer.append(loss_unit)
        for unit in net[-1]:
            loss_unit.plug(unit=unit)
            
        net.append(layer)

        self.net = net

    def forward(self,i):
        z = self.net[-1][0].forward(i)
        return z

    def backprop(self,i):
        self.net[self.K][0].backprop(i)
        deltas = self.net[self.K][0].delta
        for k in range(self.K-1,0,-1):
            deltas_new = np.zeros((self.net[k][0].npr,))
            for unit in self.net[k]:
                unit.backprop(i,deltas)
                deltas_new += unit.delta
            deltas = deltas_new
    
    def update(self,eta):
        for layer in self.net:
            for unit in layer:
                if isinstance(unit,NeuralUnit):
                    unit.w = unit.w - eta * unit.w_grad
                    unit.b = unit.b - eta * unit.b_grad
    
    def train(self, epochs, eta):
        for epoch in range(epochs):
            loss_epoch = 0

            for i in range(self.n):
                loss_epoch += self.forward(i)
                self.backprop(i)
                self.update(eta)

            print("epoch", epoch, "loss", loss_epoch / self.n)
    
    def predict(self,i):
        z = self.net[-2][0].forward(i)
        return z

    def visualize(self):
        nb_params = 0
        for i, layer in enumerate(self.net):
            print(f"Layer{i}")
            s=''
            for j,unit in enumerate(layer):
                if i == 0:
                    s += f'|I| '
                elif i == len(self.net) -1:
                    s += '|O| '
                else :
                    s += f'|N{unit.u}| '
                    nb_params += len(unit.w) + 1
            print(s)
        print(f'Nomber of parameters : {nb_params}')
        

