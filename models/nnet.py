import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class Linear:
    
    linear_act = linear_act = lambda x : x

    def __init__(self,in_features,out_features,activation=linear_act):
        self.w = torch.rand((in_features,out_features),requires_grad=True).to(torch.float32)
        self.b = torch.rand((1,out_features),requires_grad=True).to(torch.float32)
        self.activation = activation

    def update(self,learning_rate=0.01):
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad
        self.w.grad.zero_()
        self.b.grad.zero_()
    
    def forward(self,x):
        self.g = x @ self.w + self.b
        self.z = self.activation(self.g)

class Model:
    def __init__(self,layers,n_iters,loss):
        
        self.layers = layers
        self.n_iters = n_iters
        self.loss = loss

    def train(self,X,y):

        for i in range(self.n_iters):

            x = X

            for layer in self.layers:
                layer.forward(x)
                x = layer.z

            j = self.loss(x,y)
            j.backward(retain_graph=True)

            if i%(self.n_iters//10)==0:
                print(j)

            for layer in self.layers[::-1]:
                layer.update()
            
            plt.plot(X,y)
            plt.plot(X,x)
            plt.show()
    
    def predict(self,X):
        if type(X) in [int,float]:
            X = torch.tensor([X]).to(torch.float32)
        if type(X)!= torch.Tensor:
            raise "Please pass Tensor or int or float as input."  
        x = X

        for layer in self.layers:
            layer.forward(x)
            x = layer.z

        return x
    
# model = Model([Linear(1,2,nn.Sigmoid()),Linear(2,2,nn.ReLU()),Linear(2,1)],10000,nn.MSELoss())

# x = np.linspace(-5,5,50000)
# noise = np.random.random(50000)
# y= x**2
# # plt.plot(x,y,'.')

# # x = np.linalg.norm(x)

# # x = (x-x.mean())/x.std()

# x = x.reshape(-1,1).astype(np.float32)
# y = y.reshape(-1,1).astype(np.float32)
# x,y = torch.tensor(x),torch.tensor(y)

# model.train(x,y)

# # print(np.sin(4)*10)
# print(model.predict(5))