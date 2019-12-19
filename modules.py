### Network modules

import torch.nn as nn

class DenseLayer(nn.Module):   
    def __init__(self,n_in,act):
        super(DenseLayer, self).__init__()
        
        self.weights = nn.Linear(n_in,n_in) 
        self.bn = nn.BatchNorm1d(n_in)
        self.activation = act
        
    def forward(self,x):
        x = self.weights(x.squeeze())
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class CausalLayer(nn.Module): 
    def __init__(self,kernel_size,n_in,act):
        super(CausalLayer, self).__init__()
        
        self.n_in = n_in
        self.weights = nn.Conv1d(in_channels=1,out_channels=1, kernel_size=kernel_size,dilation=1)
        self.bn = nn.BatchNorm1d(n_in)
        self.activation = act
        
    def forward(self,x):
        x = self.weights(x).view(-1,self.n_in)
        x = self.bn(x)
        x = self.activation(x).view(-1,1,self.n_in)
        return x

    
class DilationLayer(nn.Module):
    def __init__(self,kernel_size,n_in,dil,act):
        super(DilationLayer,self).__init__()
        
        self.n_in = n_in
        self.weights = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,dilation=dil)
        self.bn = nn.BatchNorm1d(n_in)
        self.activation = act
        
    def forward(self,x):
        x = self.weights(x).view(-1,self.n_in)
        x = self.bn(x)
        x = self.activation(x).view(-1,1,self.n_in)
        return x
    


class OutputModule(nn.Module):
    def __init__(self,n_in,act):
        super(OutputModule, self).__init__()
        
        self.weights = nn.Linear(n_in,1) 
        self.activation = act
        
    def forward(self,x):
        x = self.weights(x.squeeze())
        x = self.activation(x)
        return x    
    
