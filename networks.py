### Network architectures
import torch
import torch.nn as nn
cuda = torch.device('cuda')

from modules import *

    

class AudioNet(nn.Module):

    def __init__(self,**prms):
        super(AudioNet, self).__init__()
        
        ### LAYER PARAMS
        ###############
        activations = {
            'elu': nn.ELU(alpha = prms['act_p1']),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'identity': nn.Identity(),
        }
        act = activations[prms['activation']]
        act_out = activations[prms['activation_output']]
        n_in = prms['n_input']
        n_lay = prms['n_layers']
        n_ker = prms['n_kernel']
        n_st = prms['n_stacks']
        self.skip_type = prms['skip_type']
        
        ### HIDDEN LAYERS
        ###############
         # npl : num. per. layer.
        if prms['module_type'] == 'dense':
            npl = [n_in] * n_lay  
            self.layers = nn.ModuleList([DenseLayer(n,act) for n in range(npl)])
        elif prms['module_type'] == 'causal':
            npl = [n_in - i * (n_ker - 1)  for i in range(1,n_lay+1)]   
            self.layers = nn.ModuleList([CausalLayer(n_ker,n,act) for n in npl])
        elif prms['module_type'] == 'dilated':
            dil = [2**k for k in range(n_lay)] * n_st
            npl = []
            tmp = n_in
            for d in dil:
                tmp -=  d * (n_ker - 1)
                npl.append(tmp) 
            self.layers = nn.ModuleList(DilationLayer(n_ker,n,d,act) for n,d in zip(npl,dil))   

        ### OUTPUT MODULE
        try:
            self.readout = prms['readout']
        except KeyError:
            self.readout = 'read-all'

            
        if self.readout == 'read-last':
            self.output = nn.Identity()
        elif self.readout == 'read-1st_last':
            self.output = OutputModule(2,act_out)
        elif self.readout == 'read-all':
            self.output = OutputModule(n_lay*n_st+1,act_out)
    
    
    def forward(self,x):
        to_output = x[:,:,-1]
        x0 = x
        for layer in self.layers:
            if self.skip_type == 'skip-none':
                skip = torch.zeros(x.shape).to(device=cuda)
            elif self.skip_type == 'skip-input':
                skip = x0 
            elif self.skip_type == 'skip-layer':
                skip = x   
            x = layer(x)
            x = x + skip[:,:,-x.shape[2]:]                
            to_output = torch.cat((to_output,x[:,:,-1]),1) 
            
        if self.readout == 'read-last':
            to_output = to_output[:,-1]
        elif self.readout == 'read-1st_last':
            to_output = to_output[:,[0,-1]]
        elif self.readout == 'read-all':
            pass
        x = self.output(to_output)
        return x 