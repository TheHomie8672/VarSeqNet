import numpy as np
import torch.nn as nn

input_sizes =  [1]
hidden_sizes =  [1]
output_sizes =  [1]
data_type =   ""
description =  ""

size = (input_sizes,hidden_sizes,output_sizes)

class NeuralNetwork(nn.Module):
    
    
    
    def __init__(self, input_sizes, hidden_sizes, output_sizes, data_type, description,size):
        super(NeuralNetwork, self).__init__()
        
        self.size = size
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.data_type = data_type
        self.description = description
        self.model = nn.Sequential(
            nn.Linear(input_sizes, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_sizes)
           
        )

    def forward(self, x):
        return self.model(x)
    
model = NeuralNetwork(input_sizes, size, hidden_sizes, output_sizes, data_type, description)
    
    






     
    
    
   


    