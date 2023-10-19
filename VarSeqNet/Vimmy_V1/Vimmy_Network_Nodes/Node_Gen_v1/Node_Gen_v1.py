import numpy as np
import torch.nn as nn
from PT_TRAINER_V1 import Node_Trainer_Setup, train_neural_network

input_sizes = 1
hidden_sizes = 1
output_sizes = 1
data_type = ""
description = ""

class NeuralNetwork_node(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, output_sizes, data_type, description):
        super(NeuralNetwork_node, self).__init__()
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.data_type = data_type
        self.description = description
        self.model = nn.Sequential(
            nn.Linear(input_sizes, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, output_sizes)
        )

    def forward(self, x):
        return self.model(x)

input_size = 1
hidden_sizes = 1
output_size = 1
data_type = "Logarithmic"
description = "First of many nodes in the Vimmy Network. This node is trained on randomly generated sets of Logarithmic pattern with variable noise."

Log_Node_Model_v1 = NeuralNetwork_node(input_size, hidden_sizes, output_size, data_type, description)

Node_Trainer_Setup()  # You can call this function if needed

# Training your model
# train_neural_network(Log_Node_Model_v1, input_data, target_data, learning_rate, epochs)
