from log_set_gen_v1 import logarithmic_dataset_generator
from PT_TRAINER_V1 import Node_Trainer_Setup
from PT_MODEL_GEN import NeuralNetwork_node
import os




# Node Details:

input_size = 1  
hidden_sizes = 4  
output_size = 1  
learning_rate = 0.10
epochs = 100
data_type = "Logarithmic"  
description = "First of many nodes in the Vimmy Network. This node is trained on randomly generated sets of Logarithmic pattern with variable noise. "  

Log_Node_Model_v1 = NeuralNetwork_node(input_size, hidden_sizes, output_size, data_type, description)

# Training Data Details:

folder_path = 'D:\VarSeqNet\Data_Sets\Example_Data_Sets'
num_samples = 100
start = 1
stop = 5
base = 10
noise_stddev = 0.1
file_name = "logarithmic_dataset_T.csv"
file_path = os.path.join(folder_path, file_name)



# Save Data-Set 

# Self Explanitory
model = Log_Node_Model_v1
Node_Trainer_Setup()
