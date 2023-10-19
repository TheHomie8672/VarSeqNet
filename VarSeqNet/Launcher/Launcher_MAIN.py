import torch
import numpy as np
import os
import sys
import time
import torch.nn as nn
from PT_MODEL_GEN import model, optim, LEARNING_RATE
from PT_TRAINER_V1 import PT_NN_Trainer

# So look, I am a backend kinda guy, This UI shit is totally out if ny skillset. Here is a rough overview of what I an looking for though!

# user enters launcher  ->  create <-> train <-> utilize <-> return or view results 

# Supported Model types: PyTorch, TensorFlow, Vimmy (in house model)

# UI Type: Minimalist, Sleek, Intuititive

# UI Colour Scheme: Matte Greyscale Blue, Deep Grey, White (Various Shades)

# Must Have Features: Data Visualization Suite, Integrated Command Terminal, Webcam display for Visual Network Development 




class ML_Model_Launcher:
    def __init__(self):
        # Initialize UI with the specified color scheme and features
        self.initialize_ui()
        # Load supported models (PyTorch, TensorFlow, Vimmy)
        self.load_models()

    def initialize_ui(self):
        # Initialize UI here with the specified color scheme and features
        # Not my area of expertise
        pass

    def load_models(self):
        # Load the supported models (PyTorch, TensorFlow, Vimmy)
        # define loading procedures for each model here
        pass

    def create_model(self, model_type):
        # Allow the user to create a new instance of a specific model type
        pass

    def train_model(self, model_instance, training_data, target_labels, epochs, learning_rate):
        # Train the selected model with the provided data
        pass

    def utilize_model(self, model_instance, input_data):
        # Utilize the selected model with input data and return results
        pass

    def view_results(self, results):
        # Display the results using data visualization features
        pass

    def run(self):
        while True:
            # Main program loop, allow the user to create, train, utilize, or view results
            # Based on user input, call the corresponding methods
            pass

if __name__ == "__main__":
    # Create an instance of the MLModelLauncher
    launcher = ML_Model_Launcher()

    # Start the main program loop
    launcher.run()
