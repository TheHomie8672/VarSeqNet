import torch
import numpy as np
import os
import sys
import time
import curses
import torch.nn as nn
from PT_MODEL_GEN import NeuralNetwork, model, optim, LEARNING_RATE,input_size,hidden_sizes,output_size

# This trainer is specifically meant for models generated using  'PT_MODEL_GEN' In other words, if you try to train a Vimmy or TS Model it WONT WORK!
# Also will probibly break something so maybe just don't do it. This isnt Nike after all (;
# Acually, You know what Its not like ive tried, I havnt even built them yet. Give it hell, who knows. Maybe something interesting will happen!


class PT_NN_Trainer:
    def __init__(self, neural_network, training_data, epochs, learning_rate,target_labels):
        self.neural_network = neural_network
        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target_labels = target_labels
        
        # planning on adding an auto-trainer that uses a NN trained on a small user provided dataset and produces a larger, labeled and divided set.
        # Just havnt gotten around to it yet!
        
    def PT_Trainer_Setup(epochs,learning_rate,training_data):
        print("Welcome to the Training Suite")
        time.sleep(0.5)
        print("Please select an Option:  Select Data File | Generate Data | Enter Dataset | BACK ")
        print()
        selection = input("YOU:  ")
        if "select data file" in selection:
            print() # placeholder, dont worry
            
        elif "generate data" in selection:
            print() # placeholder# placeholder
            
        elif "enter dataset" in selection:
            print("Please specify data type")
            print("Please select an Option:  numerical | nlp | Intelli-Sense | BACK ") # Intelli-Sense coming soon! hopefully... Also no NLP support yet
            
        elif "back" in selection:
            print() # placeholder
    def Node_Trainer_Setup(self,epochs,learning_rate):
     while True:
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "Node Trainer Setup:")
        self.stdscr.addstr(1, 0, "1. Select a model")
        self.stdscr.addstr(2, 0, "2. Select a dataset")
        self.stdscr.addstr(3, 0, "3. Back to the main menu")
        self.stdscr.refresh()
        key = self.stdscr.getch()

        if key == ord('1'):
            model_files = self.display_models(1)
            while True:
                key = self.stdscr.getch()
                if key == curses.KEY_UP and self.current_row > 1:
                    self.current_row -= 1
                elif key == curses.KEY_DOWN and self.current_row < len(model_files):
                    self.current_row += 1
                elif key == 10:  # Enter key
                    selected_model = model_files[self.current_row - 1]

                    # Set the model save path to the specified folder and file name
                    model_save_path = r"D:\VarSeqNet\MODELS\Vimmy_V1\Vimmy_Network_Nodes\trained_model.pth"

                    # Load your model and prepare input_data and target_labels
                    model = NeuralNetwork(input_size, hidden_sizes, output_size)  # Replace with your model
                    input_data = torch.tensor([input_data], dtype=torch.float32)  # Replace with your input data
                    target_labels = torch.tensor([target_labels], dtype=torch.float32)  # Replace with your target labels
                    epochs = 100  # Replace with the number of epochs
                    learning_rate = 0.001  # Replace with your learning rate

                    self.train_neural_network(model, input_data, target_labels, epochs, learning_rate, model_save_path)
                    break
                elif key == ord('M') or key == ord('m'):
                    break
                elif key == 27:  # Escape key to exit
                    self.exit()
        elif key == ord('2'):
            dataset_files = self.display_datasets(1)
            while True:
                key = self.stdscr.getch()
                if key == curses.KEY_UP and self.current_row > 1:
                    self.current_row -= 1
                elif key == curses.KEY_DOWN and self.current_row < len(dataset_files):
                    self.current_row += 1
                elif key == 10:  # Enter key
                    selected_dataset = dataset_files[self.current_row - 1]
                    # You can do further processing with the selected_dataset here
                    break
                elif key == ord('M') or key == ord('m'):
                    break
                elif key == 27:  # Escape key to exit
                    self.exit()
        elif key == ord('3'):
            return  # Return to the main menu
    Node_Trainer_Setup()

        

def train_neural_network(self, neural_network, input_data, target_labels, epochs, learning_rate, model_save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Forward pass
        outputs = neural_network(input_data)

        # Compute the loss
        loss = criterion(outputs, target_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring training progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

        # Update the target labels for the next iteration using the predictions
        if epoch < epochs - 1:
            target_labels = outputs.clone().detach()

    # Save the trained model to the specified file path
    torch.save(neural_network.state_dict(), model_save_path)


PT_NN_Trainer()