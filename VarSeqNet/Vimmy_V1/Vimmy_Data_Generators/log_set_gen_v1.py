import os
import csv
import numpy as np



def logarithmic_dataset_generator(num_samples, start, stop, base=10, noise_stddev=0.1, file_path="log_dataset.csv"):
    # Generate logarithmic values
    log_values = np.logspace(start, stop, num_samples, base=base)
    
    # Add optional noise
    if noise_stddev > 0:
        noise = np.random.normal(0, noise_stddev, num_samples)
        log_values += noise

    # Format the dataset as a list of lists
    dataset = [[value] for value in log_values]

    # Save the dataset to a CSV file
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(dataset)

    return log_values

# Define the folder name

folder_name = "Example_Data_Sets"

# Define the full path to the folder

folder_path = 'D:\VarSeqNet\Data_Sets\Example_Data_Sets'

# Create the folder if it doesn't exist

if not os.path.exists(folder_path):
    os.mkdir(folder_path)


# Example usage:
# num_samples = 100
# start = 5
# stop = 10
# base = 10
# noise_stddev = 0.1
# file_name = "logarithmic_dataset_V.csv"
# file_path = os.path.join(folder_path, file_name)
# Save the dataset to the specified folder
# log_dataset = generate_and_save_logarithmic_dataset(num_samples, start, stop, base, noise_stddev, file_path)
# print("Data-Set Generation Complete")
