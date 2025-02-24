# SimpleNet Project
## Project Overview
The SimpleNet project is a basic implementation of a feedforward neural network using PyTorch. It serves as an introductory example for understanding the structure and training of neural networks.
## File Descriptions

### model.py
In this file, the SimpleNet class is defined. This class inherits from nn.Module and represents a simple feedforward neural network with two fully connected layers. The first layer transforms input data from size 10 to size 5, followed by a ReLU activation function, and the second layer outputs a single value.
### train.py
This file contains the training loop for the neural network. It initializes the model, defines the loss function and optimizer, and iterates over the training dataset for a specified number of epochs. During each epoch, it computes the loss and updates the model's parameters.
### save_load_model.py
This file provides functionality to save and load the trained model. It demonstrates how to save the model's state dictionary after training and how to load it back for making predictions.
### data_loading.py
In this file, a custom dataset class is defined using PyTorch's Dataset API. It allows for easy loading of data samples and their corresponding labels. Additionally, it creates a DataLoader to handle batching and shuffling of data during training.
### main.py
This is the main entry point of the project. It ties together all components by performing tensor operations, creating an instance of the neural network, preparing dummy data, training the model, saving it, loading it back, and making predictions with random input.
## Installation Instructions
To set up the SimpleNet project on your local machine, follow these steps:
1. Install Python: Ensure you have Python (version 3.x) installed on your system.
2. Install PyTorch: You can install PyTorch via pip.
3. Install NumPy: If you don't have NumPy installed
Use the following commands
```
pip install torch torchvision
pip install numpy
```
4. Clone or Download the Project: Obtain a copy of this project by cloning it from its repository or downloading it as a ZIP file.
5. Run the Main Script: Navigate to the project directory in your terminal or command prompt and execute:
```
python main.py
```
This will run the entire workflow of creating tensors, training the model, saving it, loading it back, and making predictions.


