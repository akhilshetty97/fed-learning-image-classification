Introduction:  
Federated Learning (FL) is a type of machine learning where the model is trained directly on multiple devices (clients) that hold local data, rather than moving the data to a central location (server). Instead of sending raw data to a central server, only the model updates which can be gradients are transmitted to a central server for aggregation. This helps in maintaining data privacy as the data stays on the local devices. This approach was first discussed in 2016, when a paper was published by Google AI.  

Goal:  
Simulation of a federated learning setup using the Flower framework, where a Convolutional Neural Network (CNN) is trained on a subset of the Federated Extended MNIST (FEMNIST) dataset. Each client holds its own local data (handwritten digit images) and trains the model locally, sharing only the model updates with the central server. The server aggregates these updates using the Federated Averaging (FedAvg) strategy to create a global model. This approach simulates a real-world federated learning environment, where data privacy is maintained as raw data remains on the clients, while the server orchestrates the overall model training process.

Dataset:  
For this project, a subset of the Federated Extended MNIST (FEMNIST) dataset has been used, which contains grayscale images of handwritten digits (0–9). The data is distributed across multiple clients in an IID (Independent and Identically Distributed) manner, with each client folder containing images of digits organized by label. Each client folder has subdirectories for each digit (0-9), and inside these are the respective images. This setup simulates a federated learning scenario where clients train models locally on their own data before contributing updates to a central server.

Model Architecture:  
A CNN model has been trained from scratch. The input is of grayscale images of size 32x32. The first layer has 32 filters of size 3x3. The activation function used here is ReLU. This layer extracts edges and textures from the image. The second layer has 64 filters of size 3x3, which is activated by ReLU as well. After each layer, a max pooling layer of 2x2 is applied to reduce the spatial complexity. The layers are then flattened into a 1D array to be processed further. There are 2 further Dense layers. The first dense layer has 128 neurons with ReLU activation and the other layer is the output layer. The output has 10 neurons with softmax activation for the 10 digits (0-9) from the FEMNIST dataset.  

Model weight initialization:  
The model weights are initialized using the Xavier initialization which is implemented using GlorotUniform(). It is directly used from the keras initializers library.  

Broadcast:  
After the initial model is developed, it is shared with all participating clients. Each client then fine-tunes the model using its local FEMNIST data, contributing to the global model in the federated framework. The server collects the updates from various clients to enhance the global model. FedAvg has been used because it enables efficient training of the global model. FedAvg reduces communication overhead and also ensures robust performance.
