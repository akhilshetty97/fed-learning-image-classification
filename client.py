import os
import numpy as np
import tensorflow as tf
import flwr as fl
from sklearn.model_selection import train_test_split 
from keras.preprocessing.image import load_img, img_to_array
import argparse
from CNNmodel import create_model  

# Function to load FEMNIST dataset for a client
def load_femnist_data(client_id: int):
    images = []
    labels = []
    client_folder = f"femnist_subset/client_{client_id}"
    # Labels range from 0 to 9
    for label in range(10):  
        label_folder = os.path.join(client_folder, str(label))
        for filename in os.listdir(label_folder):
            image_path = os.path.join(label_folder, filename)
            image = load_img(image_path, target_size=(32, 32), color_mode='grayscale')
            # Normalize to [0, 1]
            image_array = img_to_array(image) / 255.0  
            images.append(image_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Parse the "client_id"
parser = argparse.ArgumentParser(description="Flower client id for FEMNIST dataset.")
parser.add_argument('--client_id', type=int, required=True, help="The client ID (0-9) to specify which client's data to use.")
args = parser.parse_args()

# Load data for the specified client
x_data, y_data = load_femnist_data(client_id=args.client_id)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Model creation
model = create_model() 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
        
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10, batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start the Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
