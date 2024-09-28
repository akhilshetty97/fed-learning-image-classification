import tensorflow as tf
from keras.initializers import GlorotUniform

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=GlorotUniform(), input_shape=(32, 32, 1)),  
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=GlorotUniform()),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=GlorotUniform()),
        tf.keras.layers.Dense(10, activation='softmax')  
    ])
    return model
