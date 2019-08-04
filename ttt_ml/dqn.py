from collections import deque

import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 100

class DQNAgent:

    def create_model(self):
        model = Sequential()
        model.add(Dense(54, input_dim=27, activation='relu'))
        model.add(Dense(9, input_dim=54, activation='relu'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0



