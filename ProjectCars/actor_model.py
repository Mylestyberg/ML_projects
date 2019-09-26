
from collections import deque
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, np

from keras.layers import  Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


REPLAY_MEMORY_SIZE = 128000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 250
UPDATE_TARGET_EVERY = 5
import time
MODEL_NAME = 'actor'



class create_actor_network():
    tau = 0.001
    def create_actor_model(self):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(60,80,3), kernel_size=(11,11), strides=(4,4), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(4))
        model.add(Activation('softmax'))

        model.summary()

        # Compile the model
        model.compile(loss="mse", optimizer=Adam(lr=0.002), metrics=['accuracy'])

        return model


    def __init__(self):

        # Main model
        self.model = self.create_actor_model()

        # Target network
        self.target_model = self.create_actor_model()

        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def get_actor_policy(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]




