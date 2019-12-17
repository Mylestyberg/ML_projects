
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



import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten



class create_actor_network():



    def __init__(self,  out_dim, act_range, lr):
        self.env_dim = (60,80,3)
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = 0.001
        self.lr = lr
        self.model = self.create_actor_model()
        self.target_model = self.create_actor_model()
        self.adam_optimizer = self.optimizer()
    def create_actor_model(self):
        inp = Input((60,80,3))


        # 1st Convolutional Layer
        x = (Conv2D(filters=96, input_shape=(60,80,3), kernel_size=(11,11), strides=(4,4), padding='same'))(inp)
        x = GaussianNoise(1.0)(x)
        x = Activation('relu')(x)
        # Max Pooling
        x = (MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))(x)

        # 2nd Convolutional Layer
        x =(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))(x)
        x = GaussianNoise(1.0)(x)
        x =(Activation('relu'))(x)
        # Max Pooling
        x =(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))(x)

        # 3rd Convolutional Layer
        x =(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))(x)
        x = (Activation('relu'))(x)


        # 4th Convolutional Layer
        x= (Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))(x)
        x = GaussianNoise(1.0)(x)
        x =(Activation('relu'))(x)

        # 5th Convolutional Layer
        x =(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))(x)
        x = GaussianNoise(1.0)(x)
        x =(Activation('relu'))(x)
        # Max Pooling
        x = (MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))(x)

        # Passing it to a Fully Connected layer
        x = (Flatten())(x)
        # 1st Fully Connected Layer
        x = (Dense(4096, input_shape=(224*224*3,)))(x)
        x = GaussianNoise(1.0)(x)
        x = (Activation('relu'))(x)
        # Add Dropout to prevent overfitting
        x = (Dropout(0.4))(x)

        # 2nd Fully Connected Layer
        x = (Dense(4096))(x)
        x = GaussianNoise(1.0)(x)
        x =Activation('relu')(x)
        # Add Dropout
        x = (Dropout(0.4))(x)

        # 3rd Fully Connected Layer
        x = Dense(1000)(x)
        x = GaussianNoise(1.0)(x)
        x = (Activation('relu'))(x)
        # Add Dropout
        x =(Dropout(0.4))(x)

        # Output Layer
        x =(Dense(4))(x)
        x = GaussianNoise(1.0)(x)
        out = (Activation('sigmoid'))(x)

        return Model(inp, out)

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])


    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def get_actor_policy(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)




