import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten
REPLAY_MEMORY_SIZE = 128000
MODEL_NAME = "critic"


TAU = 0.001
class create_critic_network():
    def create_critic_model(self):
            """ Assemble Critic network to predict q-values
            """
            state = Input((self.env_dim))
            action = Input((4,))
            x = Dense(256, activation='relu')(state)
            x = concatenate([Flatten()(x), action])
            x = Dense(128, activation='relu')(x)
            out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
            return Model([state, action], out)

    def __init__(self, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = (60,80,3)
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.create_critic_model()
        self.target_model = self.create_critic_model()
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]],
                                       K.gradients(self.model.output, [self.model.input[1]]))

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = TAU * critic_weights[i] + (1 - TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])
