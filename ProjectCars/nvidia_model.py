import random
from collections import deque
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
import numpy as np
from tqdm import tqdm

previous_reward = 0
from get_screen import  make_move, start_screen


"""""""""
TO DO:
- Add tensorboard for reward tracking 
- Start Environment method
- save model only when reward is at peak 

"""""""""

np.random.seed(1000)


value_discount = 0.99
EPISODES = 10000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes


import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


REPLAY_MEMORY_SIZE = 128000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 128
UPDATE_TARGET_EVERY = 5
import time
MODEL_NAME = 'projectcars'


class DQNAgent:
    def create_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(60,80,3)))
        model.add(Conv2D(filters=24,kernel_size=(5,5), activation='elu', strides=(2, 2),padding='same'))
        model.add(Conv2D(filters=36,kernel_size=(5,5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(filters=48,kernel_size=(5,5), activation='elu', strides=(2, 2),padding='same'))
        model.add(Conv2D(filters=64,kernel_size=(3,3), activation='elu',padding='same'))
        model.add(Conv2D(filters=64,kernel_size=(3,3), activation='elu',padding='same'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(4))
        model.summary()

        # Compile the model
        model.compile(loss="mse", optimizer=Adam(lr=0.002), metrics=['accuracy'])

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

    def update_replay_memory(self, transition):
        for transitions in transition:
         self.replay_memory.append(transitions)

    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):


            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + value_discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

            # And append to our training data

            X.append(current_state)

            y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

          # Update target network counter every episode
        if terminal_state:
                self.target_update_counter += 1

            # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0


    def get_qs(self, state):
     return self.model.predict(np.array(state).reshape(-1,*state.shape))[0]
global action




##running the program

agent = DQNAgent()
count =0

global reward

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    episode_reward = 0



    done = False
    history = deque(maxlen=REPLAY_MEMORY_SIZE)
    index = 0
    current_state = start_screen()



    while not done :

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs((current_state)))
        else:
            action = np.random.randint(0, 4)

        new_state, reward = make_move(action)


        history.append((current_state, action, reward, new_state, done))
        episode_reward += reward
        current_state = new_state

        count = count + 1
        if count % 100 == 0:
            print(count)
            done = True

        if done:
            agent.update_replay_memory(history)
            agent.train(done)






    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode_reward > previous_reward:
        agent.model.save("C:\\Users\\myles.MSI\\Documents\\models\\models.h5")

    print(episode_reward)

    previous_reward = episode_reward









