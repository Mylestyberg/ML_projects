from datetime import time

import carseour
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
import random
from collections import deque
from keras.layers import  Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from tqdm import tqdm
from keras.layers import Dense, Flatten, Input, merge, Lambda

from actor_model import create_actor_network
from critic_model import create_critic_network
from get_screen import start_screen, make_move

REPLAY_MEMORY_SIZE = 128000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 250
UPDATE_TARGET_EVERY = 5


np.random.seed(1000)


value_discount = 0.99
EPISODES = 10000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


actor = create_actor_network()
critic = create_critic_network()
history = deque(maxlen=REPLAY_MEMORY_SIZE)


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x










def update(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)



        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.critic.predict(current_states)


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


        self.critic.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)


        actions = self.actor.model.predict(minibatch[0])
        grads = self.critic.gradients(minibatch[0], actions)

        self.actor.train(minibatch[0], actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()


"""""""""
          # Update target network counter every episode
        if terminal_state:
                self.target_update_counter += 1

            # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

"""""""""



game = carseour.live()

prev_sector_time = [-1,-1,-1]


global reward

sector_count = 0



for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    done = False
    index = 0
    current_state = start_screen()
    noise = OrnsteinUhlenbeckProcess(size=4)


    if sector_count < 2:
        sector_count = sector_count + 1
    else:
        sector_count = 0

    while not done:
        action = np.argmax(actor.get_actor_policy((current_state)))
        # Clip continuous values to be valid w.r.t. environment
        action = np.clip(action + noise.generate(time), - 4, 4)
        new_state, reward = make_move(action)
        current_sector_time = [game.mCurrentSector1Time, game.mCurrentSector2Time, game.mCurrentSector3Time]

        if current_sector_time[sector_count] != prev_sector_time[sector_count]:
            done = True
            reward = 1
            prev_sector_time[sector_count - 1] = current_sector_time[sector_count - 1]
            print(sector_count)

        history.append((current_state, action, reward, new_state, done))


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














