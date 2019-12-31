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

from OUNoise import OUNoise
from actor_model import create_actor_network
from critic_model import create_critic_network
from get_screen import start_screen, make_move
from utils.memory_buffer import MemoryBuffer

REPLAY_MEMORY_SIZE = 128000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

buffer = MemoryBuffer(128000)


np.random.seed(1000)


value_discount = 0.99
EPISODES = 10000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


actor = create_actor_network(4,4,0.001)

critic = create_critic_network(4,0.001,0.001)
history = deque(maxlen=REPLAY_MEMORY_SIZE)





def update_models(states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = actor.model.predict(states)
        grads = critic.gradients(states, actions)
        # Train actor
        actor.train(states, actions, np.array(grads).reshape((-1, 4)))
        # Transfer weights to target networks at rate Tau
        actor.transfer_weights()
        critic.transfer_weights()


def bellman(rewards, q_values, dones):
    critic_target = np.asarray(q_values)
    for i in range(q_values.shape[0]):
        if dones[i]:
            critic_target[i] = rewards[i]
        else:
            critic_target[i] = rewards[i] + 0.9 * q_values[i]

        return critic_target


game = carseour.live()

prev_sector_time = [-1,-1,-1]


global reward

sector_count = 0


def memorize(state, action, reward, done, new_state):
    """ Store experience in memory buffer
    """
    buffer.memorize(state, action, reward, done, new_state)


def sample_batch( batch_size):
    return buffer.sample_batch(batch_size)

count = 0

exploration_noise = OUNoise(4)

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    done = False
    index = 0
    current_state = start_screen()

    actions, states, rewards = [], [], []
    time =0





    while not done:
        action = actor.get_actor_policy(current_state)
        action_with_noise = action*10 + abs(exploration_noise.noise())
        new_state, reward = make_move(action_with_noise)
        count = count + 1
        memorize(current_state, action, reward, done,new_state)
        if count % 100 == 0:
            done = True
            states, actions, rewards, dones, new_states, _ = sample_batch(64)
            q_values = critic.target_predict([new_states, actor.target_predict(new_states)])
            critic_target = bellman(rewards, q_values, dones)
            # Train both networks on sampled batch, update target networks
            update_models(states, actions, critic_target)
            time += 1




















