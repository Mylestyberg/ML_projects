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
from get_screen import start_screen

REPLAY_MEMORY_SIZE = 128000


np.random.seed(1000)


value_discount = 0.99
EPISODES = 10000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


actor = create_actor_network()
critic = create_critic_network()
history = deque(maxlen=REPLAY_MEMORY_SIZE)


for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    done = False
    index = 0
    current_state = start_screen()

    while not done:

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














