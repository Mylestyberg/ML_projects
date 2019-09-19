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


REPLAY_MEMORY_SIZE = 128000

actor = create_actor_network()
critic = create_critic_network()
history = deque(maxlen=REPLAY_MEMORY_SIZE)


for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    episode_reward = 0

        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_dim])
        noise_t = np.zeros([1, action_dim])

        a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        # The following code do the stochastic brake
        # if random.random() <= 0.1:
        #    print("********Now we apply the brake***********")
        #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0])

        s_t1 = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

        # Do the batch update
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if (train_indicator):
            loss += critic.model.train_on_batch([states, actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states, a_for_grad)
            actor.train(states, grads)
            actor.target_train()
            critic.target_train()

        total_reward += r_t
        s_t = s_t1










