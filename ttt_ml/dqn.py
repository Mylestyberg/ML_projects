import random
from collections import deque
import  numpy as np
from tqdm import tqdm

from ttt_ml import board
from ttt_ml.board import ttt_board




from ttt_ml.tensor_board import ModifiedTensorBoard

learning_rate = 0.9
value_discount = 0.95
EPISODES = 20_000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes


import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 5
UPDATE_TARGET_EVERY = 5
import time
MODEL_NAME = 'TTT'

class DQNAgent:

    def create_model(self):
        model = Sequential()
        model.add((Dense(54, input_shape=(27,), activation='relu')))
        model.add(Dense(9,input_dim=54, activation='relu'))
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
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return


        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)


        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)


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

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

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
     return self.model.predict(np.array(state))
global action

agent = DQNAgent()


##Each episode will be a game and when finised, then check when to train with that info

aboard =  board.ttt_board.board
global reward
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    episode_reward = 0
    step = 1
    current_state = board.ttt_board().reset(aboard)
    done = False


    while not done :
        is_place = True


        action = -50

        done,new_rnd_state= board.ttt_board().make_random_move(current_state)

        if done:
            is_place= False
            current_state = new_rnd_state




        #dqn make move
        while is_place:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(ttt_board().reshape_for_nn(current_state)))
            else:
                action = np.random.randint(0, 9)
            if ttt_board().check_if_position(action, current_state):
                 actions = agent.get_qs(ttt_board().reshape_for_nn(current_state))
                 actions[0,action] = -1
            elif not  ttt_board().check_if_position(action, current_state):
                is_place = False

        if not done:
         new_state, reward, done = board.ttt_board().make_move(action,current_state)









        # Transform new continous state to new discrete state and count reward
        episode_reward += reward


        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1


    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

