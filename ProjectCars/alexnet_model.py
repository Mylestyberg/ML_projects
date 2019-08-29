import random

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
np.random.seed(1000)
#Instantiate an empty model


learning_rate = 0.9
value_discount = 0.95
EPISODES = 20_000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
winner =0
loser =0
draw = 0


#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes


import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 10
UPDATE_TARGET_EVERY = 5
import time
MODEL_NAME = 'projectcars'


class DQNAgent:
    def create_model(self):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(80,60,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

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
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.summary()

        # Compile the model
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])


    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        from collections import deque
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        for transitions in transition:
         self.replay_memory.append(transitions)

    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return


        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_qs_list = []
        future_qs_list =[]
        current_states = np.array( [ttt_board().reshape_for_cnn( transition[0]) for transition in minibatch])
        for c in current_states:
            current_qs_list.append(self.model.predict(c))


        new_current_states = np.array([ttt_board().reshape_for_cnn( transition[3]) for transition in minibatch])

        for n in new_current_states:
            future_qs_list.append(self.model.predict(n))




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
            current_qs = current_qs[0]
            current_qs[action]   = new_q

            # And append to our training data
            xx = ttt_board().reshape_for_cnn(current_state)
            X.append(xx[0])

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

    current_state = board.ttt_board().reset(aboard)
    done = False
    history = deque(maxlen=REPLAY_MEMORY_SIZE)
    index = 0

    while not done :
        can_place_piece = False

        done,reward, cboard= board.ttt_board().make_random_move(current_state)

        if not done:
        #dqn make move
            while not can_place_piece:
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(ttt_board().reshape_for_cnn(current_state)))
                else:
                    action = np.random.randint(0, 9)
                if ttt_board().check_if_position(action, current_state): ##if postion is taken
                     actions = agent.get_qs(ttt_board().reshape_for_cnn(current_state))
                     actions[0,action] = -1
                elif not  ttt_board().check_if_position(action, current_state):
                    can_place_piece = True
                    new_state, done, reward = board.ttt_board().make_move(action,current_state)






        history.append((current_state, action, reward, new_state, done))
        index+=1
        episode_reward += reward
        current_state = new_state.copy()


        if done:

            if reward == -1 or reward==0.5:
               remove= history.pop()
               append = history.pop()
               append = list(append)
               append[2] = reward
               append[4] = True
               history.append(tuple(append))


            if reward == 1:
                winner = winner + 1
            elif reward == -1 :
                loser = loser + 1
            else:
                draw = draw + 1






            agent.update_replay_memory(history)
            agent.train(done)






    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % 100==0:
        print("   ")
        print("   ")
        print(winner, loser, draw)
        print(winner - winnerprev, loser - loserprev)
        winnerprev = winner
        loserprev = loser


agent.model.save("C:\\Users\\myles.MSI\\Documents\\models\\models.h5")




