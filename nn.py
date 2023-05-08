from gameEnv import CarRacingEnv
from ple import PLE
import time
import numpy as np
import gym
import os.path
from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model



learning_rate = 0.001
epsilon = 1e-5
decay_rate = 0.90
gamma = 0.99  # factor to discount reward

resume = False
render = True


def build_model():
        model = Sequential()
        model.add(Reshape((7,), input_shape=(7, 1)))
        # Miks RELU ei tööta nii hästi
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(25, activation="tanh"))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer=RMSprop(lr=learning_rate), metrics=["accuracy"],
                       loss="categorical_crossentropy")
        return model

if resume and os.path.isfile('model_pole.h5'):
    model = load_model('model_pole.h5')
else:
    model = build_model()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# PLE SPECIFIC
def process_state(state):
    return np.array(list(state.values())).flatten()
game = CarRacingEnv()
p = PLE(game, fps=30, state_preprocessor=process_state, display_screen=render)
p.init()
i = 0

p.act(p.NOOP)
observation = p.getGameState()

observations, taken_actions, rewards = [], [], []

reward_sum = 0
episode_number = 0

possible_actions = game.actions

last_n_scores = []


while True:

    x = observation
    a_probs = model.predict_on_batch(np.reshape(x, (1, 7, 1))).flatten()
    prob = a_probs / np.sum(a_probs)
    action = np.random.choice(2, 1, p=prob)[0]

    p.act(possible_actions[action])

    observation = p.getGameState()
    reward = 1
    done = p.game_over()

    taken_action = np.zeros([2])
    taken_action[action] = 1

    taken_actions.append(taken_action)
    observations.append(x)
    rewards.append(reward)
    reward_sum += reward

    if done:
        episode_number += 1

        taken_actions = np.vstack(taken_actions)
        rewards = np.vstack(rewards)
        rewards = discount_rewards(rewards)

        # ?????
        advantage = rewards - np.mean(rewards)
        # print("\nAdvantage: ", advantage)

        X = np.reshape(observations, (len(observations), 7, 1))
        Y = taken_actions

        model.train_on_batch(X, Y, sample_weight=advantage.flatten())

        observations, taken_actions, rewards = [], [], []  # reset array memory

        if len(last_n_scores) >= 1:
            print("Average score: " + str(np.average(last_n_scores[0])))
            with open("history_racing_old.txt", "a+") as data:
                data.write(str(episode_number) + ", " + str(np.average(last_n_scores[0])) + ", " + str(time.time()) + "\n")
            last_n_scores = []
        else:
            last_n_scores.append(reward_sum)


        reward_sum = 0
        # if episode_number % 1 == 0 and resume:
        #     #model.save_weights('model_pole_weights.h5')
        #     model.save('model_old.h5')

        p.reset_game()
        p.act(p.NOOP)






















