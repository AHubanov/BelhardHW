import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import gym
from collections import deque
import random

from tensorflow.python.keras.engine.training_arrays_v1 import model_iteration


class MyDQN:
    def __init__(self, env, epsilon, gamma, lr, epsilon_decay):
        self.env = env

        self.action_space = env.action_space
        self.num_action_space = self.action_space.n

        self.observation_space = env.observation_space
        self.num_observation_space = self.observation_space.shape[0]

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.counter = 0
        self.rewards_list = []

        self.replay_buffer = deque(maxlen = 500000)
        self.batch_size = 64

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.model = self.initialize_model()

        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def initialize_model(self):
        one = tf.keras.layers.Input(shape=[self.num_observation_space])
        input_layer = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)(one)
        middle_layer = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(input_layer)
        last_layer = tf.keras.layers.Dense(self.num_action_space, activation='linear')(middle_layer)
        model = tf.keras.Model(inputs=[one], outputs=[last_layer])
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.lr), loss = tf.keras.losses.MeanSquaredError())
        return model

    def get_action(self, state):
        if random.randrange(self.num_action_space) > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return np.random.randint(self.num_action_space)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_counter(self):
        self.counter+=1
        step_size = 5
        self.counter = self.counter % step_size

    def get_attributes_from_sample(self, sample):
        states = np.squeeze(np.squeeze(np.array([i[0] for i in sample])))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.squeeze(np.array([i[3] for i in sample]))
        done_list = np.array([i[4] for i in sample])
        return states, actions, rewards, next_states, done_list


    def update_model(self):
        if len(self.replay_buffer) < self.batch_size or self.counter != 0:
            return

        if np.mean(self.rewards_list[-10:]) > 180:
            return

        random_sample = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)

        targets = rewards + self.gamma * (np.max(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)


    def learn(self, num_episodes = 2000):
        for episode in range(num_episodes):
            state = self.env.reset()[0]

            reward_for_episode = 0
            num_steps = 500

            state = np.reshape(state, [1, self.num_observation_space])
            for step in range(num_steps):
                received_action = self.get_action(state)
                next_state, reward, terminated, done, info = env.step(received_action)
                self.env.render()

                next_state = np.reshape(next_state, [1, self.num_observation_space])
                self.add_to_replay_buffer(state, received_action, reward, next_state, done)
                reward_for_episode+=reward
                state = next_state

                self.update_counter()
                self.update_model()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            last_reward_mean = np.mean(self.rewards_list[-100:])
            if last_reward_mean > 200:
                print("DQN Training Complete....")
                break

            print(f"Episode: {episode} \n Reward: {reward_for_episode} \n Average Reward: {last_reward_mean} \n Epsilon: {self.epsilon}")

    def save(self, name):
        self.model.save(name)

def run_already_trained_model(modelfile_loc, num_episodes=100):
    reward_list = []
    num_test_episode = num_episodes
    env = gym.make('LunarLander-v2')
    print("Starting Testing of the trained model...")

    # Load the model
    trained_model = tf.keras.models.load_model(modelfile_loc)

    step_count = 5000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        print(f'current_state.shape = {current_state.shape}')
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode+=reward
            if done:
                break
        reward_list.append(reward_for_episode)
        print(f"{test_episode} : Episode || Reward: {reward_for_episode}")

    return reward_list


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000

    model = MyDQN(env, epsilon, gamma, lr, epsilon_decay)
    for i in range(10):
        for j in range(200):
            model.learn(training_episodes)
        model.save(f'LL1_model_{i*200}.h5')



#    reward_list = run_already_trained_model('LL1_model.h5', 2)
#    print(reward_list)
    env.close()
