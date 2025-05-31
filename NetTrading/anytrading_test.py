import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import Actions

from stable_baselines3 import A2C
import quantstats as qs

df = gym_anytrading.datasets.STOCKS_GOOGL.copy()

window_size = 50
start_index = window_size
end_index = len(df)

env = gym.make(
    'stocks-v0',
    df=df,
    window_size=window_size,
    frame_bound=(start_index, end_index)
)

print("observation_space:", env.observation_space)

env.reset(seed=2023)
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1_000)

action_stats = {Actions.Sell: 0, Actions.Buy: 0}
observation, info = env.reset(seed=2023)

while True:
    action, _states = model.predict(observation)

    action_stats[Actions(action)] += 1
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        break

env.close()

print("action_stats:", action_stats)
print("info:", info)

plt.figure(figsize=(16, 6))
env.unwrapped.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.unwrapped.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
#qs.reports.html(returns, output='SB3_a2c_quantstats.html')