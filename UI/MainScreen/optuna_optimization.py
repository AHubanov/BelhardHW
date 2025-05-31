import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class StockFixedWindowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset, initial_balance=10000):
        super(StockFixedWindowEnv, self).__init__()
        self.close_df = dataset[['ClosePrice']].copy()
        self.dataset = dataset.drop(columns=['ClosePrice'])
        self.initial_balance = initial_balance
        self.num_entries = len(dataset)

        self.current_index = 0
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = initial_balance
        self.hold_multiplier = 1.0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60 * 6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        self.hold_multiplier = 1.0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.dataset.iloc[self.current_index].to_numpy(dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False

        current_close = self.close_df.iloc[self.current_index][0]
        prev_portfolio_value = self.portfolio_value

        if action == 1 and self.position == 0:
            self.hold_multiplier = 1.0
            self.position = 1
            self.entry_price = current_close
        elif action == 2 and self.position == 1:
            profit = current_close - self.entry_price
            self.balance += profit
            self.position = 0
            self.entry_price = 0
            reward = profit * self.hold_multiplier
            self.hold_multiplier = 1.0
        elif action == 0 and self.position == 1:
            self.hold_multiplier *= 1.5

        unrealized = (current_close - self.entry_price) if self.position == 1 else 0
        self.portfolio_value = self.balance + unrealized

        if self.position == 1:
            reward += self.portfolio_value - prev_portfolio_value  # Награда за рост

        self.current_index += 1
        done = self.current_index >= self.num_entries

        if done and self.position == 1:
            final_profit = current_close - self.entry_price
            self.balance += final_profit
            self.position = 0
            self.entry_price = 0
            reward += final_profit

        obs = np.zeros_like(self.observation_space.low) if done else self._get_observation()

        info = {
            "portfolio": self.portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "action": action
        }

        if done:
            print(f'{self.balance}')

        return obs, reward, done, False, info

    def render(self):
        print(f"Step {self.current_index} | Balance: {self.balance:.2f} | Position: {self.position} | Portfolio: {self.portfolio_value:.2f}")


# Создаём векторизированную обёртку
def make_env():
    file_name = "Prepared_60d/DASH_1d.csv"
    dataset = pd.read_csv(file_name, sep=',')

    # Здесь можно передать путь к вашему датасету
    env = StockFixedWindowEnv(dataset)
    return env

# Функция оптимизации
def optimize_agent(trial):
    # Гиперпараметры, которые будет подбирать Optuna:
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.85, 0.99)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-4, 1e-1)
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_int('n_steps', 256, 2048)

    # Обёртка над окружением
    env = DummyVecEnv([make_env])

    # Инициализация модели с текущими параметрами
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=0
    )

    # Обучение на небольшой выборке (для быстрой оценки)
    model.learn(total_timesteps=10_000)

    # Оценка производительности
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    return mean_reward

# Запуск Optuna
study = optuna.create_study(direction="maximize")
study.optimize(optimize_agent, n_trials=30)

# Вывод лучших параметров
print("Best trial:")
print(study.best_trial.params)
