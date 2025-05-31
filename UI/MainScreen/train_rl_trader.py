import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class RolloutCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_num = 0

    def _on_rollout_end(self) -> None:
        self.rollout_num += 1
        print(f"✅ Rollout #{self.rollout_num} завершён (эквивалент эпохи PPO).")

    def _on_step(self) -> bool:
        return True  # Нужно вернуть True, чтобы обучение продолжалось


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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60 * 6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        return self.dataset.iloc[self.current_index].to_numpy(dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False

        hold_multiplier = 1.0
        current_close = self.close_df.iloc[self.current_index][0]
        prev_portfolio_value = self.portfolio_value

        if action == 1 and self.position == 0:
            hold_multiplier = 1.0
            self.position = 1
            self.entry_price = current_close
        elif action == 2 and self.position == 1:
            profit = current_close - self.entry_price
            self.balance += profit
            self.position = 0
            self.entry_price = 0
            reward = profit * hold_multiplier
            hold_multiplier = 1.0
        elif action == 0 and self.position == 1:
            hold_multiplier *= 1.5

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


#dataset = np.load("your_dataset.npy")

file_name = "Prepared_60d/DASH_1d.csv"
dataset = pd.read_csv(file_name, sep=',')

env_fn = lambda: StockFixedWindowEnv(dataset)
vec_env = make_vec_env(env_fn, n_envs=1)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=1814,
    batch_size=32,
    n_epochs=10,
    learning_rate=1.1317339539623335e-05,
    gamma = 0.9788082086028914,
    gae_lambda=0.9715916406189629,
    ent_coef=0.001623459948568323
)

callback = RolloutCallback()

model.learn(total_timesteps=50000, callback=callback)

model.save("ppo_stock_model")

dataset = pd.read_csv(file_name, sep=',')
test_env = StockFixedWindowEnv(dataset)
obs, _ = test_env.reset()
done = False
portfolio_values = []
rewards = []
actions = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    actions.append(action)
    rewards.append(reward)
    portfolio_values.append(info["portfolio"])

positive_rewards = [r for r in rewards if r > 0]
negative_rewards = [r for r in rewards if r < 0]
win_rate = len(positive_rewards) / (len(positive_rewards) + len(negative_rewards) + 1e-8)

returns = np.array(rewards)
sharpe_ratio = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

sell_rewards = [r for a, r in zip(actions, rewards) if a == 2]
correct_sells = [r for r in sell_rewards if r > 0]
precision = len(correct_sells) / (len(sell_rewards) + 1e-8)

print("\n📊 Evaluation Metrics:")
print(f"Win Rate:       {win_rate:.2%}")
print(f"Sharpe Ratio:   {sharpe_ratio:.2f}")
print(f"Sell Precision: {precision:.2%}")

# === ГРАФИК ПОРТФЕЛЯ ===
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label="Portfolio Value")
plt.title("Portfolio Value Over Time")
plt.xlabel("Step")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("portfolio_plot.png")
plt.show()

