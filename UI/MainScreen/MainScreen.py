import os
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import numpy as np
import requests
from io import StringIO
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

def file_exists(filename):
    return os.path.exists(filename)

def fetch_data(ticker, time_interval_str, start_date, end_date):
    filename = f"{ticker}_{time_interval_str}.csv"
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        existing_data.index = pd.to_datetime(existing_data.index)  # Приводим индекс к Timestamp
    else:
        existing_data = pd.DataFrame()

    if not existing_data.empty:
        existing_start = existing_data.index.min()
        existing_end = existing_data.index.max()
    else:
        existing_start, existing_end = None, None

    missing_start = start_date if existing_start is None or start_date < existing_start else None
    missing_end = end_date if existing_end is None or end_date > existing_end else None

    new_data = pd.DataFrame()
    if missing_start or missing_end:
        new_data = yf.download(ticker, start=start_date, end=end_date)
        new_data.index = pd.to_datetime(new_data.index)  # Приводим индекс к Timestamp

    combined_data = pd.concat([existing_data, new_data]).drop_duplicates().sort_index()

    combined_data.to_csv(filename)

    print(combined_data[['Open']])

    return combined_data.loc[start_date:end_date]

def fetch(ticker, time_interval_str, start_date, end_date):
    filename = f"Files/{ticker}_{time_interval_str}.csv"
    # normalized_filename = f"norm_{ticker}_{time_interval_str}.csv"
    if not file_exists(filename):
        new_data = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        new_data.to_csv(filename, index=True)

    # last_close = new_data["Close"].iloc[-1]
    # X = 100 / last_close
    # columns_to_normalize = [col for col in new_data.columns if col not in ["Volume", "Date"]]
    # new_data[columns_to_normalize] = new_data[columns_to_normalize] * X
    # new_data.to_csv(normalized_filename, index=True)
    #
    # df = pd.read_csv(filename, index_col=0, parse_dates=True)
    # norm_df = pd.read_csv(normalized_filename, index_col=0, parse_dates=True)
    #
    # required_cols = {"Open", "High", "Low", "Close", "Volume"}
    # if not required_cols.issubset(df.columns):
    #     raise ValueError(
    #         f"Файл {filename} имеет некорректный формат. Отсутствуют колонки {required_cols - set(df.columns)}")
    #
    # mpf.plot(df, type="candle", volume=True, style="charles")
    # mpf.plot(norm_df, type="candle", volume=True, style="charles")

def plot_prices():
    date1 = date_entry1.get_date().strftime("%Y-%m-%d")
    date2 = date_entry2.get_date().strftime("%Y-%m-%d")

    fetch_and_plot("MSFT", "1d","2020-01-01", "2025-01-01")
    #data = fetch_data('MSFT', '1d', date1, date2)
    # if not data.empty:
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     mpf.plot(data, type='candle', ax=ax)
    #
    #     global canvas
    #     if canvas:
    #         canvas.get_tk_widget().destroy()
    #     canvas = FigureCanvasTkAgg(fig, master=frame)
    #     canvas.get_tk_widget().pack()

def getSP():
    df_read = pd.read_csv("russell2000_tickers.csv")
    tickers_list = df_read["Ticker"].tolist()

    for ticker in tickers_list:
        print(ticker)
        fetch(ticker, "1d", "2020-01-01", "2025-01-01")

def PrepareData():
    daysCountToDetermine = 50
    daysCountPredict = 5

    path = "Files/"
    resultPath = "Prepared_50d/"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for index, file in enumerate(files):
        if "DS_Store" in file:
            continue
        print(f'{index}|{len(files)} {file} processing')
        if not file_exists(f'{resultPath}/{file}'):
            df = pd.read_csv(f'{path}{file}')
            normalized_dfs = []
            for x in range(daysCountToDetermine, len(df.index) - daysCountPredict):
                current_df = pd.DataFrame(df[x - daysCountToDetermine:x + daysCountPredict].values, columns=df.columns)
                one_Line_df = normalize_data(current_df, daysCountToDetermine, daysCountPredict)
                normalized_dfs.append(one_Line_df)
            final_df = pd.concat(normalized_dfs, ignore_index=True)
            final_df.to_csv(f'{resultPath}/{file}', index=False)
            print(f'{file} prepared')
        else:
            print(f'{file} already prepared')

def normalize_data(df, daysCountToDetermine, daysCountPredict):
    last_close = df["Close"].iloc[-(daysCountPredict + 1)]
    X = 100 / last_close
    columns_to_normalize = [col for col in df.columns if col not in ["Volume", "Date"]]
    df[columns_to_normalize] = df[columns_to_normalize] * X
    volume_sum = df['Volume'].iloc[:daysCountToDetermine].sum()
    df['Volume'] = df['Volume'] / volume_sum
    subset = df.iloc[daysCountToDetermine:daysCountToDetermine + daysCountPredict][['Close', 'High', 'Low', 'Open']]
    min_value = subset.min().min()
    max_value = subset.max().max()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.weekday

    df_new = pd.DataFrame(df.iloc[:daysCountToDetermine].values.flatten()).T
    columns = [f"{col}{i}" for i in range(daysCountToDetermine) for col in ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df_new.columns = columns
    df_new['min_value'] = min_value
    df_new['max_value'] = max_value

    return df_new

input_size = 300
hidden_sizes = [2048, 1024, 1024, 256]
output_size = 1
n_epochs = 51

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

    # def predict(self, x):
    #     with torch.no_grad():
    #         output = self.forward(x)
    #         output = torch.round(output)
    #     return output

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def training():
    model = NeuralNet()
    model.reset_weights()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    resultPath = "Prepared_50d_filtered/"
    files = [f for f in listdir(resultPath) if isfile(join(resultPath, f))]

    dataframes = []

    for file in files:
        print(f"Добавляем файл: {file}")
        df = pd.read_csv(f'{resultPath}/{file}', sep=',')
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    rewardMin = -20
    rewardFlat = -10
    rewardWin = 60
    df['Reward'] = df.apply(
        lambda row: rewardMin if row['min_value'] < 98 else
        rewardFlat if row['min_value'] > 98 and row['max_value'] < 110 else
        rewardWin if row['max_value'] >= 110 else 0,
        axis=1
    )

    n = (df['Reward'] == rewardWin).sum()
    df_30 = df[df['Reward'] == rewardWin]
    df_neg10 = df[df['Reward'] == rewardMin].sample(n=2 * n, random_state=42)
    df_neg1 = df[df['Reward'] == rewardFlat].sample(n=2 * n, random_state=42)
    df = pd.concat([df_30, df_neg10, df_neg1], ignore_index=True)

    df_reward = df[['Reward']].copy()
    df.drop(columns=['max_value', 'min_value', 'Reward'], inplace=True)

    data_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    rewards_tensor = torch.tensor(df_reward.to_numpy(), dtype=torch.float32)

    for epoch in range(n_epochs):
        predictions = model(data_tensor)

        model_name = f"Model/model_epoch_{epoch + 1}.pth"
        if epoch % 10 == 0 and epoch > 0:
            range_00_01 = ((predictions >= 0.0) & (predictions <= 0.1)).sum().item()
            range_01_03 = ((predictions > 0.1) & (predictions < 0.3)).sum().item()
            range_03_05 = ((predictions >= 0.3) & (predictions <= 0.5)).sum().item()
            range_05_07 = ((predictions > 0.5) & (predictions <= 0.7)).sum().item()
            range_07_09 = ((predictions > 0.7) & (predictions < 0.9)).sum().item()
            range_07_10 = ((predictions >= 0.9) & (predictions <= 1.0)).sum().item()

            torch.save(model.state_dict(), model_name)
            print(f"Epoch {epoch}: .0-.1: {range_00_01}, .1-.3: {range_01_03}, .3-.5: {range_03_05} .5-.7:{range_05_07} .7-.9: {range_07_09} .9-1.: {range_07_10}")

        # if epoch % 200 == 0 and epoch > 0:
        #     model_eval = NeuralNet()
        #     model_eval.load_state_dict(torch.load(model_name))
        #     model_eval.eval()
        #
        #     with torch.no_grad():
        #         predictions_eval = model_eval(data_tensor)
        #
        #     preds = predictions_eval.cpu().numpy().flatten()
        #     preds_step = preds.max() / 10
        #     bins = np.arange(0, preds_step*11, preds_step)
        #     digitized = np.digitize(preds, bins) - 1
        #
        #     fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        #     axs = axs.flatten()
        #
        #     for i in range(10):
        #         bin_preds = preds[(digitized == i)]
        #
        #         axs[i].hist(bin_preds, bins=20, color='skyblue', edgecolor='black')
        #         axs[i].set_title(f"Interval {bins[i]:.1f} - {bins[i + 1]:.1f}")
        #         axs[i].set_xlim(0, 1)
        #         axs[i].set_xlabel("Prediction")
        #         axs[i].set_ylabel("Count")
        #
        #     plt.tight_layout()
        #     plt.show()

            # adjusted_predictions = torch.where(predictions_eval < 0.5, torch.zeros_like(predictions_eval),
            #                                    torch.ones_like(predictions_eval))
            #
            # zero_count = (adjusted_predictions == 0).sum().item()
            # one_count = (adjusted_predictions == 1).sum().item()
            # print(f"Epoch {epoch}: zero {zero_count}: one {one_count}")
            #
            # zero_predictions = adjusted_predictions == 0
            # one_predictions = adjusted_predictions == 1
            #
            # zero_success = ((rewards_tensor == -10) & zero_predictions).sum().item()
            # zero_flat = ((rewards_tensor == -1) & zero_predictions).sum().item()
            # zero_mistake = ((rewards_tensor == 60) & zero_predictions).sum().item()
            #
            # one_mistake = ((rewards_tensor == -10) & one_predictions).sum().item()
            # one_flat = ((rewards_tensor == -1) & one_predictions).sum().item()
            # one_success = ((rewards_tensor == 60) & one_predictions).sum().item()
            #
            # batch_rewards = adjusted_predictions * rewards_tensor
            # total_reward = batch_rewards.sum().item()
            #
            # print(f"model: {model_name} reward:{total_reward}")
            # print(f"  Zero Success: {zero_success}")
            # print(f"  Zero Flat: {zero_flat}")
            # print(f"  Zero Mistake: {zero_mistake}")
            # print(f"  One Mistake: {one_mistake}")
            # print(f"  One Flat: {one_flat}")
            # print(f"  One Success: {one_success}")

        batch_rewards = predictions * rewards_tensor
        total_reward = batch_rewards.sum()
        loss = criterion(predictions, (rewards_tensor + 1) / 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with np.printoptions(precision=2):
            print(f'Epoch: {epoch}, Reward:{total_reward.item()} Loss:{loss}')

def evaluate_models():
    resultPath = "Prepared_not500/"
    files = [f for f in listdir(resultPath) if isfile(join(resultPath, f))]

    dataframes = []

    for file in files:
        print(f"{file}")
        df = pd.read_csv(f'{resultPath}/{file}', sep=',')
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    df['Reward'] = df.apply(
        lambda row: (row['max_value'] + row['min_value']) / 2 - 100 if row['min_value'] < 98 else
        (row['max_value'] + row['min_value']) / 2 - 100 if row['min_value'] > 98 and row['max_value'] < 110 else
        10  if row['max_value'] >= 110 else 0,
        axis=1
    )

    df_reward = df[['Reward']].copy()
    df.drop(columns=['max_value', 'min_value', 'Reward'], inplace=True)

    data_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    rewards_tensor = torch.tensor(df_reward.to_numpy(), dtype=torch.float32)

    model_dir = "Model"
    model_files = [os.path.join(model_dir, f"model_epoch_{i}.pth") for i in range(11, 412, 10)]

    results = []

    for model_file in model_files:
        model = NeuralNet()
        model.load_state_dict(torch.load(model_file))
        model.eval()

        with torch.no_grad():
            predictions = model(data_tensor)

        adjusted_predictions = torch.where(predictions < 0.999999, torch.zeros_like(predictions), torch.ones_like(predictions))

        #zero_predictions = adjusted_predictions == 0
        one_predictions = adjusted_predictions == 1

        # zero_success = ((rewards_tensor == -2) & zero_predictions).sum().item()
        # zero_flat = ((rewards_tensor != -2) & (rewards_tensor != 10) & zero_predictions).sum().item()
        # zero_mistake = ((rewards_tensor == 10) & zero_predictions).sum().item()
        #
        # one_mistake = ((rewards_tensor == -2) & one_predictions).sum().item()
        # one_flat = ((rewards_tensor != -2) & (rewards_tensor != 10) & one_predictions).sum().item()
        one_success = ((rewards_tensor == 10) & one_predictions).sum().item()

        batch_rewards = adjusted_predictions * rewards_tensor
        total_reward = batch_rewards.sum().item()

        results.append({"model": model_file, "reward": total_reward})
        print(f"model: {model_file} reward:{total_reward}")
        print(f"  Total Ones: {one_predictions.sum().item()}")
        # print(f"  Zero Success: {zero_success}")
        # print(f"  Zero Flat: {zero_flat}")
        # print(f"  Zero Mistake: {zero_mistake}")
        # print(f"  One Mistake: {one_mistake}")
        # print(f"  One Flat: {one_flat}")
        print(f"  One Success: {one_success}")
        print(f"  Avg Profit: {total_reward / (one_predictions.sum().item() + 1)}")
        print(f"  Score: {one_success/(one_predictions.sum().item() + 1)}")


def CopyFiles():
    # Путь к папке с файлами
    model_dir = "Prepared_50d"
    output_dir = "Prepared_not500"
    tickers_file = "sp500_tickers.csv"
    os.makedirs(output_dir, exist_ok=True)

    sp500_tickers = pd.read_csv(tickers_file)["Ticker"].tolist()

    for filename in os.listdir(model_dir):
        if filename.endswith("_1d.csv"):
            ticker = filename.split("_")[0]

            if ticker not in sp500_tickers:
                src_path = os.path.join(model_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)


#training()
evaluate_models()

#CopyFiles()

# root = tk.Tk()
# frame = tk.Frame(root)
# frame.pack(pady=10)
# plot_button = tk.Button(root, text="Do it", command=training)
# plot_button.pack(pady=10)
# root.mainloop()
