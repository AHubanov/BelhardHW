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

def file_exists(filename):
    return os.path.exists(filename)

def PrepareData():
    daysCountToDetermine = 60

    path = "Files_test/"
    resultPath = f"Prepared_{daysCountToDetermine}d/"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for index, file in enumerate(files):
        if "DS_Store" in file:
            continue
        print(f'{index}|{len(files)} {file} processing')
        if not file_exists(f'{resultPath}/{file}'):
            df = pd.read_csv(f'{path}{file}')
            normalized_dfs = []
            for x in range(daysCountToDetermine, len(df.index)):
                current_df = pd.DataFrame(df[x - daysCountToDetermine:x].values, columns=df.columns)
                one_line_df = normalize_data(current_df, daysCountToDetermine)
                normalized_dfs.append(one_line_df)
            final_df = pd.concat(normalized_dfs, ignore_index=True)
            final_df.to_csv(f'{resultPath}/{file}', index=False)
            print(f'{file} prepared')
        else:
            print(f'{file} already prepared')

def normalize_data(df, daysCountToDetermine):
    last_close = df["Close"].iloc[-1]
    X = 100 / last_close
    columns_to_normalize = [col for col in df.columns if col not in ["Volume", "Date"]]
    df[columns_to_normalize] = df[columns_to_normalize] * X
    volume_sum = df['Volume'].iloc[:daysCountToDetermine].sum()
    df['Volume'] = df['Volume'] / volume_sum
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.weekday

    df_new = pd.DataFrame(df.iloc[:daysCountToDetermine].values.flatten()).T
    columns = [f"{col}{i}" for i in range(daysCountToDetermine) for col in ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df_new.columns = columns
    df_new["ClosePrice"] = last_close
    return df_new

PrepareData()