import pandas as pd
import requests
import json
import seaborn as sns

class DataLoader:
    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    @staticmethod
    def load_api(url):
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data)

    @staticmethod
    def load_dataset(name):
        return sns.load_dataset('titanic')
