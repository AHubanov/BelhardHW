import pandas as pd

class DataProcessor:
    
    @staticmethod
    def count_missing_values(data):
        return data.isnull().sum()

    @staticmethod
    def report_missing_values(data):
        missing_values = data.isnull().sum()
        missing_report = pd.DataFrame({
            'Column': data.columns,
            'Missing Values': missing_values,
            'Percentage': (missing_values / len(data)) * 100
        })
        print(missing_report)

    @staticmethod
    def fill_missing_values(data, column, method='mean'):
        if method == 'mean':
            value = data[column].mean()
        elif method == 'median':
            value = data[column].median()
        elif method == 'mode':
            value = data[column].mode()[0]
        else:
            raise ValueError("Неверный метод. Используйте 'mean', 'median' или 'mode'.")
        
        data[column].fillna(value, inplace=True)
        return data
