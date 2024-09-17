import matplotlib.pyplot as plt

class DataVisualizer:
    
    @staticmethod
    def plot_histogram(data, column, bins=10):
        plt.figure(figsize=(10, 6))
        plt.hist(data[column], bins=bins, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def plot_line(data, x_col, y_col):
        plt.figure(figsize=(10, 6))
        plt.plot(data[x_col], data[y_col], marker='o')
        plt.title(f'Line Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_scatter(data, x_col, y_col):
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_col], data[y_col], alpha=0.7)
        plt.title(f'Scatter Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()

    @staticmethod
    def remove_plot():
        plt.close()