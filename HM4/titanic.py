from data_loader import DataLoader
from data_processor import DataProcessor
from data_visualizer import DataVisualizer

data_loader = DataLoader()
data_processor = DataProcessor()

print("------------------------")

titanic_dataCSV = data_loader.load_csv('HM4/Resources/titanic.csv')
data_processor.report_missing_values(titanic_dataCSV)
titanic_dataCSV = data_processor.fill_missing_values(titanic_dataCSV, 'Age', method='median')

print("------------------------")

titanic_dataSET = data_loader.load_dataset('titanic')
data_processor.report_missing_values(titanic_dataSET)
titanic_dataSET = data_processor.fill_missing_values(titanic_dataSET, 'age', method='median')

print("------------------------")

titanic_dataJSON = data_loader.load_json('HM4/Resources/titanic.json')
data_processor.report_missing_values(titanic_dataJSON)
titanic_dataJSON = data_processor.fill_missing_values(titanic_dataJSON, 'Age', method='median')

print("------------------------")

visualizer = DataVisualizer()
visualizer.plot_histogram(titanic_dataSET, 'Age')
visualizer.plot_line(titanic_dataSET, 'Age', 'Fare')
visualizer.plot_scatter(titanic_dataSET, 'Age', 'Fare')


