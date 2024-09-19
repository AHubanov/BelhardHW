import pandas as pd
#необходимо выполнить pip install pandas openpyxl
def xlsx_to_csv(xlsx_file, csv_file):
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False)

xlsx_to_csv('HM6/Resources/visits.xlsx', 'HM6/Resources/visits.csv')