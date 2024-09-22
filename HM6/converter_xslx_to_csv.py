import pandas as pd

#--------------------
#Для ирины необходимо выполнить pip install pandas openpyxl
def xlsx_to_csv(xlsx_file, csv_file):
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False)
xlsx_to_csv('HM6/Resources/visits.xlsx', 'HM6/Resources/visits.csv')


#--------------------
#Для Ивана. Нужно определить кодировку и потом с помощью ее прочитать файл и пересохронить. 
import chardet
def csv_to_csv(csv_file, encoding_value, csv_file_new, encoding_value_new):
    df = pd.read_csv(csv_file, encoding=encoding_value)
    df.to_csv('HM6/Resources/data_from_ivan_encoded.csv', index=False, encoding='utf-8')

#читаем при помощи какой-то библиатеки файл и определяем кодировку. В дальнейшем ее будем использовать. Файл сохранен в windows-1251 кодировки.
with open('HM6/Resources/data_from_ivan.csv', 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']
print(f"Определённая кодировка: {encoding}")

#читаем и пересохроняем файл
csv_to_csv('HM6/Resources/data_from_ivan.csv', encoding, 'HM6/Resources/data_from_ivan_encoded.csv', 'utf-8')
#--------------------
