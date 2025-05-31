import os
import boto3
from botocore.config import Config

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='b5fc2231-980b-49e4-b672-26119c29a7d6',
  aws_secret_access_key='ru0au21quqdqk5ZfmztEyWhtO45cBksA',
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.polygon.io',
  config=Config(signature_version='s3v4'),
)

# List Example
# Initialize a paginator for listing objects
paginator = s3.get_paginator('list_objects_v2')


prefix = 'us_stocks_sip'
bucket_name = 'flatfiles'
year_range = range(2025, 2020, -1)  # От 2025 до 2020 включительно

# Функция для создания структуры папок
def create_folder_structure(year, month):
    folder_path = f'./{year}/{month}'
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page['Contents']:
        key = obj['Key']
        if 'us_stocks_sip/minute_aggs_v1' in key:
            # Извлекаем год и месяц из имени файла
            parts = key.split('/')
            if len(parts) >= 4:
                year = parts[2]
                month = parts[3]


                if year.isdigit() and (int(year) > 2020 or (int(year) == 2020 and int(month) > 5)):
                    # Создаем структуру папок
                    folder_path = create_folder_structure(year, month)

                    # Локальный путь для сохранения файла
                    local_file_name = key.split('/')[-1]
                    local_file_path = os.path.join(folder_path, local_file_name)

                    # Скачиваем файл
                    if not os.path.exists(local_file_path):
                        # Скачиваем файл
                        s3.download_file(bucket_name, key, local_file_path)
                        print(f"Файл {local_file_name} сохранен в {local_file_path}")
                    else:
                        print(f"Файл {local_file_name} уже существует, пропускаем загрузку.")