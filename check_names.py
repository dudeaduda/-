import numpy as np
import joblib

# Чтение и загрузка данных из текстового файла
input_file_path = r"D:\курсач инф\для токенизации и векторизации\cleaned_data.txt"
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Создание словаря для хранения имен и соответствующих рейтингов
name_rating_dict = {}

for line in lines:
    parts = line.strip().split(': ')
    if len(parts) == 2:
        name_rating_dict[parts[0].strip()] = float(parts[1])

# Сохранение словаря с использованием joblib
joblib.dump(name_rating_dict, '/mnt/data/name_rating_dict.joblib')
