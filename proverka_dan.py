import re
import pickle

def clean_data(input_file_path, output_file_path):
    # Открываем исходный файл для чтения
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    clean_lines = []

    # Регулярное выражение для проверки допустимости символов в имени
    # Паттерн допускает буквы, цифры, пробелы и дефисы
    valid_name_pattern = re.compile(r'^[\w\s-]+$', re.UNICODE)

    for line in lines:
        # Разделяем строку на имя и рейтинг по символу ':'
        parts = line.strip().split(':')
        if len(parts) == 2:
            name, rating = parts[0].strip(), parts[1].strip().replace(',', '.')  # Замена запятой на точку для корректного преобразования в float
            try:
                float_rating = float(rating)  # Попытка преобразовать рейтинг в число
                # Проверка имени на соответствие паттерну
                if valid_name_pattern.match(name):
                    clean_lines.append(f"{name}: {rating}\n")  # Сохранение строки, если имя корректно
                else:
                    # Сообщение об ошибке, если имя содержит недопустимые символы
                    print(f"Skipping line, invalid characters in name: {line.strip()}")
            except ValueError:
                # Сообщение об ошибке, если рейтинг не является числом
                print(f"Skipping line, invalid rating found: {line.strip()}")
        else:
            # Сообщение об ошибке, если строка не содержит имя или рейтинг
            print(f"Skipping line, missing name or rating: {line.strip()}")

    # Сохраняем очищенные данные в новый файл
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(clean_lines)
    print("Data cleaning complete. Output saved to:", output_file_path)

# Пути к исходному и целевому файлам
input_path = "D:\\курсач инф\\для токенизации и векторизации\\обработанный рейтинг в тексте.txt"
output_path = "D:\\курсач инф\\для токенизации и векторизации\\cleaned_data.txt"

# Вызов функции для очистки данных
clean_data(input_path, output_path)
