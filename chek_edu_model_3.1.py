import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка модели нейронной сети
model = load_model('/mnt/data/best_trained_model1.keras')

# Чтение и загрузка данных из текстового файла
input_file_path = r"D:\курсач инф\для токенизации и векторизации\cleaned_data.txt"
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

names = []
ratings = []

# Извлечение имен и рейтингов из каждой строки файла
for line in lines:
    parts = line.strip().split(': ')
    if len(parts) == 2:
        names.append(parts[0])
        ratings.append(float(parts[1]))

# Сохранение минимального и максимального значений для денормализации
min_rating = min(ratings)
max_rating = max(ratings)

# Посимвольная векторизация имен
char_vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=500,             # Максимальное количество уникальных токенов (символов)
    output_mode='int',          # Выходной формат - целые числа
    output_sequence_length=30,  # Фиксированная длина выходной последовательности
    split='character'           # Разбиение текста на символы
)
char_vectorize_layer.adapt(names)

# Проверка векторизации на первых 5 именах
sample_names = names[:5]
sample_vectorized_names = char_vectorize_layer(np.array(sample_names))
print("Sample names:", sample_names)
print("Vectorized names:", sample_vectorized_names.numpy())

def get_name_rating(name_input):
    # Векторизация имени
    name_vector = char_vectorize_layer([name_input])
    # Предсказание рейтинга моделью
    rating_prediction = model.predict(name_vector)
    # Денормализация предсказанного рейтинга
    denormalized_rating = rating_prediction[0][0] * (max_rating - min_rating) + min_rating
    return denormalized_rating

def query_names():
    # Запрос ввода имен от пользователя
    names = input("Введите имена через запятую для получения среднего рейтинга (или 'exit' для выхода): ")
    if names.lower() == 'exit':
        return
    name_list = names.split(',')
    # Получение рейтингов для введенных имен
    ratings = [get_name_rating(name.strip()) for name in name_list]
    valid_ratings = [rating for rating in ratings if rating is not None]
    if valid_ratings:
        # Вычисление среднего рейтинга
        average_rating = np.mean(valid_ratings)
        print(f"Средний предсказанный рейтинг: {average_rating:.2f}")
        # Вывод рейтингов для каждого имени
        for name, rating in zip(name_list, ratings):
            if rating is not None:
                print(f"Рейтинг для '{name.strip()}': {rating:.2f}")
            else:
                print(f"Рейтинг для '{name.strip()}': ошибка обработки")
    else:
        print("Ни одно из имен не было обработано.")

query_names()
