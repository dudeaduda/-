import numpy as np
import joblib

# Загрузка словаря с именами и рейтингами
name_rating_dict = joblib.load('/mnt/data/name_rating_dict.joblib')

def get_name_rating(name_input):
    # Поиск рейтинга для заданного имени
    return name_rating_dict.get(name_input.strip(), None)

def query_names():
    names = input("Введите имена через запятую для получения среднего рейтинга (или 'exit' для выхода): ")
    if names.lower() == 'exit':
        return
    name_list = names.split(',')
    ratings = [get_name_rating(name.strip()) for name in name_list]
    valid_ratings = [rating for rating in ratings if rating is not None]
    if valid_ratings:
        average_rating = np.mean(valid_ratings)
        print(f"Средний рейтинг: {average_rating:.2f}")
        for name, rating in zip(name_list, ratings):
            if rating is not None:
                print(f"Рейтинг для '{name.strip()}': {rating:.2f}")
            else:
                print(f"Рейтинг для '{name.strip()}': не найден")
    else:
        print("Ни одно из имен не было найдено.")

query_names()
