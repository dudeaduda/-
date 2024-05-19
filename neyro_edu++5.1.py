import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt

# Очистка среды TensorFlow для предотвращения конфликтов
#tf.keras.backend.clear_session()            данная строка неактивна, использовалась при изменении структуры модели

# Чтение и загрузка данных из текстового файла
input_file_path = r"D:\курсач инф\для токенизации и векторизации\cleaned_data.txt"
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

names = []
ratings = []

# Токенизация и обработка строк
# Разделяем каждую строку на имя и рейтинг, добавляем их в соответствующие списки
for line in lines:
    parts = line.strip().split(': ')
    if len(parts) == 2:
        names.append(parts[0])
        ratings.append(float(parts[1]))

# Посимвольная векторизация имен
# Создание слоя TextVectorization для преобразования текста в числовые векторы
char_vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=500,  # Максимальное количество уникальных символов (токенов)
    output_mode='int',  # Выходной формат - целые числа
    output_sequence_length=30,  # Фиксированная длина выходной последовательности
    split='character'  # Векторизация на уровне символов
)
# Обучение слоя TextVectorization на списке имен
char_vectorize_layer.adapt(names)

# Преобразование всех имен в векторы
char_vectorized_names = char_vectorize_layer(np.array(names))

# Нормализация рейтингов
# Преобразование рейтингов в массив NumPy и нормализация их в диапазоне [0, 1]
ratings = np.array(ratings)
normalized_ratings = (ratings - np.min(ratings)) / (np.max(ratings) - np.min(ratings))

# Проверка векторизации на примере первых 5 имен
# Преобразование нескольких имен для проверки работы векторизации
sample_names = ['Melda Yazgi', 'Zeynep Satir', 'Tahsin Taskin', 'Satilmis Yildirim', 'Recep Yenice']
sample_vectorized_names = char_vectorize_layer(np.array(sample_names))
print("Sample names:", sample_names)
print("Vectorized names:", sample_vectorized_names.numpy())

# Подготовка данных для обучения
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(char_vectorized_names.numpy(), normalized_ratings, test_size=0.2, random_state=42)

# Функция создания модели для Keras Tuner
# Определение архитектуры модели с помощью гиперпараметров
def build_model(hp):
    model = Sequential([
        Input(shape=(30,)),  # Входной слой с фиксированной длиной последовательности
        Embedding(input_dim=500, output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=16)),  # Слой Embedding
        Flatten(),  # Преобразование выходных данных Embedding в плоский вид
        Dense(units=hp.Int('units1', min_value=64, max_value=512, step=32), activation='relu'),  # Полносвязный слой
        Dropout(0.5),  # Слой Dropout для предотвращения переобучения
        Dense(units=hp.Int('units2', min_value=32, max_value=256, step=32), activation='relu'),  # Полносвязный слой
        Dropout(0.5),  # Слой Dropout для предотвращения переобучения
        Dense(1, activation='linear')  # Выходной слой с линейной активацией для регрессии
    ])
    # Компиляция модели с использованием Adam оптимизатора и функции потерь MSE
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model

# Настройка и запуск поиска гиперпараметров
# Создание объекта RandomSearch для подбора гиперпараметров модели
tuner = RandomSearch(
    build_model,  # Функция создания модели
    objective='val_loss',  # Целевая метрика для оптимизации
    max_trials=10,  # Максимальное количество попыток
    executions_per_trial=1,  # Количество запусков для каждой комбинации гиперпараметров
    directory='/mnt/data',  # Каталог для сохранения результатов
    project_name='name_rating'  # Название проекта
)

# Запуск поиска гиперпараметров
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Получение лучшей модели и гиперпараметров
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]
print('Лучшие гиперпараметры:', best_hyperparameters.values)

# Добавим раннюю остановку
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Обучение и сохранение лучшей модели
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
best_model.save('/mnt/data/best_trained_model1.keras')

# Отображение графиков потерь
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
