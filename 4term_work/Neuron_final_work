"""Код для курсовой работы по теме "Обнаружение аномалий в медицинских анализах"
Супрунова Е.В.; КВБО-03-23
Тема: обнаружение аномалий в медицинских анализах крови"""

# Раздел 1. Преподготовка
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

# Засекаем время выполнения кода
start_time = time.time()

# Для воспроизодимости результатов
np.random.seed(42)

# Раздел 2. Подготовка датасета
# Загрузка датасета
health_markers = pd.read_csv("\health_markers_dataset.csv",
                             encoding='utf-8')  # подключение датасета
print("Размер исходного датасета:", health_markers.shape)
print("Названия столбцов:")
print(health_markers.columns)
print("Первые пять строк датасета:")
print(health_markers.head())
print("\nОписание:")
print(health_markers.describe())

# Обработка датасета
health_markers = health_markers.query('Condition == "Fit" or Condition == "Diabetes"')
health_markers = health_markers.select_dtypes(
    include=[np.number]).dropna()  # удаление нечислового столбца "Condition" и строк, где есть пропущенные значения
print("Размер обработанного датасета:", health_markers.shape)
amount, _ = health_markers.shape
print(health_markers.describe())
columns_names = health_markers.columns
# В данном случае предположим, что меток нет (неконтролируемое обучение)

plt.figure(figsize=(10, 6))
plt.title("Исходные данные")
plt.scatter(health_markers.iloc[:, 0], health_markers.iloc[:, 1], alpha=0.6)
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.show()

# Генерация аномальных данных
# Создаём массив данных
data = {
    0: np.random.uniform(60, 210, size=int(amount * 0.05)),
    1: np.random.uniform(2, 12, size=int(amount * 0.05)),
}

# Заполняем остальные столбцы (2-8) нулями
for col in range(2, 9):
    data[col] = np.zeros(shape=int(amount * 0.05))

# Создаем DataFrame
df_anomaly = pd.DataFrame(data)

# Переименовываем столбцы
df_anomaly.columns = [i for i in columns_names]

X_all = pd.concat([health_markers, df_anomaly])

# Перемешиваем строки DataFrame случайным образом, чтобы убрать исходный порядок
# используем random_state для воспроизводимости, затем сбрасываем индексы
X_all = X_all.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.title("Данные с аномалиями")
plt.scatter(X_all.iloc[:, 0], X_all.iloc[:, 1], alpha=0.6)
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.show()

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X = X_scaled

# Раздел 3. Isolation forest
# Засекаем время выполнения скрипта
start_time_iso = time.time()

# Создаём объект модели Isolation Forest
# n_estimators=150 задаёт количество деревьев в ансамбле
# max_samples=256 определяет выборку данных
# contamination=0.0476 указывает ожидаемую долю аномалий (4,76% от общего числа пациентов (результатов анализов))
# max_features=0.8 использует 80% признаков для каждого разбиения
iso_forest = IsolationForest(n_estimators=200,  max_samples=256, contamination=0.0476, max_features=0.8)

# Обучаем модель на данных признаков X и получаем предсказания
# Метод fit_predict возвращает -1 для аномальных точек и 1 для нормальных
y_pred_iso = iso_forest.fit_predict(X)

# Вывод времени выполнения скрипта для оценки производительности
time_iso = time.time() - start_time_iso
print("\nВремя выполнения скрипта Isolation forest: {:.2f} сек".format(time_iso))

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_iso[:], cmap='coolwarm', alpha=0.6)
plt.title('Isolation Forest: обнаруженные аномалии')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()

# Раздел 4. LocalOutlierFactor (LOF)
# Засекаем время выполнения скрипта
start_time_lof = time.time()

# Инициализируем модель Local Outlier Factor:
# - n_neighbors=1 определяет, сколько соседей будет учитываться при оценке локальной плотности каждой точки
# - contamination=0.0476 задаёт ожидаемую долю аномальных точек в данных (4,76%)
lof = LocalOutlierFactor(n_neighbors=1, contamination=0.0476)

# Обучаем модель на данных признаков X и одновременно получаем предсказания
# Метод fit_predict возвращает массив, где аномальные точки обозначаются значением -1, а нормальные — 1
y_pred_lof = lof.fit_predict(X)

# Вывод времени выполнения скрипта для оценки производительности
time_lof = time.time() - start_time_lof
print("\nВремя выполнения скрипта LocalOutlierFactor (LOF): {:.2f} сек".format(time_lof))

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_lof, cmap='coolwarm', alpha=0.6)
plt.title('Local Outlier Factor: обнаруженные аномалии')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()

# Раздел 5. One-Class SVM
# Засекаем время выполнения скрипта
start_time_ocs = time.time()

# kernel="linear" определеяет тип ядра, используемого в алгоритме.
# Выбрано линейное, так как данные явно расположены в прямоугольниках
# nu=0.7 граница допустимых ошибок модели
clf_svm = OneClassSVM(kernel="linear", nu=0.7)
# Обучаем модель на данных признаков X и одновременно получаем предсказания
y_pred_ocs = clf_svm.fit_predict(X)

# Вывод времени выполнения скрипта для оценки производительности
time_ocs = time.time() - start_time_ocs
print("\nВремя выполнения скрипта One-Class SVM: {:.2f} сек".format(time_ocs))

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_ocs, cmap='coolwarm', alpha=0.6)
plt.title('One-Class SVM: обнаруженные аномалии')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()


# Раздел 6. Автоэнкодер
# Обучение автоэнкодера по данным c Isolation forest
X_marked = X_all.copy()
X_marked["Anomaly"] = y_pred_iso

# Масштабирование данных
scaler = StandardScaler()
X_marked_scaled = scaler.fit_transform(X_marked)
X_train = X_marked_scaled[X_marked["Anomaly"] == 1]
X_test = X_marked_scaled

# Засекаем время выполнения скрипта
start_time_autoencoder = time.time()

# Создание и обучение автоэнкодера
input_dim = X_train.shape[1]  # Размер входного вектора (количество признаков)
encoding_dim = 2 # Размер латентного представления
hidden_dim = 4      # Размер промежуточного слоя

# Определение архитектуры автоэнкодера
# Входной слой принимает вектор размерности input_dim
input_layer = Input(shape=(input_dim,))

# Промежуточный слой
encoded = Dense(hidden_dim, activation='relu')(input_layer)
# Слой кодировщика уменьшает размерность до encoding_dim с нелинейной активацией ReLU
encoded = Dense(encoding_dim, activation='relu')(input_layer)

decoded = Dense(hidden_dim, activation='relu')(encoded)
# Слой декодировщика восстанавливает исходное количество признаков, используя линейную активацию для регрессии
decoded = Dense(input_dim, activation='linear')(encoded)

# Формируем модель автоэнкодера, связывающую входной слой с декодированным выходом
autoencoder = Model(inputs=input_layer, outputs=decoded)
# Компилируем модель с оптимизатором Adam и функцией потерь MSE
autoencoder.compile(optimizer='adam', loss='mse')

# Обучение автоэнкодера
# Модель обучается реконструировать нормальные данные (X_train -> X_train)
# с использованием 30 эпох, размера батча 32, и валидационной выборки (10% от обучающей) для контроля переобучения
history = autoencoder.fit(X_train, X_train,
                          epochs=30,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=0) # Вывод логов, выключено

# Вывод времени выполнения скрипта для оценки производительности
time_autoencoder = time.time() - start_time_autoencoder
print("\nВремя выполнения скрипта Autoencoder: {:.2f} сек".format(time_autoencoder))

# Получение восстановленных данных
# Прогоняем весь тестовый набор через автоэнкодер, чтобы получить его реконструкцию
X_pred = autoencoder.predict(X_test)
# Вычисляем ошибку восстановления для каждой точки как среднеквадратичную ошибку между исходным и восстановленным входом
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

# Определение аномалий (верхние 4,76% ошибок)
threshold = np.percentile(mse, 95.24)

# Присваиваем метку -1 (аномалия) для точек, где ошибка восстановления выше порога, иначе 1
y_pred_ae = np.where(mse > threshold, -1, 1)

# график истории обучения
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.title('График обучения автоэнкодера')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_ae, cmap='coolwarm', alpha=0.6)
plt.title('Автоэнкодер: обнаруженные аномалии')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()
"""
# автоэнкодер
# Масштабирование данных
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(health_markers)

#input_dim = X_normal_scaled.shape[1]  # Размер входного вектора (количество признаков)

# Обучение автоэнкодера 
# Модель обучается реконструировать нормальные данные (X_normal_scaled -> X_normal_scaled)
# с использованием 30 эпох, размера батча 32, и валидационной выборки (10% от обучающей) для контроля переобучения
history = autoencoder.fit(X_normal_scaled, X_normal_scaled,
                          epochs=30,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=0) # Вывод логов, выключено
                          

# Получение восстановленных данных
# Прогоняем весь тестовый набор через автоэнкодер, чтобы получить его реконструкцию
X_pred = autoencoder.predict(X)
# Вычисляем ошибку восстановления для каждой точки как среднеквадратичную ошибку между исходным и восстановленным входом
mse = np.mean(np.power(X - X_pred, 2), axis=1)



# Обучение автоэнкодера по данным c Isolation forest
X_marked = X_all.copy()
X_marked["Anomaly"] = y_pred_iso

# Масштабирование данных
scaler = StandardScaler()
X_marked_scaled = scaler.fit_transform(X_marked)
X_train = X_marked_scaled[X_marked["Anomaly"] == 1]
X_test = X_marked_scaled

# Создание и обучение автоэнкодера
input_dim = X_train.shape[1]  # Размер входного вектора (количество признаков)
encoding_dim = 6 # Размер латентного представления

# Определение архитектуры автоэнкодера
# Входной слой принимает вектор размерности input_dim
input_layer = Input(shape=(input_dim,))

# Слой кодировщика уменьшает размерность до encoding_dim с нелинейной активацией ReLU
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Слой декодировщика восстанавливает исходное количество признаков, используя линейную активацию для регрессии
decoded = Dense(input_dim, activation='linear')(encoded)

# Формируем модель автоэнкодера, связывающую входной слой с декодированным выходом
autoencoder = Model(inputs=input_layer, outputs=decoded)
# Компилируем модель с оптимизатором Adam и функцией потерь MSE
autoencoder.compile(optimizer='adam', loss='mse')

# Модель обучается реконструировать нормальные данные (X -> X)
# с использованием 50 эпох, размера батча 32, и валидационной выборки (10% от обучающей) для контроля переобучения
history = autoencoder.fit(X_train, X_train,
                          epochs=50,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=0) # Вывод логов, выключено

# Вывод времени выполнения скрипта для оценки производительности
time_autoencoder = time.time() - start_time_autoencoder
print("\nВремя выполнения скрипта Autoencoder: {:.2f} сек".format(time_autoencoder))

# Получение восстановленных данных
# Прогоняем весь тестовый набор через автоэнкодер, чтобы получить его реконструкцию
X_pred = autoencoder.predict(X_test)
# Вычисляем ошибку восстановления для каждой точки как среднеквадратичную ошибку между исходным и восстановленным входом
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

# Определение аномалий (верхние 5% ошибок)
threshold = np.percentile(mse, 95)

# Присваиваем метку -1 (аномалия) для точек, где ошибка восстановления выше порога, иначе 1
y_pred_ae_on_iso = np.where(mse > threshold, -1, 1)

# график истории обучения
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.title('График обучения автоэнкодера')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_ae_on_iso, cmap='coolwarm', alpha=0.6)
plt.title('Автоэнкодер: обнаруженные аномалии')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()"""

# Раздел 7. Голосование
# Создание DataFrame с результатами всех методов
results_df = pd.DataFrame({
    'Isolation_Forest': y_pred_iso,
    'OneClassSVM': y_pred_ocs,
    'Autoencoder': y_pred_ae,
})

# Голосование: если минимум 2 метода считают точку аномальной
results_df['Final_Anomaly'] = results_df.sum(axis=1).apply(lambda x: -1 if x <= -2 else 1)

# Визуализация итоговых результатов
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=results_df['Final_Anomaly'], cmap='coolwarm', alpha=0.6)
plt.title('Итоговые обнаруженные аномалии (голосование)')
plt.xlabel(columns_names[0])
plt.ylabel(columns_names[1])
plt.colorbar(label='Аномалия (-1) / Норма (1)')
plt.show()

print(results_df)

print("\nВремя выполнения программы: {:.2f} сек".format(time.time() - start_time))
"""
accuracy = accuracy_score(y_pred_ae_on_iso, results_df['Final_Anomaly'])
print(f"Согласованность между моделями: {accuracy:.4f}")

# Создание DataFrame с результатами моделей
results_df_1 = pd.DataFrame({
    'Suggestion': results_df['Final_Anomaly'],
    'Autoencoder_on_iso': y_pred_ae_on_iso,
})
results_df_1['Final_Anomaly'] = results_df_1.sum(axis=1).apply(
    lambda x: 'Аномалия' if x == -2 else ('Вероятно аномалия' if x == 0 else 'Верный результат'))
print(results_df_1.shape)
print(results_df_1)"""
