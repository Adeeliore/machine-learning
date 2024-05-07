import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
data = pd.read_csv('bikes_rent.csv')

# Пункт 2: Простая линейная регрессия
X = data[['weathersit']]
y = data['cnt']

regression = LinearRegression()
regression.fit(X, y)

plt.scatter(X, y, color='green')
plt.plot(X, regression.predict(X), color='blue')
plt.title('Прогноз спроса на основе благоприятности погоды')
plt.xlabel('Благоприятность погоды')
plt.ylabel('Спрос')
plt.show()

# Пункт 3: Предсказание значения cnt
input_value = np.array([[2]])
predicted_value = regression.predict(input_value)
print(f'Предсказанное количество аренд: {predicted_value[0]}')

# Пункт 4: Уменьшение размерности и построение 2D графика
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(data.drop('cnt', axis=1))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.title('2D график предсказания cnt')
plt.show()

# Пункт 5: Регуляризация Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(data.drop('cnt', axis=1), y)

# Определение признака, который оказывает наибольшее влияние на cnt
coefficients = pd.Series(lasso_reg.coef_, index = data.drop('cnt', axis=1).columns)
print(f'Признак, оказывающий наибольшее влияние на cnt: {coefficients.idxmax()}')
