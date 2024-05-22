import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import imageio
import os

# Загрузка датасета ирисов
iris = load_iris()
X = iris.data

# Функция для вычисления расстояния
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Инициализация центроидов
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    centroids = X[indices[:k]]
    return centroids

# Определение ближайших центроидов
def closest_centroid(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return np.argmin(distances, axis=1)

# Пересчет центроидов
def compute_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)
    return centroids

# Функция для визуализации кластеров
def plot_clusters(X, labels, centroids, iteration, image_folder):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(np.unique(labels))):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title(f'Iteration {iteration + 1}')
    plt.legend()
    image_path = os.path.join(image_folder, f'kmeans_step_{iteration + 1}.png')
    plt.savefig(image_path)
    plt.close()

# K-means алгоритм
def kmeans(X, k, max_iters=100, plot_steps=False, image_folder=None):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = closest_centroid(X, centroids)
        if plot_steps and image_folder:
            plot_clusters(X, labels, centroids, i, image_folder)
        new_centroids = compute_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Сохранение изображений в GIF
def create_gif(image_folder, gif_name):
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=0.5)

# Создание папки для хранения изображений
image_folder = 'kmeans_images'
os.makedirs(image_folder, exist_ok=True)

# Определение оптимального количества кластеров (например, из шага 1)
k = 3  # оптимальное количество кластеров

# Запуск алгоритма и сохранение шагов
centroids, labels = kmeans(X, k, plot_steps=True, image_folder=image_folder)

# Создание GIF
create_gif(image_folder, 'kmeans_animation.gif')