import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
import math


def gradient_descent(x, y, z):
    cur_A = 3  # The algorithm starts at A=3
    cur_B = 3  # The algorithm starts at B=3
    cur_C = 3  # The algorithm starts at C=3
    cur_D = 3  # The algorithm starts at D=3
    rate = 0.01  # Learning rate
    precision = 0.000001  # This tells us when to stop the algorithm
    previous_step_size = 1  #
    max_iters = 10000  # maximum number of iterations
    iters = 0  # iteration counter
    df = lambda A, B, C, D: A * x + B * y + C * z + D  # Вот наша функция, в которой мы будем искать минимальные A,B,C,D

    while previous_step_size > precision and iters < max_iters:
        prev_A = cur_A
        prev_B = cur_B
        prev_C = cur_C
        prev_D = cur_D
        cur_A = cur_A - rate * df(prev_A, prev_B, prev_C, prev_D)
        cur_B = cur_B - rate * df(prev_B, prev_B, prev_C, prev_D)
        cur_C = cur_C - rate * df(prev_B, prev_B, prev_C, prev_D)
        previous_step_size = abs(cur_C - prev_A)
        iters = iters + 1  # iteration count
    print("The local minimum occurs at", cur_A, cur_B, cur_C, cur_D)
    return cur_A * x + cur_B * y + cur_C * z + cur_D


if __name__ == '__main__':
    # colors = ("#FF0000", "#0000FF")
    # рандомно задаем точки
    n = 100  # количество точек
    # координаты точек для 3d
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    z = np.random.randint(0, 100, n)
    # должны каким-то образом разделить точки на классы
    # в частности применить k-means
    points = []
    for i in range(n):
        points.append([x[i], y[i], z[i]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    clusters = kmeans.labels_
    # print(clusters)
    colors = ['red'] * n
    for i in range(n):
        if clusters[i] == 1:
            colors[i] = 'blue'
        # рисуем их при помощи plotly
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                       marker=dict(color=colors))])
    fig.show()
    # минимизируем сумму логарифмов и вытаскиваем параметры плоскости
    # тут нарочно неправильно
    # нужно: найти параметры A,B,C,D минимизацией того p что написан ниже
    # можно использовать любые готовые библиотеки
    # например что-то связанное с градиентным спуском
    p = 0

    alpha, hidden_dim = (0.5, 4)
    synapse_0 = 2 * np.random.random((3, hidden_dim)) - 1
    synapse_1 = 2 * np.random.random((hidden_dim, 1)) - 1
    for i in range(n):
        p += math.log(1 + math.e ** gradient_descent(x[i], y[i], z[i]))

    # отсроить разделяющую плоскость (картинка) + точки отрисованные выше
    # предиктить новые точки (тоже на вход рандомно)
    x_new = np.random.randint(0, 100)
    y_new = np.random.randint(0, 100)
    z_new = np.random.randint(0, 100)
    fig = go.Figure(data=[go.Scatter3d(x=[x_new], y=[y_new], z=[z_new],
                                       mode='markers')])
    fig.show()
