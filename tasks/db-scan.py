# coding=utf-8
import math
from sklearn.datasets.samples_generator import make_blobs, make_moons
from matplotlib import pyplot as plt
import seaborn as sns


def euclidian_distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += (point2[i] - point1[i]) ** 2
    return math.sqrt(sum)


def range_query(collection, point, distFunc, eps):
    neighbors = []
    for i in range(len(collection)):
        if distFunc(collection[i], point) < eps:
            neighbors.append(i)

    return neighbors


def custom_dbscan(X, eps, minPts, dist_func='euclidean'):
    if dist_func == "euclidean":
        dist_func = euclidian_distance

    lenght_x = len(X)

    labels = [None] * lenght_x  # начать со всех меток undefined
    cluster = 0  # cluster counter

    for i in range(lenght_x):
        if labels[i] is not None:
            continue

        neighbors = range_query(X, X[i], dist_func, eps)

        if len(neighbors) < minPts:
            labels[i] = -1
            continue

        labels[i] = cluster

        j = 0
        while j < len(neighbors):
            p = neighbors[j]
            if labels[p] == -1:
                labels[p] = cluster

            if labels[p] is not None:
                j += 1
                continue

            labels[p] = cluster
            new_neighbors = range_query(X, X[p], dist_func, eps)
            if len(new_neighbors) >= minPts:
                for n in new_neighbors:
                    if n not in neighbors:
                        neighbors.append(n)

            j += 1

        cluster += 1

    return labels


if __name__ == '__main__':
    blobs_X, blobs_y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=2, random_state=42)
    sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=blobs_y)
    clusters = custom_dbscan(blobs_X, eps=2, minPts=10)
    print(clusters)
    sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=clusters)
    plt.title("My DBSCAN Blobs")
    plt.show()
