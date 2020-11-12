import pandas as pd
import math
import pylab as pl
import numpy as np
import random
from matplotlib.colors import ListedColormap


def generate_data(number_class, class_numbers) -> list:
    data = []
    for classNum in range(class_numbers):
        center_x, center_y = random.random() * 5.0, random.random() * 5.0
        for rowNum in range(number_class):
            data.append([[random.gauss(center_x, 0.5), random.gauss(center_y, 0.5)], classNum])
    return data


def show_data(n_classes: int, n_items_in_class: int) -> None:
    train_data = generate_data(n_items_in_class, n_classes)
    train_data_range = range(len(train_data))
    class_colormap = ListedColormap(['#FF0000', '#00FF00', '#2036f7'])
    pl.scatter([train_data[i][0][0] for i in train_data_range],
               [train_data[i][0][1] for i in train_data_range],
               c=[train_data[i][1] for i in train_data_range],
               cmap=class_colormap)
    pl.show()


def split_train_test(data, test_percent):
    train_data = []
    test_data = []
    for row in data:
        if random.random() < test_percent:
            test_data.append(row)
        else:
            train_data.append(row)
    return train_data, test_data


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def devide_knn(train_data, test_data, k, number_of_classes):
    test_labels = []
    for testPoint in test_data:
        test_dist = [[dist(testPoint, train_data[i][0]), train_data[i][1]] for i in range(len(train_data))]
        stat = [0 for i in range(number_of_classes)]
        for d in sorted(test_dist)[0:k]:
            stat[d[1]] += 1
        test_labels.append(sorted(zip(stat, range(number_of_classes)), reverse=True)[0][1])
    return test_labels


def calculate_accuracy(n_classes, k, test_percent):
    data = generate_data(n_item_in_class, n_classes)
    train_data, test_data_with_labels = split_train_test(data, test_percent)
    test_data = [test_data_with_labels[i][0] for i in range(len(test_data_with_labels))]
    test_data_labels = devide_knn(train_data, test_data, k, n_classes)
    print("Точность: ",
          sum([int(test_data_labels[i] == test_data_with_labels[i][1]) for i in
               range(len(test_data_with_labels))]) / float(
              len(test_data_with_labels)))


def generate_test_mesh(train_data):
    train_data_range = range(len(train_data))
    x_min = min([train_data[i][0][0] for i in train_data_range]) - 1.0
    x_max = max([train_data[i][0][0] for i in train_data_range]) + 1.0
    y_min = min([train_data[i][0][1] for i in train_data_range]) - 1.0
    y_max = max([train_data[i][0][1] for i in train_data_range]) + 1.0
    h = 0.05
    test_x, test_y = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
    return [test_x, test_y]


def show_data_on_mesh(n_classes, n_items_in_class, k):
    train_data = generate_data(n_items_in_class, n_classes)
    test_mesh = generate_test_mesh(train_data)
    test_mesh_labels = devide_knn(train_data, zip(test_mesh[0].ravel(), test_mesh[1].ravel()), k, n_classes)
    class_colormap = ListedColormap(['#FF0000', '#00FF00', '#2036f7'])
    test_colormap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAAA'])
    train_data_range = range(len(train_data))
    pl.pcolormesh(test_mesh[0],
                  test_mesh[1],
                  np.asarray(test_mesh_labels).reshape(test_mesh[0].shape),
                  cmap=test_colormap)
    pl.scatter([train_data[i][0][0] for i in train_data_range],
               [train_data[i][0][1] for i in train_data_range],
               c=[train_data[i][1] for i in train_data_range],
               cmap=class_colormap)
    pl.show()


if __name__ == '__main__':
    n_classes = 2
    n_item_in_class = 10
    testPercent = 0.023
    k = 15
    show_data(n_classes, n_item_in_class)
    show_data_on_mesh(n_classes, n_item_in_class, k)

    data = generate_data(n_item_in_class, n_classes)
    train_data, test_data_with_labels = split_train_test(data, testPercent)
    test_data = [test_data_with_labels[i][0] for i in range(len(test_data_with_labels))]
    calculate_accuracy(n_classes, k, testPercent)
