from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import math


def count_age(test_dataset, survived_dataset):
    """Return count of passangers. Key is an age of person."""
    return dict(Counter([round(value) for value in test_dataset.Age if not math.isnan(value)]))


def show_count_age():
    test_dataset = pd.read_csv("/home/emil/Desktop/MachineLearningLessons/titanic_dataset/test.csv")
    survived_dataset = pd.read_csv("/home/emil/Desktop/MachineLearningLessons/titanic_dataset/gender_submission.csv")

    age_count = count_age(test_dataset, survived_dataset)

    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.scatter(age_count.keys(), age_count.values(), c='blue')
    plt.show()


if __name__ == '__main__':
    show_count_age()
