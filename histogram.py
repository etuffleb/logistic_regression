#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import describe


def histogram(houses, data, np_data, index):
    for h in houses:
        house = []
        for i, row in zip(range(len(data)), np_data):
            if data["Hogwarts House"][i] == h and not np.isnan(row[index]):
                house.append(row[index])
        plt.hist(house, alpha=0.5)



if __name__ == "__main__":
    try:
        data = pd.read_csv(r'dataset_train.csv')
    except IOError:
        print("Can not read file")
        exit(-1)

    numeric_data = data.select_dtypes('number')
    len_numeric = len(data.columns) - len(numeric_data.columns)

    all_metrics = describe.Describe()
    index_min_std = np.argmin(all_metrics.describe['std']) + len_numeric
    plt.title(data.columns[index_min_std])

    np_data = data.to_numpy()
    houses = data["Hogwarts House"].unique()

    histogram(houses, data, np_data, index_min_std)
    plt.show()
