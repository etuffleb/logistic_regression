#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


if __name__ == "__main__":
    try:
        data = pd.read_csv(r'dataset_train.csv')
    except IOError:
        print("Can not read file")
        exit(-1)
    numeric_data = data.select_dtypes('number')
    len_numeric = len(data.columns) - len(numeric_data.columns) + 1
    data = data.replace({"Ravenclaw": 0, "Hufflepuff": 1, "Slytherin": 2, "Gryffindor": 3})
    data = data.select_dtypes('number')
    crop_data = data[len_numeric:]
    crop_data = crop_data.drop("Index", axis = 1)
    crop_data = crop_data.replace({0: "Ravenclaw", 1: "Hufflepuff", 2: "Slytherin", 3: "Gryffindor"})
    sb.pairplot(crop_data, hue='Hogwarts House', diag_kind="hist", plot_kws={"s": 3})
    plt.show()
