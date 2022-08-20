#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sys import argv


def parser():
    pars = argparse.ArgumentParser(add_help=True, conflict_handler='resolve',
                                   description=''' 
    Эта программа выводит сравнительный график по двум курсам:
    Arithmancy
    Astronomy
    Herbology
    Defense Against the Dark Arts
    Muggle Studies
    Ancient Runes
    History of Magic
    Transfiguration
    Potions
    Care of Magical Creatures
    Charms
    Flying''',

                                 epilog='''
            (c) April 2021. Авторы программы, как всегда, пусечки и лапочки''')
    pars.add_argument('first', type=str, help='Первый курс для сравнения')
    pars.add_argument('second', type=str, help='Второй курс для сравнения')
    return pars


def SwitchCase(args):
    courses = {'Arithmancy': 1,
               'Astronomy': 2,
               'Herbology': 3,
               'Defense Against the Dark Arts': 4,
               'Muggle Studies': 5,
               'Ancient Runes': 6,
               'History of Magic': 7,
               'Transfiguration': 8,
               'Potions': 9,
               'Care of Magical Creatures': 10,
               'Charms': 11,
               'Flying': 12}
    if args not in courses.keys():
        raise NameError('Неверное имя курса. Все курсы перечислены в help')
    return courses[args]


def scatter_plot(houses, data, np_data, index1, index2):
    for h in houses:
        x = []
        y = []
        for i, row in zip(range(len(data)), np_data):
            if data["Hogwarts House"][i] == h and not np.isnan(row[index1]) and not np.isnan(row[index2]):
                x.append(row[index1])
                y.append(row[index2])
        plt.scatter(x, y, alpha=0.7, s=9)


if __name__ == "__main__":
    pars = parser()
    name = pars.parse_args(argv[1:])
    try:
        data = pd.read_csv(r'dataset_train.csv')
    except IOError:
        print('Can not read file')
        exit(0)
    numeric_data = data.select_dtypes('number')
    len_numeric = len(data.columns) - len(numeric_data.columns)
    arg_1 = SwitchCase(name.first) + len_numeric
    arg_2 = SwitchCase(name.second) + len_numeric
    np_data = data.to_numpy()
    plt.xlabel(data.columns[arg_1])
    plt.ylabel(data.columns[arg_2])
    houses = data["Hogwarts House"].unique()
    scatter_plot(houses, data, np_data, arg_1, arg_2)
    plt.show()
