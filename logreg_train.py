#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sys import argv
from argparse import ArgumentParser
import os


class LogRegTrainer():
    def __init__(self):
        self.parser = self._parser_trainer()
        self.data = self._validation()
        self._other_validation()
        self.X = self._data_selection(['Astronomy', 'Herbology', 'Divination',
                                       'Ancient Runes', 'Charms', 'Flying'])
        self.y = self._data_selection(['Hogwarts House'])['Hogwarts House']
        self.houses = self.y.unique()
        self.weights_list = []
        self._trainer()

    def _trainer(self):
        self.X = self._scale()
        self.X.fillna(self.X.mean(), inplace=True)
        self.X['free'] = 1
        for i in self.houses:
            self.weights_list.append(self.log_reg(i))
        self._to_csv()

    '''
    Стандартизация данных(каждый признак имеет mean=0 и disp=1)
    '''
    def _scale(self):
        scale_X = self.X.copy()
        for i in self.X.columns:
            std = self.X[i].std()
            mean = self.X[i].mean()
            for j in range(self.X.shape[0]):
                scale_X[i][j] = (self.X[i][j] - mean) / std
        return scale_X

    '''
    Расчитывание весов с помощью градиентного спуска, где:
    1 / (1 + np.exp(-y * np.sum(self.X * w, axis=1))) - сигмоидная функция
    (вероятность правильной классификации X[i])
    lr - градиентный шаг
    w[i] - нынешее значение веса
    '''

    def _gradient(self, w, y, C=5, lr=10**(-4)):
        w_new, evklid = [0] * len(w), 0
        sigmoid = self._sigmoid(y * np.sum(self.X * w, axis=1))
        for i, j in enumerate(self.X.columns):
            w_new[i] = w[i] + lr * np.mean(y * self.X[j] * (1 - sigmoid)) \
                - lr*C*w[i]
            evklid += (w_new[i] - w[i])**2
        return w_new, evklid

    def _sgd_function(self, w, y, row_number, C=5, lr=10**(-3.5)):
        w_new, evklid = [0] * len(w), 0
        sigmoid = self._sigmoid(y[row_number] *
                                np.sum(self.X.iloc[row_number] * w, axis=1))
        for i, j in enumerate(self.X.columns):
            w_new[i] = w[i] + lr * np.mean(y * self.X[j] * (1 - sigmoid)) \
                       - lr * C * w[i]
            evklid += (w_new[i] - w[i]) ** 2
        return w_new, evklid

    def _sigmoid(self, grade):
        return 1 / (1 + np.exp(-grade))

    '''
    В этом методе создаётся новый y, где все значения, равные house,
    заменяются на 1, остальные - 0.
    Также создаётся нудевой вектор весов, по одному для каждой фичи.
    Далее пока корень из значения изменения ошибки(evklid) > 10**(-5),
    веса пересчитываются.
    '''
    def log_reg(self, house):
        y_copy = np.where(self.y == house, 1, 0)
        w = [0]*self.X.shape[1]
        ln = self.X.shape[0]
        operations = 0
        evklid = 1
        if self.sgd:
            while evklid**0.5 > 10**(-5):
                row_number = np.random.randint(0, ln, size=20)
                w_new, evklid = self._sgd_function(w, y_copy, row_number)
                w = w_new
                operations += 1
            if self.operations:
                print(f'''Количество эпох стохастического градиентного спуска
для факультета \033[34;1m{house}\033[0m = \033[31m{operations}\033[0m''')
        else:
            while evklid**0.5 > 10**(-5):
                w_new, evklid = self._gradient(w, y_copy)
                w = w_new
                operations += 1
            if self.operations:
                print(f'''Количество эпох градиентного спуска
для факультета \033[34;1m{house}\033[0m = \033[31m{operations}\033[0m''')
        return w

    '''
    Запись весов для каждого факультета в csv.
    '''
    def _to_csv(self):
        csv = pd.DataFrame({'Houses': self.houses,
                            'Weights': self.weights_list})
        csv.to_csv('weights.csv', index=False)

    def _parser_trainer(self):
        parser = ArgumentParser(prog='logreg_train', description='''
            Эта программа создаёт модель, обученную на тестовых данных''',
                                add_help=True, epilog='''
            (c) April 2021. Авторы программы, как всегда, пусечки и лапочки''')
        parser.add_argument('--data', '-data', default='dataset_train.csv',
                            help='''Датасет для обучения модели''')
        parser.add_argument('--sgd', '-sgd', action='store_const', const=True,
                            help='Стохастический градиентный спуск')
        parser.add_argument('--operations', '-operations',
                            action='store_const', const=True,
                            help='Количество итераций градиентного спуска')
        return parser

    def _validation(self):
        name = self.parser.parse_args(argv[1:])
        self.data = name.data
        if not os.path.isfile(self.data):
            raise FileNotFoundError('Wrong path for file!')
        try:
            return pd.read_csv(self.data)
        except Exception as e:
            print(e)
            exit()

    def _other_validation(self):
        name = self.parser.parse_args(argv[1:])
        self.sgd, self.operations = name.sgd, name.operations

    def _data_selection(self, columns):
        columns_of_df = self.data.columns
        for i in columns:
            if i not in columns_of_df:
                raise ValueError(f'Столбца {i} нет в датафрейме!')
        return self.data[columns]


if __name__ == '__main__':
    LogRegTrainer()
