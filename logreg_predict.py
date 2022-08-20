#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logreg_train import LogRegTrainer
import pandas as pd
from argparse import ArgumentParser
import os


class LogRegPredictor(LogRegTrainer):
    def __init__(self):
        self.parser = self._parser_predictor()
        self.data = self._validation()
        self.X = self._data_selection(['Astronomy', 'Herbology', 'Divination',
                                       'Ancient Runes', 'Charms', 'Flying'])
        self.houses, self.weights_list = self._data_from_weights_csv()
        self.predict_weights = []
        self._predictor()

    def _predictor(self):
        self.X = self._scale()
        self.X.fillna(self.X.mean(), inplace=True)
        self.X['free'] = 1
        for i, j in self.X.iterrows():
            self.predict_weights.append(self.houses[self._result(j)])
        self._predict_to_csv()

    '''
    В этом методе высчитывается условная вероятность принадлежности ученика
    к каждому из факультетов.
    Факультет с наибольшей вероятностью считается верным
    '''
    def _result(self, student):
        ln = len(self.weights_list)
        houses = [0] * ln
        for i in range(ln):
            for j in range(len(self.weights_list[i])):
                houses[i] += student[j] * self.weights_list[i][j]
            houses[i] = self._sigmoid(houses[i])
        return houses.index(max(houses))

    '''
    Запись результата в csv
    '''
    def _predict_to_csv(self):
        csv = pd.DataFrame({'Index': [x for x in range(self.X.shape[0])],
                            'Hogwarts House': self.predict_weights})
        csv.to_csv('houses.csv', index=False)

    '''
    Проверка данных из файла weights.csv на корректность.
    Получение названий факультета и их весов.
    '''
    def _data_from_weights_csv(self):
        if not os.path.isfile('weights.csv'):
            raise FileNotFoundError('''File with weights is not exist.
            Please, run logreg_train.py''')
        try:
            weights_csv = pd.read_csv('weights.csv')
        except Exception as e:
            print(e)
            exit()
        if weights_csv.shape != (4, 2):
            raise ValueError('weights.csv is broken')
        houses = weights_csv.Houses.to_list()
        if {'Ravenclaw', 'Slytherin', 'Gryffindor',
                'Hufflepuff'} != set(houses):
            raise ValueError('weights.csv is broken')
        try:
            weights_data = weights_csv.Weights.tolist()
            weights_data = [i.split(', ') for i in weights_data]
            for i in weights_data:
                i[0] = i[0][1:]
                i[-1] = i[-1][:-1]
            for i, j in enumerate(weights_data):
                weights_data[i] = [float(x) for x in j]
                if len(weights_data[i]) != self.X.shape[1] + 1:
                    raise ValueError('weights.csv is broken')
            return houses, weights_data
        except Exception as e:
            print(e)
            exit()

    def _parser_predictor(self):
        parser = ArgumentParser(prog='logreg_predict', description='''
            Эта программа создаёт csv файл с предсказаниями
            для переданного набора данных''',
                                add_help=True, epilog='''
            (c) April 2021. Авторы программы, как всегда, пусечки и лапочки''')
        parser.add_argument('--data', '-data', default='dataset_test.csv',
                            help='''Датасет для предсказания факультета''')
        return parser


if __name__ == '__main__':
    LogRegPredictor()
