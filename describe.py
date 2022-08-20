#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from tabulate import tabulate
from sys import argv
from argparse import ArgumentParser
import os


class Describe():
    def __init__(self):
        self.parser = self._parser_describer()
        self.data = self._validation()
        self.describe = {'count': [0] * 14, 'null': [0] * 14, 'mean': [0] * 14,
                         'std': [0] * 14, 'min': [10**5] * 14, 'max': [0] * 14,
                         '25%': [0] * 14, '50%': [0] * 14, '75%': [0] * 14,
                         'uniq': [0] * 14, 'Disp': [0] * 14, 'sum': [0] * 14,
                         'range': [0] * 14}
        self._create_describe()

    def _create_describe(self):
        pd.options.display.float_format = '{:.6f}'.format
        self._data_selection()
        self.data = self.data.fillna('Null')
        for j, k in enumerate(self.data):
            for_quartils = self._count_sum_etc(self.data[k], j)
            for_quartils = self._quicksort(for_quartils)
            self._quartils(for_quartils, j)
            self._disp_std(self.data[k], j)


    def print_info(self):
        describe_frame = pd.DataFrame(self.describe)
        describe_frame = describe_frame.transpose()
        describe_frame.columns = self.data.columns
        for i in range(0, len(describe_frame.columns), 7):
            print(tabulate(describe_frame[describe_frame.columns[i:i + 7]],
                  headers='keys', tablefmt='fancy_grid', numalign='left'),
                  end='\n\n')

    '''
    Выбираем только те столбцы, где нданные - это числа
    '''
    def _data_selection(self):
        columns = []
        for i in self.data.columns:
            if self.data[i].dtype in ('float64', 'int64'):
                columns.append(i)
        self.data = self.data[columns]

    def _binary_search_recursive(self, arr, elem, start=0):
        mid = len(arr) // 2
        if arr[mid - 1] <= elem and arr[mid] >= elem:
            return start + mid
        if arr[mid - 1] >= elem and arr[mid - 2] <= elem:
            return start + mid - 1
        if elem < arr[mid - 1]:
            return self._binary_search_recursive(arr[0:mid], elem,
                                                 start=start)
        return self._binary_search_recursive(arr[mid:], elem, start+mid)

    '''
    В этом методе расчитываются показатели:
    count - количество заполненных позиций
    null - количество пустых ячеек
    sum - сумма всех значений в столбце
    min - минимальное значение
    max - максимальное значение
    uniq - количество неповторяющися значений
    mean - среднее значение по столбцу
    range - размах
    '''
    def _count_sum_etc(self, data, j):
        for i in data:
            if i != 'Null':
                self.describe['count'][j] += 1
                self.describe['sum'][j] += i
                if i < self.describe['min'][j]:
                    self.describe['min'][j] = i
                if i > self.describe['max'][j]:
                    self.describe['max'][j] = i
            else:
                self.describe['null'][j] += 1
        self.describe['uniq'][j] = len(set(data))
        self.describe['mean'][j] = self.describe['sum'][j] \
            / self.describe['count'][j]
        self.describe['range'][j] = self.describe['max'][j] \
            - self.describe['min'][j]
        for_quartils = [l for l in data if l != 'Null']
        return for_quartils

    '''
    Метод быстрой сортировки
    '''
    def _quicksort(self, data):
        if len(data) <= 1:
            return data
        else:
            num = data[len(data) // 2]
            more, less, equal = [], [], []
            for k in data:
                if k > num:
                    more.append(k)
                elif k < num:
                    less.append(k)
                else:
                    equal.append(k)
            return self._quicksort(less) + equal + self._quicksort(more)

    '''
    Подсчёт значений квартилей
    (числа, которые делят упорядоченные значения столбца на 4 равные группы)
    '''
    def _quartils(self, data, j):
        elem = len(data)
        half = int(elem // 2)
        if elem % 2 == 0:
            self.describe['50%'][j] = (data[half] + data[half - 1]) / 2
            median = half - 0.5
        else:
            self.describe['50%'][j] = data[half]
            median = half
        qw_1 = median / 2
        qw_3 = median + qw_1
        if qw_1 is int:
            self.describe['75%'][j] = data[qw_3]
            self.describe['25%'][j] = data[qw_1]
        else:
            int_qw_3 = int(qw_3)
            int_qw_1 = int(qw_1)
            self.describe['25%'][j] = data[int_qw_1] + (qw_1 % int_qw_1) \
                * (data[int_qw_1 + 1] - data[int_qw_1])
            self.describe['75%'][j] = data[int_qw_3] + (qw_3 % int_qw_3) \
                * (data[int_qw_3 + 1] - data[int_qw_3])

    '''
    Расчёт среднеквадратичного отклонения и дисперсии -
    показатели рассеивания/разброса
    '''
    def _disp_std(self, data, j):
        mean = self.describe['mean'][j]
        for i in data:
            if i != 'Null':
                self.describe['Disp'][j] += (i - mean) ** 2
        self.describe['Disp'][j] /= (self.describe['count'][j] - 1)
        self.describe['std'][j] = self.describe['Disp'][j] ** 0.5

    def _parser_describer(self):
        parser = ArgumentParser(prog='describe', description='''
            Эта программа выдаёт данные для анализа содержимого фрейма''',
                                add_help=True, epilog='''
            (c) April 2021. Авторы программы, как всегда, пусечки и лапочки''')
        parser.add_argument('--data', '-data', default='dataset_train.csv',
                            help='''Датасет для анализа''')
        return parser

    def _validation(self):
        self.data = self.parser.parse_args(argv[1:]).data
        if not os.path.isfile(self.data):
            raise FileNotFoundError('Wrong path for file!')
        try:
            return pd.read_csv(self.data)
        except Exception as e:
            print(e)
            exit()


if __name__ == '__main__':
    d = Describe()
    d.print_info()
