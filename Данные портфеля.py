import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
import matplotlib.dates as mdates
import plotly.graph_objects as go
from statistics import median
import pickle
import pandas_datareader as pdr
import datetime
from scipy.signal import correlate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from scipy import stats

stocks = ['LKOH', 'ROSN', 'NVTK', 'SNGS', 'SNGSP', 'TRNFP', 'TATN', 'TATNP',
          'NLMK', 'CHMF', 'MTLR', 'MTLRP', 'MAGN', 'GMKN',
          'IRAO', 'HYDR', 'FEES', 'MRKC', 'UPRO', 'SBER', 'SBERP', 'MOEX', 'VTBR', 'MGNT', 'RTKM', 'MTSS']
dates = []
full_info = {}
aton = {}
for i in stocks:
    filename = str(i) + '.csv'

    table_to_explore = pd.read_csv('stocks/' + filename)
    table_to_explore.set_index('dates', inplace=True)

    full_info[i] = table_to_explore

    for date in table_to_explore.index:
        if date not in aton:
            aton[date] = pd.DataFrame()

for stock, info in full_info.items():
    for date in info.index:
        if date not in aton:
            aton[date] = pd.DataFrame(index=[stock], columns=[['result']])
        try:
            aton[date].loc[stock, 'result'] = info.loc[date, 'result']

        except KeyError:
            pass

del aton['2022-09-29']  # Удаление элемента по ключу '2022-09-29'
del aton['2020-02-06']

for k, v in aton.items():
    v['position'] = 0
    prev_table = v
    break
aton
for k, v in aton.items():
    v['position'] = 0

stocks = []

for date, table in aton.items():

    stocks = []
    for index, row in prev_table.iterrows():
        stocks.append(index)

    aton[date]['position'] = prev_table['position']

    current_stocks = []

    for index, row in aton[date].iterrows():
        current_stocks.append(index)

    if len(current_stocks) < len(stocks):
        for i in range(len(stocks)):
            if stocks[i] not in current_stocks:
                aton[date].loc[stocks[i], 'position'] = prev_table.loc[stocks[i], 'position']
                aton[date].loc[stocks[i], 'result'] = prev_table.loc[stocks[i], 'result']

    for index, row in table.iterrows():
        print(date)
        if row['result'] < 0:
            aton[date].loc[index, 'position'] = 0
        if row['result'] > 2:
            aton[date].loc[index, 'position'] = row['result'] / aton[date][(aton[date]['result'] > 2)]['result'].sum()

    sum = table['position'].sum()

    for index, row in table.iterrows():
        if sum != 0:
            aton[date].loc[index, 'position'] = aton[date].loc[index, 'position'] / sum

    prev_table = table

p = 0
keys_to_delete = []
for k, v in aton.items():
    if p != 5:
        keys_to_delete.append(k)
        p += 1
    else:
        p = 0

for k in keys_to_delete:
    del aton[k]

neft_index=34.75/(15.14+10.02+34.75+10.84)
met_index=10.84/(15.14+10.02+34.75+10.84)
fin_index=15.14/(15.14+10.02+34.75+10.84)
ros_index=10.02/(15.14+10.02+34.75+10.84)

for k, v in aton.items():
    sum_lkoh=0
    sum_met=0
    sum_fin=0
    sum_ros=0
    for index, row in v.iterrows():
        if index in ['LKOH', 'ROSN', 'NVTK', 'SNGS', 'SNGSP', 'TRNFP', 'TATN', 'TATNP']:
            sum_lkoh+=row['position']
        elif index in ['NLMK', 'CHMF', 'MTLR', 'MTLRP', 'MAGN', 'GMKN']:
            sum_met+=row['position']
        elif index in [ 'SBER', 'SBERP', 'MOEX', 'VTBR']:
            sum_fin+=row['position']
        else:
            sum_ros+=row['position']

    for index, row in v.iterrows():
        if index in  ['LKOH', 'ROSN', 'NVTK', 'SNGS', 'SNGSP', 'TRNFP', 'TATN', 'TATNP']:
            if sum_lkoh!=0:
                aton[k].loc[index, 'position']*=neft_index/sum_lkoh
        elif index in  ['NLMK', 'CHMF', 'MTLR', 'MTLRP', 'MAGN', 'GMKN']:
            if sum_met!=0:
                aton[k].loc[index, 'position']*=met_index/sum_met
        elif index in  [ 'SBER', 'SBERP', 'MOEX', 'VTBR']:
            if sum_fin!=0:
                aton[k].loc[index, 'position']*=fin_index/sum_fin
        else:
            if sum_ros!=0:
                aton[k].loc[index, 'position']*=ros_index/sum_ros



# Сохранение объекта aton в формате Pickle
with open('aton.pkl', 'wb') as f:
    pickle.dump(aton, f)