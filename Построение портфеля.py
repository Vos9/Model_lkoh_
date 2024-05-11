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

# Укажите путь к файлу base_dict.p
file_path = 'data_set.p'

# Чтение данных из файла
with open('data_set.p', 'rb') as file:
    info = pickle.load(file)


# Загрузка объекта aton из файла Pickle
with open('aton.pkl', 'rb') as f:
    info_df4 = pickle.load(f)

for k, v in info_df4.items():
    info_df4[k]['price']=0
    info_df4[k]['type']='long'
    for index, row in v.iterrows():
        try:
            info_df4[k].loc[index, 'price']=info[index].loc[k, 'CLOSE']
        except KeyError :
            pass

keys_to_remove = []

for k, v in info_df4.items():
    for index, row in v.iterrows():
        if row['price'] == 0:
            keys_to_remove.append(k)
            break

for key in keys_to_remove:
    del info_df4[key]

info_df3=pd.DataFrame()
info_df3['type']='long'
info_df3['price']=0
info_df3['position']=0
info_df3['date']=0

for k, v in info_df4.items():
    for index, row in v.iterrows():
        row_to_append={'type':'long','price': row['price'],  '<TICKER>': index, 'position': row['position'], 'date': k }
        info_df3 = info_df3.append(row_to_append, ignore_index=True)

for index, row in info_df3.iterrows():
    if np.isnan(row['position']):
        info_df3.loc[index, 'position']=0
info_df3['date'] = pd.to_datetime(info_df3['date'])
dates=[]
for index, row in info_df3.iterrows():
    dates.append(row['date'])

# создание базы данных с позициями (пустую)
aton = {}
# info_df3
I = 1000000
full_info = pd.DataFrame()
# dates=[]
# for index, row in info_df3.iterrows():
# dates.append(row['date'])

I = 1000000
stocks = []
for index, row in info_df3.iterrows():
    stocks.append(row['<TICKER>'])
unique_stocks_set = set(stocks)

# Преобразуем множество обратно в список, если требуется
stocks = list(unique_stocks_set)

# Создаем пустой DataFrame для сохранения данных
full_info = pd.DataFrame(
    columns=['<TICKER>', 'position', 'money', 'price_first', 'price_now', 'pnl', 'dadao', 'moneyo', 'type', 'posa'])

# Заполняем его данными из DataFrame info_df3 без строк, содержащих NaN
for index, row in info_df3.iterrows():
    if pd.notna(row['<TICKER>']):
        ticker = row['<TICKER>']
        info_to_add = {'<TICKER>': ticker, 'position': 0, 'money': 0, 'price_first': 0, 'price_now': 0, 'pnl': 0,
                       'dadao': 0, 'moneyo': 0, 'type': 0, 'posa': 0}
        full_info = full_info.append(info_to_add, ignore_index=True)

# Устанавливаем столбец '<TICKER>' в качестве индекса
full_info = full_info.drop_duplicates(subset='<TICKER>')
info_to_add = {'<TICKER>': 'cash', 'position': 0, 'money': 1000000, 'price_first': 0, 'price_now': 0, 'pnl': 0,
               'dadao': 0, 'moneyo': 0, 'type': 0, 'posa': 0}
full_info = full_info.append(info_to_add, ignore_index=True)
full_info.set_index('<TICKER>', inplace=True)
for i in dates:
    aton[i] = full_info

prev_table = None
moneyo = 0
prev_price = 0
stock_to_sell = ''
need_money = 0
positions_to_close = 0
I = 1000000
flag = False
for date, table in aton.items():
    table_copy = table.copy()  # Создаем копию DataFrame table

    for index, row in info_df3.iterrows():

        if pd.to_datetime(row['date']) == date:

            # проверяем хватает ли денег
            if row['position'] > 0:

                # отркываем новую позицию

                if prev_table is not None:

                    prev_price = prev_table.at[row['<TICKER>'], 'price_now']
                    table_copy = prev_table.copy()
                    table_copy.at[row['<TICKER>'], 'price_now'] = row['price']
                    if prev_price != 0:

                        table_copy.at[row['<TICKER>'], 'pnl'] = ((row['price'] - prev_price) / prev_price) * \
                                                                table_copy.at[row['<TICKER>'], 'position']

                    else:
                        table_copy.at[row['<TICKER>'], 'pnl'] = 0
                    table_copy.at[row['<TICKER>'], 'position'] = 0
                    table_copy.at[row['<TICKER>'], 'money'] = table_copy.at[row['<TICKER>'], 'moneyo'] + table_copy.at[
                        row['<TICKER>'], 'pnl']

                    table_copy.at['cash', 'money'] += table_copy.at[row['<TICKER>'], 'money']

                    table_copy.at[row['<TICKER>'], 'money'] = 0
                    table_copy.at[row['<TICKER>'], 'pnl'] = 0
                    table_copy.at[row['<TICKER>'], 'position'] = 0

                    prev_table = table_copy.copy()


            # если в таблице сделок дано закрытие позиции, то соответственно закрываем
            elif row['position'] == 0 and prev_table is not None:

                prev_price = prev_table.at[row['<TICKER>'], 'price_now']
                table_copy = prev_table.copy()
                table_copy.at[row['<TICKER>'], 'price_now'] = row['price']
                if prev_price != 0:

                    table_copy.at[row['<TICKER>'], 'pnl'] = ((row['price'] - prev_price) / prev_price) * table_copy.at[
                        row['<TICKER>'], 'position']
                else:
                    table_copy.at[row['<TICKER>'], 'pnl'] = 0
                table_copy.at[row['<TICKER>'], 'position'] = 0
                table_copy.at[row['<TICKER>'], 'money'] = table_copy.at[row['<TICKER>'], 'moneyo'] + table_copy.at[
                    row['<TICKER>'], 'pnl']

                table_copy.at['cash', 'money'] += table_copy.at[row['<TICKER>'], 'money']
                table_copy.at[row['<TICKER>'], 'money'] = 0
                table_copy.at[row['<TICKER>'], 'pnl'] = 0
                table_copy.at[row['<TICKER>'], 'position'] = 0
                table_copy.at[row['<TICKER>'], 'posa'] = 0

                # table_copy.at[row['<TICKER>'], 'money']-=row['change']

                table_copy.at[row['<TICKER>'], 'moneyo'] = 0

                prev_table = table_copy.copy()

    sum_money = table_copy.loc['cash', 'money']
    for index, row in info_df3.iterrows():

        if pd.to_datetime(row['date']) == date:

            # проверяем хватает ли денег
            if row['position'] > 0:

                table_copy.at['cash', 'money'] -= row['position'] * sum_money
                table_copy.at[row['<TICKER>'], 'position'] += row['position'] * sum_money
                table_copy.at[row['<TICKER>'], 'price_first'] = row['price']
                table_copy.at[row['<TICKER>'], 'price_now'] = row['price']

                table_copy.at[row['<TICKER>'], 'dadao'] = date
                table_copy.at[row['<TICKER>'], 'moneyo'] = table_copy.at[row['<TICKER>'], 'position']
                table_copy.at[row['<TICKER>'], 'type'] = row['type']
                table_copy.at[row['<TICKER>'], 'money'] = table_copy.at[row['<TICKER>'], 'position']
                table_copy.at[row['<TICKER>'], 'posa'] = row['position']
                prev_table = table_copy.copy()


            # если в таблице сделок дано закрытие позиции, то соответственно закрываем
            elif row['position'] == 0 and prev_table is not None:

                prev_price = prev_table.at[row['<TICKER>'], 'price_now']
                table_copy = prev_table.copy()
                table_copy.at[row['<TICKER>'], 'price_now'] = row['price']
                if prev_price != 0:

                    table_copy.at[row['<TICKER>'], 'pnl'] = ((row['price'] - prev_price) / prev_price) * table_copy.at[
                        row['<TICKER>'], 'position']
                else:
                    table_copy.at[row['<TICKER>'], 'pnl'] = 0
                table_copy.at[row['<TICKER>'], 'position'] = 0
                table_copy.at[row['<TICKER>'], 'money'] = table_copy.at[row['<TICKER>'], 'moneyo'] + table_copy.at[
                    row['<TICKER>'], 'pnl']

                table_copy.at['cash', 'money'] += table_copy.at[row['<TICKER>'], 'money']
                table_copy.at[row['<TICKER>'], 'money'] = 0
                table_copy.at[row['<TICKER>'], 'pnl'] = 0
                table_copy.at[row['<TICKER>'], 'position'] = 0
                table_copy.at[row['<TICKER>'], 'posa'] = 0

                # table_copy.at[row['<TICKER>'], 'money']-=row['change']

                table_copy.at[row['<TICKER>'], 'moneyo'] = 0

                prev_table = table_copy.copy()

    aton[date] = table_copy
    print(date)

full_info
data_dates=[]
data_num=[]

p=0
for k, v in aton.items():
    data_dates.append(k)
    for index, row in v.iterrows():
        p += row['money']

    data_num.append(p)
    p = 0
fig = go.Figure()

# Добавление данных на график
fig.add_trace(go.Scatter(x=data_dates, y=data_num, mode='lines+markers', name='Значения'))

# Настройки осей и меток
fig.update_xaxes(title_text='Дата')
fig.update_yaxes(title_text='pnl')

# Настройки заголовка и размеров графика
fig.update_layout(title_text='Портфель', title_x=0.5, width=800, height=500)

# Отображение графика
fig.show()