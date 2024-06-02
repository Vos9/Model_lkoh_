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
from sklearn.model_selection import TimeSeriesSplit


def resh(stock_to_do):
    # Устанавливаем даты начала и конца периода
    start_date = datetime.datetime(2000, 1, 5)
    end_date = datetime.datetime(2023, 8, 10)

    # Загружаем котировки USDRUB
    usdrub = pdr.get_data_moex("USD000UTSTOM", start=start_date, end=end_date)

    # Загружаем данные по индексу Московской биржи (IMOEX)
    imoex = pdr.get_data_moex("IMOEX", start=start_date, end=end_date)

    # Обрабатываем данные для usdrub,imoex
    usdrub['CLOSE'] = usdrub['CLOSE'].astype(float)
    imoex['CLOSE'] = imoex['CLOSE'].astype(float)

    # Укажите путь к файлу base_dict.p
    file_path = 'base_dict1.p'

    # Чтение данных из файла
    with open('base_dict1.p', 'rb') as file:
        data_set = pickle.load(file)

    #обрабатываем данные
    koef = 0
    for k, v in data_set.items():
        v['CLOSE'] = 0
        v['LOW'] = 0
        v['OPEN'] = 0
        v['HIGH'] = 0
        for index, row in v.iterrows():
            koef = row['PX_LAST_ADJ'] / row['PX_LAST']
            v.at[index, 'CLOSE'] = koef * row['PX_LAST']
            v.at[index, 'LOW'] = koef * row['PX_LOW']
            v.at[index, 'OPEN'] = koef * row['PX_OPEN']
            v.at[index, 'HIGH'] = koef * row['PX_HIGH']

    stock_mapping = \
        {
        'LKOH': ['LKOH','ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                 'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                 'FEES', 'MRKC', 'UPRO'],
        'ROSN': ['ROSN','LKOH', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                 'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                 'FEES', 'MRKC', 'UPRO'],
        'NVTK': ['NVTK','ROSN', 'LKOH', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                 'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                 'FEES', 'MRKC', 'UPRO'],
            'TATN': ['TATN', 'ROSN', 'NVTK', 'LKOH', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'SNGS': ['SNGS', 'ROSN', 'NVTK', 'TATN', 'LKOH', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'SNGSP': ['SNGSP', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'LKOH', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'TRNFP': ['TRNFP', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'LKOH', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'TATNP': ['TATNP', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'LKOH', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],

            'GMKN': ['GMKN', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'LKOH', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'NLMK': ['NLMK', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'LKOH', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'MAGN': ['MAGN', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'LKOH', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'PLZL': ['PLZL', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'LKOH',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'ALRS': ['ALRS', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'LKOH', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'MTLR': ['MTLR', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'LKOH', 'CHMF', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'CHMF': ['CHMF', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'LKOH', 'SELG', 'RASP','MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'MTLRP': ['MTLRP', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','LKOH', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'PHOR': ['PHOR', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SELG', 'RASP','MTLRP', 'LKOH', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],

            'SBER': ['SBER', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT','MTLRP', 'PHOR', 'LKOH', 'VTBR', 'MOEX', 'MTSS', 'RTKM', 'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'SBERP': ['SBERP', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'LKOH', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'MOEX': ['MOEX', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'LKOH', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'VTBR': ['VTBR', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'LKOH', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'MTSS': ['MTSS', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'LKOH', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'RTKM': ['RTKM', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'LKOH',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'IRAO': ['IRAO', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'LKOH', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],
            'HYDR': ['HYDR', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'LKOH',
                     'FEES', 'MRKC', 'UPRO'],
            'FEES': ['FEES', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'LKOH', 'MRKC', 'UPRO'],
            'MRKC': ['MRKC', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'LKOH', 'UPRO'],
            'UPRO': ['UPRO', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'MGNT', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'LKOH'],
            'MGNT': ['MGNT', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'TRNFP', 'TATNP', 'GMKN', 'NLMK', 'MAGN', 'PLZL',
                     'ALRS', 'MTLR', 'CHMF', 'SBERP', 'LKOH', 'MTLRP', 'PHOR', 'SBER', 'VTBR', 'MOEX', 'MTSS', 'RTKM',
                     'IRAO', 'HYDR',
                     'FEES', 'MRKC', 'UPRO'],


    }

    #берем котировки рассматриваемых акций
    if str(stock_to_do) in stock_mapping:
        stocks = stock_mapping[str(stock_to_do)]
        lkoh = data_set[stocks[0]]
        rosn = data_set[stocks[1]]
        nvtk = data_set[stocks[2]]
        tatn = data_set[stocks[3]]
        sngs = data_set[stocks[4]]
        sngsp = data_set[stocks[5]]
        trnfp = data_set[stocks[6]]
        tatnp = data_set[stocks[7]]

        # металлурги
        gmkn = data_set[stocks[8]]
        nlmk = data_set[stocks[9]]
        magn = data_set[stocks[10]]
        plzl = data_set[stocks[11]]
        alrs = data_set[stocks[12]]
        mtlr = data_set[stocks[13]]
        chmf = data_set[stocks[14]]
        selg = data_set[stocks[15]]
        rasp = data_set[stocks[16]]
        mtlrp = data_set[stocks[17]]
        phor = data_set[stocks[18]]

        # финансовый сектор
        sber = data_set[stocks[19]]
        vtbr = data_set[stocks[20]]
        moex = data_set[stocks[21]]

        # сотовая связь
        mtss = data_set[stocks[22]]
        rtkm = data_set[stocks[23]]
        irao = data_set[stocks[24]]
        hydr = data_set[stocks[25]]
        fees = data_set[stocks[26]]
        mrkc = data_set[stocks[27]]
        upro = data_set[stocks[28]]

    # открываем файлы commodities с котировками
    urals = pd.read_csv('commodities/' +'brent.csv')
    metalurgi = pd.read_csv('commodities/' +'metalurgi.csv')
    zoloto = pd.read_csv('commodities/' +'zoloto.csv')
    palladiy = pd.read_csv('commodities/' +'palladiy.csv')
    platina = pd.read_csv('commodities/' +'platina.csv')
    ugol = pd.read_csv('commodities/' +'ugol.csv')
    nikel = pd.read_csv('commodities/' +'nikel.csv')
    ghelezo = pd.read_csv('commodities/' +'ghelezo.csv')
    med = pd.read_csv('commodities/' +'med.csv')
    RUGBITR10Y = pd.read_csv('commodities/' +'RUGBITR10Y.csv')
    RUGBITR1Y = pd.read_csv('commodities/' +'RUGBITR1Y.csv')
    RGBITR = pd.read_csv('commodities/' +'RGBITR.csv')

    #добавляем Лаги и Change в котировки акций для добавления признаков
    # Список фреймов данных для обработки
    data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl, alrs,
                   mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]

    # Итерируемся по каждому фрейму данных
    for df in data_frames:
        columns_to_convert = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'PX_VOLUME']

        # Преобразовываем указанные столбцы к типу float
        df[columns_to_convert] = df[columns_to_convert].astype(float)

    data_frames = [imoex, usdrub, lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk,
                   magn, plzl, alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]
    data_frame_names = ['imoex', 'usdrub', 'lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg',
                        'rasp', 'mtlrp', 'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                        'moex', 'mtss', 'rtkm', 'irao',
                        'hydr', 'fees', 'mrkc', 'upro']
    # Список DataFrame, которые не должны исключаться из операции
    exclude_frames = ['usdrub', 'imoex']

    # Переберите все данные и выполните операции с ними
    for i in range(1, 120):
        for idx, df in enumerate(data_frames):
            df_name = data_frame_names[idx]
            df[f'lag_value{i}c{df_name}'] = df['CLOSE'].shift(i)

    for i in range(1, 120):
        for df in data_frames[2:]:
            df[f'lag_value{i}c_val'] = df['PX_VOLUME'].shift(i)

    data_frames = [imoex, usdrub, lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk,
                   magn, plzl, alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]
    data_frame_names = ['imoex', 'usdrub', 'lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg',
                        'rasp', 'mtlrp', 'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                        'moex', 'mtss', 'rtkm', 'irao',
                        'hydr', 'fees', 'mrkc', 'upro']

    # Переберите все данные и выполните операции с ними
    for i in range(2, 120):
        for idx, df in enumerate(data_frames):
            df_name = data_frame_names[idx]
            df[f'change{i}_{df_name}'] = ((df[f'lag_value{i - 1}c{df_name}'] - df[f'lag_value{i}c{df_name}']) / df[
                f'lag_value{i}c{df_name}']) * 100

    for df_name, df in zip(data_frame_names, data_frames):
        df[f'change1_{df_name}'] = ((df['CLOSE'] - df[f'lag_value1c{df_name}']) / df[f'lag_value1c{df_name}']) * 100

    data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl, alrs,
                   mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]
    data_frame_names = ['lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg', 'rasp', 'mtlrp',
                        'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr', 'moex', 'mtss',
                        'rtkm', 'irao',
                        'hydr', 'fees', 'mrkc', 'upro']
    for i in range(2, 120):
        for idx, df in enumerate(data_frames):
            df_name = data_frame_names[idx]

            df[f'change{i}_{df_name}_val'] = ((df[f'lag_value{i - 1}c_val'] - df[f'lag_value{i}c_val']) / df[
                f'lag_value{i}c_val']) * 100

    for df_name, df in zip(data_frame_names, data_frames):
        df[f'change1_{df_name}_val'] = ((df['PX_VOLUME'] - df['lag_value1c_val']) / df['lag_value1c_val']) * 100

    data_frames = [imoex, usdrub, lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk,
                   magn, plzl, alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]
    data_frame_names = ['imoex', 'usdrub', 'lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg',
                        'rasp', 'mtlrp', 'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                        'moex', 'mtss', 'rtkm', 'irao',
                        'hydr', 'fees', 'mrkc', 'upro']
    #добавляем в признаки изменение за последний месяц/неделю/полгода
    def add_change_columns(df, name):
        df[f'change_yesterday_{name}'] = df[f'change2_{name}']

        df[f'change_3days_{name}'] = df[f'change2_{name}'] + df[f'change3_{name}'] + df[f'change4_{name}']
        df[f'change_last_month_{name}'] = df[f'change2_{name}']
        df[f'change_last_3month_{name}'] = df[f'change2_{name}']
        df[f'change_last_half_of_year_{name}'] = df[f'change2_{name}']
        df[f'change_last_week_{name}'] = df[f'change2_{name}']

        for i in range(3, 120):
            df[f'change_last_half_of_year_{name}'] += df[f'change{i}_{name}']
            if i <= 32:
                df[f'change_last_month_{name}'] += df[f'change{i}_{name}']
            if i <= 9:
                df[f'change_last_week_{name}'] += df[f'change{i}_{name}']
            if i <= 92:
                df[f'change_last_3month_{name}'] += df[f'change{i}_{name}']

    # Создайте список фреймов данных и их имена
    data_frames = [
        (lkoh, 'lkoh'),
        (usdrub, 'usdrub'),
        (imoex, 'imoex'),
        (metalurgi, 'metalurgi'),
        (urals, 'urals'),
        (zoloto, 'zoloto'),
        (palladiy, 'palladiy'),
        (platina, 'platina'),
        (ugol, 'ugol'),
        (nikel, 'nikel'),
        (ghelezo, 'ghelezo'),
        (med, 'med'),
        (RUGBITR10Y, 'RUGBITR10Y'),
        (RUGBITR1Y, 'RUGBITR1Y'),
        (RGBITR, 'RGBITR')

    ]

    # Примените функцию add_change_columns к каждому фрейму данных
    for df, name in data_frames:
        add_change_columns(df, name)

    #вычисляем волатильность рассматриваемой акции для добавления в признаки
    lkoh['volatility_3'] = lkoh[['change2_lkoh', 'change3_lkoh', 'change4_lkoh']].std(axis=1)
    lkoh['volatility_7'] = lkoh[
        ['change2_lkoh', 'change3_lkoh', 'change4_lkoh', 'change5_lkoh', 'change6_lkoh', 'change7_lkoh',
         'change8_lkoh']].std(axis=1)
    lkoh['volatility_10'] = lkoh[
        ['change2_lkoh', 'change3_lkoh', 'change4_lkoh', 'change5_lkoh', 'change6_lkoh', 'change7_lkoh', 'change8_lkoh',
         'change9_lkoh', 'change10_lkoh', 'change11_lkoh']].std(axis=1)
    lkoh['volatility_15'] = lkoh[
        ['change2_lkoh', 'change3_lkoh', 'change4_lkoh', 'change5_lkoh', 'change6_lkoh', 'change7_lkoh', 'change8_lkoh',
         'change9_lkoh', 'change10_lkoh', 'change11_lkoh', 'change12_lkoh', 'change13_lkoh', 'change14_lkoh',
         'change15_lkoh', 'change16_lkoh']].std(axis=1)

    lkoh['volatility_30'] = lkoh[
        ['change2_lkoh', 'change3_lkoh', 'change4_lkoh', 'change5_lkoh', 'change6_lkoh', 'change7_lkoh', 'change8_lkoh',
         'change9_lkoh', 'change10_lkoh', 'change11_lkoh', 'change12_lkoh', 'change13_lkoh', 'change14_lkoh',
         'change15_lkoh', 'change16_lkoh', 'change17_lkoh', 'change18_lkoh', 'change19_lkoh', 'change20_lkoh',
         'change21_lkoh', 'change22_lkoh', 'change23_lkoh', 'change24_lkoh', 'change25_lkoh', 'change26_lkoh',
         'change27_lkoh', 'change28_lkoh', 'change29_lkoh', 'change30_lkoh', 'change31_lkoh']].std(axis=1)

    #переименовая индекса для будущего joinа
    # Список фреймов данных
    data_frames = [lkoh, rosn, nvtk, sngsp, trnfp, tatn, tatnp, sngs, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl, alrs,
                   mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]

    # Примените переименование индекса к каждому фрейму данных
    for df in data_frames:
        df.index = df.index.rename('TRADEDATE')

    #чистим на nan значения по последнему лагу

    data_frames = [imoex, usdrub, lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk,
                   magn, plzl, alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                   hydr, fees, mrkc, upro]
    data_frame_names = ['imoex', 'usdrub', 'lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg',
                        'rasp', 'mtlrp', 'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                        'moex', 'mtss', 'rtkm', 'irao',
                        'hydr', 'fees', 'mrkc', 'upro']

    for df in range(len(data_frames)):
        data_frames[df].dropna(subset=[f'lag_value119c{data_frame_names[df]}'], inplace=True)

    RUGBITR10Y['lag_doha_RUGBITR10Y'] = RUGBITR10Y['doha_RUGBITR10Y'].shift(1)
    RUGBITR1Y['lag_doha_RUGBITR1Y'] = RUGBITR1Y['doha_RUGBITR1Y'].shift(1)
    RUGBITR10Y['change_doha_RUGBITR10Y'] = RUGBITR10Y['doha_RUGBITR10Y'] - RUGBITR10Y['lag_doha_RUGBITR10Y']
    RUGBITR1Y['change_doha_RUGBITR1Y'] = RUGBITR1Y['doha_RUGBITR1Y'] - RUGBITR1Y['lag_doha_RUGBITR1Y']
    RUGBITR10Y.drop(0, inplace=True, axis=0)
    RUGBITR1Y.drop(0, inplace=True, axis=0)

    def preprocess_dataframe(df, date_column_name, index_column_name):
        df.rename(columns={date_column_name: 'TRADEDATE'}, inplace=True)
        df.set_index('TRADEDATE', inplace=True)
        df.index = pd.to_datetime(df.index)

    # Применяем функцию к каждому DataFrame
    preprocess_dataframe(metalurgi, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(urals, 'Unnamed: 0', 'TRADEDATE')
    preprocess_dataframe(zoloto, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(palladiy, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(ugol, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(nikel, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(ghelezo, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(med, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(platina, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(RUGBITR10Y, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(RUGBITR1Y, '<DATE>', 'TRADEDATE')
    preprocess_dataframe(RGBITR, '<DATE>', 'TRADEDATE')

    RGBITR['doha_RGBITR'] = RGBITR['Unnamed: 18'].astype(float)

    # создание итогового датафрейма для теста модельки
    itog = pd.DataFrame()
    itog.index = imoex.index

    # добавление стобцов из других датафреймов
    itog['change1_imoex'] = imoex['change1_imoex']
    itog['change2_imoex'] = imoex['change2_imoex']
    itog['change3_imoex'] = imoex['change3_imoex']
    itog['change4_imoex'] = imoex['change4_imoex']
    itog['change5_imoex'] = imoex['change5_imoex']
    itog['change6_imoex'] = imoex['change6_imoex']
    itog['change7_imoex'] = imoex['change7_imoex']
    itog['change8_imoex'] = imoex['change8_imoex']
    itog['change9_imoex'] = imoex['change9_imoex']
    itog['change10_imoex'] = imoex['change10_imoex']

    itog['change29_imoex'] = imoex['change_3days_imoex']
    itog['change22_imoex'] = imoex['change_last_month_imoex']
    itog['change23_imoex'] = imoex['change_last_3month_imoex']
    itog['change24_imoex'] = imoex['change_last_half_of_year_imoex']
    itog['change25_imoex'] = imoex['change_last_week_imoex']

    commodities = ['usdrub', 'metalurgi', 'urals', 'zoloto', 'palladiy', 'platina', 'ugol', 'nikel', 'ghelezo', 'med',
                   'RUGBITR10Y', 'RUGBITR1Y', 'RGBITR']
    change_columns = ['change1', 'change2', 'change3', 'change4', 'change5', 'change6', 'change7', 'change8', 'change9',
                      'change10', 'change_3days', 'change_last_month', 'change_last_3month', 'change_last_half_of_year',
                      'change_last_week']

    for commodity in commodities:
        for column in change_columns:
            column_name = f'{column}_{commodity}'
            itog = itog.join(eval(f'{commodity}["{column}_{commodity}"]'), on='TRADEDATE')

    itog = itog.join(RUGBITR10Y['doha_RUGBITR10Y'], on='TRADEDATE')
    itog = itog.join(RUGBITR1Y['doha_RUGBITR1Y'], on='TRADEDATE')
    itog = itog.join(RUGBITR10Y['change_doha_RUGBITR10Y'], on='TRADEDATE')
    itog = itog.join(RUGBITR1Y['change_doha_RUGBITR1Y'], on='TRADEDATE')

    if str(stock_to_do) in ['SBER', 'IRAO', 'HYDR', 'FEES', 'MRKC', 'UPRO', 'SBERP', 'MOEX', 'VTBR', 'MGNT', 'RTKM', 'MTSS']:
        data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl,
                       alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                       hydr, fees, mrkc, upro]
        data_frame_names = ['lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg', 'rasp', 'mtlrp',
                            'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr', 'moex',
                            'mtss', 'rtkm', 'irao',
                            'hydr', 'fees', 'mrkc', 'upro']
    else:
        data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl,
                       alrs, mtlr, chmf, phor, sber, vtbr, mtss, rtkm, irao,
                       hydr, fees, mrkc, upro]
        data_frame_names = ['lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg', 'rasp', 'mtlrp',
                            'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                            'mtss', 'rtkm', 'irao',
                            'hydr', 'fees', 'mrkc', 'upro']

    for i in range(len(data_frame_names)):
        itog = itog.join(data_frames[i]['change1' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change2' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change3' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change4' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change5' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change6' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change7' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change8' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change9' + '_' + data_frame_names[i]], on='TRADEDATE')
        itog = itog.join(data_frames[i]['change10' + '_' + data_frame_names[i]], on='TRADEDATE')

    itog = itog.join(lkoh['volatility_3'], on='TRADEDATE')
    itog = itog.join(lkoh['volatility_7'], on='TRADEDATE')
    itog = itog.join(lkoh['volatility_10'], on='TRADEDATE')
    itog = itog.join(lkoh['volatility_15'], on='TRADEDATE')
    itog = itog.join(lkoh['volatility_30'], on='TRADEDATE')

    itog = itog.join(lkoh['change1_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change2_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change3_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change4_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change5_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change6_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change7_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change8_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change9_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change10_lkoh_val'], on='TRADEDATE')
    itog = itog.join(lkoh['change_last_month_lkoh'], on='TRADEDATE')
    itog = itog.join(lkoh['change_last_3month_lkoh'], on='TRADEDATE')
    itog = itog.join(lkoh['change_last_half_of_year_lkoh'], on='TRADEDATE')
    itog = itog.join(lkoh['change_last_week_lkoh'], on='TRADEDATE')
    itog = itog.join(lkoh['change_3days_lkoh'], on='TRADEDATE')

    itog.dropna(inplace=True)
    itog1 = itog.copy()

    #логорифмирование данных
    itog['change1_usdrub'] = itog['change1_usdrub'].replace(np.inf, np.nan)
    itog['change2_usdrub'] = itog['change2_usdrub'].replace(np.inf, np.nan)
    itog['change3_usdrub'] = itog['change3_usdrub'].replace(np.inf, np.nan)
    itog['change4_usdrub'] = itog['change4_usdrub'].replace(np.inf, np.nan)
    itog['change5_usdrub'] = itog['change5_usdrub'].replace(np.inf, np.nan)
    itog['change6_usdrub'] = itog['change6_usdrub'].replace(np.inf, np.nan)
    itog['change7_usdrub'] = itog['change7_usdrub'].replace(np.inf, np.nan)
    itog['change8_usdrub'] = itog['change8_usdrub'].replace(np.inf, np.nan)
    itog['change9_usdrub'] = itog['change9_usdrub'].replace(np.inf, np.nan)
    itog['change10_usdrub'] = itog['change10_usdrub'].replace(np.inf, np.nan)
    itog['change_3days_usdrub'] = itog['change_3days_usdrub'].replace(np.inf, np.nan)
    itog['change_last_month_usdrub'] = itog['change_last_month_usdrub'].replace(np.inf, np.nan)
    itog['change_last_3month_usdrub'] = itog['change_last_3month_usdrub'].replace(np.inf, np.nan)
    itog['change_last_half_of_year_usdrub'] = itog['change_last_half_of_year_usdrub'].replace(np.inf, np.nan)
    itog['change_last_week_usdrub'] = itog['change_last_week_usdrub'].replace(np.inf, np.nan)

    # логорифмируем данные
    # логорифмируем данные
    for i in ['metalurgi', 'zoloto', 'palladiy', 'platina', 'ugol', 'nikel', 'ghelezo', 'med', 'urals', 'RUGBITR10Y',
              'RUGBITR1Y', 'RGBITR']:
        itog['change1_' + i] = np.log1p((itog['change1_' + i] - itog['change1_' + i].min()) / (
                    itog['change1_' + i].max() - itog['change1_' + i].min()))
        itog['change2_' + i] = np.log1p((itog['change2_' + i] - itog['change2_' + i].min()) / (
                    itog['change2_' + i].max() - itog['change2_' + i].min()))
        itog['change3_' + i] = np.log1p((itog['change3_' + i] - itog['change3_' + i].min()) / (
                    itog['change3_' + i].max() - itog['change3_' + i].min()))
        itog['change4_' + i] = np.log1p((itog['change4_' + i] - itog['change4_' + i].min()) / (
                    itog['change4_' + i].max() - itog['change4_' + i].min()))
        itog['change5_' + i] = np.log1p((itog['change5_' + i] - itog['change5_' + i].min()) / (
                    itog['change5_' + i].max() - itog['change5_' + i].min()))
        itog['change6_' + i] = np.log1p((itog['change6_' + i] - itog['change6_' + i].min()) / (
                    itog['change6_' + i].max() - itog['change6_' + i].min()))
        itog['change7_' + i] = np.log1p((itog['change7_' + i] - itog['change7_' + i].min()) / (
                    itog['change7_' + i].max() - itog['change7_' + i].min()))
        itog['change8_' + i] = np.log1p((itog['change8_' + i] - itog['change8_' + i].min()) / (
                    itog['change8_' + i].max() - itog['change8_' + i].min()))
        itog['change9_' + i] = np.log1p((itog['change9_' + i] - itog['change9_' + i].min()) / (
                    itog['change9_' + i].max() - itog['change9_' + i].min()))
        itog['change10_' + i] = np.log1p((itog['change10_' + i] - itog['change10_' + i].min()) / (
                    itog['change10_' + i].max() - itog['change10_' + i].min()))

        itog['change_3days_' + i] = np.log1p((itog['change_3days_' + i] - itog['change_3days_' + i].min()) / (
                    itog['change_3days_' + i].max() - itog['change_3days_' + i].min()))
        itog['change_last_month_' + i] = np.log1p(
            (itog['change_last_month_' + i] - itog['change_last_month_' + i].min()) / (
                        itog['change_last_month_' + i].max() - itog['change_last_month_' + i].min()))
        itog['change_last_3month_' + i] = np.log1p(
            (itog['change_last_3month_' + i] - itog['change_last_3month_' + i].min()) / (
                        itog['change_last_3month_' + i].max() - itog['change_last_3month_' + i].min()))
        itog['change_last_half_of_year_' + i] = np.log1p(
            (itog['change_last_half_of_year_' + i] - itog['change_last_half_of_year_' + i].min()) / (
                        itog['change_last_half_of_year_' + i].max() - itog['change_last_half_of_year_' + i].min()))
        itog['change_last_week_' + i] = np.log1p(
            (itog['change_last_week_' + i] - itog['change_last_week_' + i].min()) / (
                        itog['change_last_week_' + i].max() - itog['change_last_week_' + i].min()))

    if str(stock_to_do) in ['SBER', 'IRAO', 'HYDR', 'FEES', 'MRKC', 'UPRO', 'SBER', 'SBERP', 'MOEX', 'VTBR', 'MGNT',
                            'RTKM', 'MTSS']:
        data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl,
                       alrs, mtlr, chmf, phor, sber, vtbr, moex, mtss, rtkm, irao,
                       hydr, fees, mrkc, upro]
        data_frame_names = ['lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg', 'rasp', 'mtlrp',
                            'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr', 'moex',
                            'mtss', 'rtkm', 'irao',
                            'hydr', 'fees', 'mrkc', 'upro']
    else:
        data_frames = [lkoh, rosn, nvtk, sngsp, sngs, trnfp, tatn, tatnp, selg, rasp, mtlrp, gmkn, nlmk, magn, plzl,
                       alrs, mtlr, chmf, phor, sber, vtbr, mtss, rtkm, irao,
                       hydr, fees, mrkc, upro]
        data_frame_names = ['lkoh', 'rosn', 'nvtk', 'sngsp', 'sngs', 'trnfp', 'tatn', 'tatnp', 'selg', 'rasp', 'mtlrp',
                            'gmkn', 'nlmk', 'magn', 'plzl', 'alrs', 'mtlr', 'chmf', 'phor', 'sber', 'vtbr',
                            'mtss', 'rtkm', 'irao',
                            'hydr', 'fees', 'mrkc', 'upro']

    for i in range(len(data_frame_names)):
        itog['change1_' + data_frame_names[i]] = np.log1p(
            (itog['change1_' + data_frame_names[i]] - itog['change1_' + data_frame_names[i]].min()) / (
                        itog['change1_' + data_frame_names[i]].max() - itog['change1_' + data_frame_names[i]].min()))
        itog['change2_' + data_frame_names[i]] = np.log1p(
            (itog['change2_' + data_frame_names[i]] - itog['change2_' + data_frame_names[i]].min()) / (
                        itog['change2_' + data_frame_names[i]].max() - itog['change2_' + data_frame_names[i]].min()))
        itog['change3_' + data_frame_names[i]] = np.log1p(
            (itog['change3_' + data_frame_names[i]] - itog['change3_' + data_frame_names[i]].min()) / (
                        itog['change3_' + data_frame_names[i]].max() - itog['change3_' + data_frame_names[i]].min()))
        itog['change4_' + data_frame_names[i]] = np.log1p(
            (itog['change4_' + data_frame_names[i]] - itog['change4_' + data_frame_names[i]].min()) / (
                        itog['change4_' + data_frame_names[i]].max() - itog['change4_' + data_frame_names[i]].min()))
        itog['change5_' + data_frame_names[i]] = np.log1p(
            (itog['change5_' + data_frame_names[i]] - itog['change5_' + data_frame_names[i]].min()) / (
                        itog['change5_' + data_frame_names[i]].max() - itog['change5_' + data_frame_names[i]].min()))
        itog['change6_' + data_frame_names[i]] = np.log1p(
            (itog['change6_' + data_frame_names[i]] - itog['change6_' + data_frame_names[i]].min()) / (
                        itog['change6_' + data_frame_names[i]].max() - itog['change6_' + data_frame_names[i]].min()))
        itog['change7_' + data_frame_names[i]] = np.log1p(
            (itog['change7_' + data_frame_names[i]] - itog['change7_' + data_frame_names[i]].min()) / (
                        itog['change7_' + data_frame_names[i]].max() - itog['change7_' + data_frame_names[i]].min()))
        itog['change8_' + data_frame_names[i]] = np.log1p(
            (itog['change8_' + data_frame_names[i]] - itog['change8_' + data_frame_names[i]].min()) / (
                        itog['change8_' + data_frame_names[i]].max() - itog['change8_' + data_frame_names[i]].min()))
        itog['change9_' + data_frame_names[i]] = np.log1p(
            (itog['change9_' + data_frame_names[i]] - itog['change9_' + data_frame_names[i]].min()) / (
                        itog['change9_' + data_frame_names[i]].max() - itog['change9_' + data_frame_names[i]].min()))
        itog['change10_' + data_frame_names[i]] = np.log1p(
            (itog['change10_' + data_frame_names[i]] - itog['change10_' + data_frame_names[i]].min()) / (
                        itog['change10_' + data_frame_names[i]].max() - itog['change10_' + data_frame_names[i]].min()))

    itog['volatility_3'] = np.log1p(
        (itog['volatility_3'] - itog['volatility_3'].min()) / (itog['volatility_3'].max() - itog['volatility_3'].min()))
    itog['volatility_7'] = np.log1p(
        (itog['volatility_7'] - itog['volatility_7'].min()) / (itog['volatility_7'].max() - itog['volatility_7'].min()))
    itog['volatility_10'] = np.log1p((itog['volatility_10'] - itog['volatility_10'].min()) / (
                itog['volatility_10'].max() - itog['volatility_10'].min()))
    itog['volatility_15'] = np.log1p((itog['volatility_15'] - itog['volatility_15'].min()) / (
                itog['volatility_15'].max() - itog['volatility_15'].min()))
    itog['volatility_30'] = np.log1p((itog['volatility_30'] - itog['volatility_30'].min()) / (
                itog['volatility_30'].max() - itog['volatility_30'].min()))

    itog['change1_lkoh_val'] = np.log1p((itog['change1_lkoh_val'] - itog['change1_lkoh_val'].min()) / (
                itog['change1_lkoh_val'].max() - itog['change1_lkoh_val'].min()))
    itog['change2_lkoh_val'] = np.log1p((itog['change2_lkoh_val'] - itog['change2_lkoh_val'].min()) / (
                itog['change2_lkoh_val'].max() - itog['change2_lkoh_val'].min()))
    itog['change3_lkoh_val'] = np.log1p((itog['change3_lkoh_val'] - itog['change3_lkoh_val'].min()) / (
                itog['change3_lkoh_val'].max() - itog['change3_lkoh_val'].min()))
    itog['change4_lkoh_val'] = np.log1p((itog['change4_lkoh_val'] - itog['change4_lkoh_val'].min()) / (
                itog['change4_lkoh_val'].max() - itog['change4_lkoh_val'].min()))
    itog['change5_lkoh_val'] = np.log1p((itog['change5_lkoh_val'] - itog['change5_lkoh_val'].min()) / (
                itog['change5_lkoh_val'].max() - itog['change5_lkoh_val'].min()))
    itog['change6_lkoh_val'] = np.log1p((itog['change6_lkoh_val'] - itog['change6_lkoh_val'].min()) / (
                itog['change6_lkoh_val'].max() - itog['change6_lkoh_val'].min()))
    itog['change7_lkoh_val'] = np.log1p((itog['change7_lkoh_val'] - itog['change7_lkoh_val'].min()) / (
                itog['change7_lkoh_val'].max() - itog['change7_lkoh_val'].min()))
    itog['change8_lkoh_val'] = np.log1p((itog['change8_lkoh_val'] - itog['change8_lkoh_val'].min()) / (
                itog['change8_lkoh_val'].max() - itog['change8_lkoh_val'].min()))
    itog['change9_lkoh_val'] = np.log1p((itog['change9_lkoh_val'] - itog['change9_lkoh_val'].min()) / (
                itog['change9_lkoh_val'].max() - itog['change9_lkoh_val'].min()))
    itog['change10_lkoh_val'] = np.log1p((itog['change10_lkoh_val'] - itog['change10_lkoh_val'].min()) / (
                itog['change10_lkoh_val'].max() - itog['change10_lkoh_val'].min()))

    itog['change12_lkoh'] = itog['change2_lkoh'] / itog['change2_imoex']
    itog['change13_lkoh'] = itog['change3_lkoh'] / itog['change3_imoex']
    itog['change14_lkoh'] = itog['change4_lkoh'] / itog['change4_imoex']
    itog['change15_lkoh'] = itog['change5_lkoh'] / itog['change5_imoex']
    itog['change16_lkoh'] = itog['change6_lkoh'] / itog['change6_imoex']
    itog['change17_lkoh'] = itog['change7_lkoh'] / itog['change7_imoex']
    itog['change18_lkoh'] = itog['change8_lkoh'] / itog['change8_imoex']
    itog['change19_lkoh'] = itog['change9_lkoh'] / itog['change9_imoex']
    itog['change110_lkoh'] = itog['change10_lkoh'] / itog['change10_imoex']

    itog['change12_lkoh'] = itog['change12_lkoh'].replace(-np.inf, np.nan)
    itog['change13_lkoh'] = itog['change13_lkoh'].replace(-np.inf, np.nan)
    itog['change14_lkoh'] = itog['change14_lkoh'].replace(-np.inf, np.nan)
    itog['change15_lkoh'] = itog['change15_lkoh'].replace(-np.inf, np.nan)
    itog['change16_lkoh'] = itog['change16_lkoh'].replace(-np.inf, np.nan)
    itog['change17_lkoh'] = itog['change17_lkoh'].replace(-np.inf, np.nan)
    itog['change18_lkoh'] = itog['change18_lkoh'].replace(-np.inf, np.nan)
    itog['change19_lkoh'] = itog['change19_lkoh'].replace(-np.inf, np.nan)
    itog['change110_lkoh'] = itog['change110_lkoh'].replace(-np.inf, np.nan)

    itog['change12_lkoh'] = itog['change12_lkoh'].replace(np.inf, np.nan)
    itog['change13_lkoh'] = itog['change13_lkoh'].replace(np.inf, np.nan)
    itog['change14_lkoh'] = itog['change14_lkoh'].replace(np.inf, np.nan)
    itog['change15_lkoh'] = itog['change15_lkoh'].replace(np.inf, np.nan)
    itog['change16_lkoh'] = itog['change16_lkoh'].replace(np.inf, np.nan)
    itog['change17_lkoh'] = itog['change17_lkoh'].replace(np.inf, np.nan)
    itog['change18_lkoh'] = itog['change18_lkoh'].replace(np.inf, np.nan)
    itog['change19_lkoh'] = itog['change19_lkoh'].replace(np.inf, np.nan)
    itog['change110_lkoh'] = itog['change110_lkoh'].replace(np.inf, np.nan)

    itog['change12_lkoh'] = np.log1p((itog['change12_lkoh'] - itog['change12_lkoh'].min()) / (
                itog['change12_lkoh'].max() - itog['change12_lkoh'].min()))
    itog['change13_lkoh'] = np.log1p((itog['change13_lkoh'] - itog['change13_lkoh'].min()) / (
                itog['change13_lkoh'].max() - itog['change13_lkoh'].min()))
    itog['change14_lkoh'] = np.log1p((itog['change14_lkoh'] - itog['change14_lkoh'].min()) / (
                itog['change14_lkoh'].max() - itog['change14_lkoh'].min()))
    itog['change15_lkoh'] = np.log1p((itog['change15_lkoh'] - itog['change15_lkoh'].min()) / (
                itog['change15_lkoh'].max() - itog['change15_lkoh'].min()))
    itog['change16_lkoh'] = np.log1p((itog['change16_lkoh'] - itog['change16_lkoh'].min()) / (
                itog['change16_lkoh'].max() - itog['change16_lkoh'].min()))
    itog['change17_lkoh'] = np.log1p((itog['change17_lkoh'] - itog['change17_lkoh'].min()) / (
                itog['change17_lkoh'].max() - itog['change17_lkoh'].min()))
    itog['change18_lkoh'] = np.log1p((itog['change18_lkoh'] - itog['change18_lkoh'].min()) / (
                itog['change18_lkoh'].max() - itog['change18_lkoh'].min()))
    itog['change19_lkoh'] = np.log1p((itog['change19_lkoh'] - itog['change19_lkoh'].min()) / (
                itog['change19_lkoh'].max() - itog['change19_lkoh'].min()))
    itog['change110_lkoh'] = np.log1p((itog['change110_lkoh'] - itog['change110_lkoh'].min()) / (
                itog['change110_lkoh'].max() - itog['change110_lkoh'].min()))

    itog['change_last_month_lkoh'] = np.log1p(
        (itog['change_last_month_lkoh'] - itog['change_last_month_lkoh'].min()) / (
                    itog['change_last_month_lkoh'].max() - itog['change_last_month_lkoh'].min()))
    itog['change_last_3month_lkoh'] = np.log1p(
        (itog['change_last_3month_lkoh'] - itog['change_last_3month_lkoh'].min()) / (
                    itog['change_last_3month_lkoh'].max() - itog['change_last_3month_lkoh'].min()))
    itog['change_last_half_of_year_lkoh'] = np.log1p(
        (itog['change_last_half_of_year_lkoh'] - itog['change_last_half_of_year_lkoh'].min()) / (
                    itog['change_last_half_of_year_lkoh'].max() - itog['change_last_half_of_year_lkoh'].min()))
    itog['change_last_week_lkoh'] = np.log1p((itog['change_last_week_lkoh'] - itog['change_last_week_lkoh'].min()) / (
                itog['change_last_week_lkoh'].max() - itog['change_last_week_lkoh'].min()))
    itog['change_3days_lkoh'] = np.log1p((itog['change_3days_lkoh'] - itog['change_3days_lkoh'].min()) / (
                itog['change_3days_lkoh'].max() - itog['change_3days_lkoh'].min()))

    itog['change1_imoex'] = np.log1p((itog['change1_imoex'] - itog['change1_imoex'].min()) / (
                itog['change1_imoex'].max() - itog['change1_imoex'].min()))
    itog['change2_imoex'] = np.log1p((itog['change2_imoex'] - itog['change2_imoex'].min()) / (
                itog['change2_imoex'].max() - itog['change2_imoex'].min()))
    itog['change3_imoex'] = np.log1p((itog['change3_imoex'] - itog['change3_imoex'].min()) / (
                itog['change3_imoex'].max() - itog['change3_imoex'].min()))
    itog['change4_imoex'] = np.log1p((itog['change4_imoex'] - itog['change4_imoex'].min()) / (
                itog['change4_imoex'].max() - itog['change4_imoex'].min()))
    itog['change5_imoex'] = np.log1p((itog['change5_imoex'] - itog['change5_imoex'].min()) / (
                itog['change5_imoex'].max() - itog['change5_imoex'].min()))
    itog['change6_imoex'] = np.log1p((itog['change6_imoex'] - itog['change6_imoex'].min()) / (
                itog['change6_imoex'].max() - itog['change6_imoex'].min()))
    itog['change7_imoex'] = np.log1p((itog['change7_imoex'] - itog['change7_imoex'].min()) / (
                itog['change7_imoex'].max() - itog['change7_imoex'].min()))
    itog['change8_imoex'] = np.log1p((itog['change8_imoex'] - itog['change8_imoex'].min()) / (
                itog['change8_imoex'].max() - itog['change8_imoex'].min()))
    itog['change9_imoex'] = np.log1p((itog['change9_imoex'] - itog['change9_imoex'].min()) / (
                itog['change9_imoex'].max() - itog['change9_imoex'].min()))
    itog['change10_imoex'] = np.log1p((itog['change10_imoex'] - itog['change10_imoex'].min()) / (
                itog['change10_imoex'].max() - itog['change10_imoex'].min()))

    itog['change29_imoex'] = np.log1p((itog['change29_imoex'] - itog['change29_imoex'].min()) / (
                itog['change29_imoex'].max() - itog['change29_imoex'].min()))
    itog['change22_imoex'] = np.log1p((itog['change22_imoex'] - itog['change22_imoex'].min()) / (
                itog['change22_imoex'].max() - itog['change22_imoex'].min()))
    itog['change23_imoex'] = np.log1p((itog['change23_imoex'] - itog['change23_imoex'].min()) / (
                itog['change23_imoex'].max() - itog['change23_imoex'].min()))
    itog['change24_imoex'] = np.log1p((itog['change24_imoex'] - itog['change24_imoex'].min()) / (
                itog['change24_imoex'].max() - itog['change24_imoex'].min()))
    itog['change25_imoex'] = np.log1p((itog['change25_imoex'] - itog['change25_imoex'].min()) / (
                itog['change25_imoex'].max() - itog['change25_imoex'].min()))

    itog['change1_usdrub'] = np.log1p((itog['change1_usdrub'] - itog['change1_usdrub'].min()) / (
                itog['change1_usdrub'].max() - itog['change1_usdrub'].min()))
    itog['change2_usdrub'] = np.log1p((itog['change2_usdrub'] - itog['change2_usdrub'].min()) / (
                itog['change2_usdrub'].max() - itog['change2_usdrub'].min()))
    itog['change3_usdrub'] = np.log1p((itog['change3_usdrub'] - itog['change3_usdrub'].min()) / (
                itog['change3_usdrub'].max() - itog['change3_usdrub'].min()))
    itog['change4_usdrub'] = np.log1p((itog['change4_usdrub'] - itog['change4_usdrub'].min()) / (
                itog['change4_usdrub'].max() - itog['change4_usdrub'].min()))
    itog['change5_usdrub'] = np.log1p((itog['change5_usdrub'] - itog['change5_usdrub'].min()) / (
                itog['change5_usdrub'].max() - itog['change5_usdrub'].min()))
    itog['change6_usdrub'] = np.log1p((itog['change6_usdrub'] - itog['change6_usdrub'].min()) / (
                itog['change6_usdrub'].max() - itog['change6_usdrub'].min()))
    itog['change7_usdrub'] = np.log1p((itog['change7_usdrub'] - itog['change7_usdrub'].min()) / (
                itog['change7_usdrub'].max() - itog['change7_usdrub'].min()))
    itog['change8_usdrub'] = np.log1p((itog['change8_usdrub'] - itog['change8_usdrub'].min()) / (
                itog['change8_usdrub'].max() - itog['change8_usdrub'].min()))
    itog['change9_usdrub'] = np.log1p((itog['change9_usdrub'] - itog['change9_usdrub'].min()) / (
                itog['change9_usdrub'].max() - itog['change9_usdrub'].min()))
    itog['change10_usdrub'] = np.log1p((itog['change10_usdrub'] - itog['change10_usdrub'].min()) / (
                itog['change10_usdrub'].max() - itog['change10_usdrub'].min()))

    itog['change_3days_usdrub'] = np.log1p((itog['change_3days_usdrub'] - itog['change_3days_usdrub'].min()) / (
                itog['change_3days_usdrub'].max() - itog['change_3days_usdrub'].min()))
    itog['change_last_month_usdrub'] = np.log1p(
        (itog['change_last_month_usdrub'] - itog['change_last_month_usdrub'].min()) / (
                    itog['change_last_month_usdrub'].max() - itog['change_last_month_usdrub'].min()))
    itog['change_last_3month_usdrub'] = np.log1p(
        (itog['change_last_3month_usdrub'] - itog['change_last_3month_usdrub'].min()) / (
                    itog['change_last_3month_usdrub'].max() - itog['change_last_3month_usdrub'].min()))
    itog['change_last_half_of_year_usdrub'] = np.log1p(
        (itog['change_last_half_of_year_usdrub'] - itog['change_last_half_of_year_usdrub'].min()) / (
                    itog['change_last_half_of_year_usdrub'].max() - itog['change_last_half_of_year_usdrub'].min()))
    itog['change_last_week_usdrub'] = np.log1p(
        (itog['change_last_week_usdrub'] - itog['change_last_week_usdrub'].min()) / (
                    itog['change_last_week_usdrub'].max() - itog['change_last_week_usdrub'].min()))

    itog['ticker'] = 'LKOH'
    if str(stock_to_do) in ['SBER', 'IRAO', 'HYDR', 'FEES', 'MRKC', 'UPRO', 'SBER', 'SBERP', 'MOEX', 'VTBR', 'MGNT',
                            'RTKM', 'MTSS']:
        number=1312

        priznaki = ['change1_imoex',
                    'change2_imoex',
                    'change3_imoex',
                    'change4_imoex',
                    'change5_imoex',
                    'change6_imoex',
                    'change7_imoex',
                    'change8_imoex',
                    'change9_imoex',
                    'change10_imoex',
                    'change29_imoex',
                    'change22_imoex',
                    'change23_imoex',
                    'change24_imoex',
                    'change25_imoex',
                    'change1_usdrub',
                    'change2_usdrub',
                    'change3_usdrub',
                    'change4_usdrub',
                    'change5_usdrub',
                    'change6_usdrub',
                    'change7_usdrub',
                    'change8_usdrub',
                    'change9_usdrub',
                    'change10_usdrub',
                    'change_3days_usdrub',
                    'change_last_month_usdrub',
                    'change_last_3month_usdrub',
                    'change_last_half_of_year_usdrub',
                    'change_last_week_usdrub',
                    'change1_metalurgi',
                    'change2_metalurgi',
                    'change3_metalurgi',
                    'change4_metalurgi',
                    'change5_metalurgi',
                    'change6_metalurgi',
                    'change7_metalurgi',
                    'change8_metalurgi',
                    'change9_metalurgi',
                    'change10_metalurgi',
                    'change_3days_metalurgi',
                    'change_last_month_metalurgi',
                    'change_last_3month_metalurgi',
                    'change_last_half_of_year_metalurgi',
                    'change_last_week_metalurgi',
                    'change1_urals',
                    'change2_urals',
                    'change3_urals',
                    'change4_urals',
                    'change5_urals',
                    'change6_urals',
                    'change7_urals',
                    'change8_urals',
                    'change9_urals',
                    'change10_urals',
                    'change_3days_urals',
                    'change_last_month_urals',
                    'change_last_3month_urals',
                    'change_last_half_of_year_urals',
                    'change_last_week_urals',
                    'change1_zoloto',
                    'change2_zoloto',
                    'change3_zoloto',
                    'change4_zoloto',
                    'change5_zoloto',
                    'change6_zoloto',
                    'change7_zoloto',
                    'change8_zoloto',
                    'change9_zoloto',
                    'change10_zoloto',
                    'change_3days_zoloto',
                    'change_last_month_zoloto',
                    'change_last_3month_zoloto',
                    'change_last_half_of_year_zoloto',
                    'change_last_week_zoloto',
                    'change1_palladiy',
                    'change2_palladiy',
                    'change3_palladiy',
                    'change4_palladiy',
                    'change5_palladiy',
                    'change6_palladiy',
                    'change7_palladiy',
                    'change8_palladiy',
                    'change9_palladiy',
                    'change10_palladiy',
                    'change_3days_palladiy',
                    'change_last_month_palladiy',
                    'change_last_3month_palladiy',
                    'change_last_half_of_year_palladiy',
                    'change_last_week_palladiy',
                    'change1_platina',
                    'change2_platina',
                    'change3_platina',
                    'change4_platina',
                    'change5_platina',
                    'change6_platina',
                    'change7_platina',
                    'change8_platina',
                    'change9_platina',
                    'change10_platina',
                    'change_3days_platina',
                    'change_last_month_platina',
                    'change_last_3month_platina',
                    'change_last_half_of_year_platina',
                    'change_last_week_platina',
                    'change1_ugol',
                    'change2_ugol',
                    'change3_ugol',
                    'change4_ugol',
                    'change5_ugol',
                    'change6_ugol',
                    'change7_ugol',
                    'change8_ugol',
                    'change9_ugol',
                    'change10_ugol',
                    'change_3days_ugol',
                    'change_last_month_ugol',
                    'change_last_3month_ugol',
                    'change_last_half_of_year_ugol',
                    'change_last_week_ugol',
                    'change1_nikel',
                    'change2_nikel',
                    'change3_nikel',
                    'change4_nikel',
                    'change5_nikel',
                    'change6_nikel',
                    'change7_nikel',
                    'change8_nikel',
                    'change9_nikel',
                    'change10_nikel',
                    'change_3days_nikel',
                    'change_last_month_nikel',
                    'change_last_3month_nikel',
                    'change_last_half_of_year_nikel',
                    'change_last_week_nikel',
                    'change1_ghelezo',
                    'change2_ghelezo',
                    'change3_ghelezo',
                    'change4_ghelezo',
                    'change5_ghelezo',
                    'change6_ghelezo',
                    'change7_ghelezo',
                    'change8_ghelezo',
                    'change9_ghelezo',
                    'change10_ghelezo',
                    'change_3days_ghelezo',
                    'change_last_month_ghelezo',
                    'change_last_3month_ghelezo',
                    'change_last_half_of_year_ghelezo',
                    'change_last_week_ghelezo',
                    'change1_med',
                    'change2_med',
                    'change3_med',
                    'change4_med',
                    'change5_med',
                    'change6_med',
                    'change7_med',
                    'change8_med',
                    'change9_med',
                    'change10_med',
                    'change_3days_med',
                    'change_last_month_med',
                    'change_last_3month_med',
                    'change_last_half_of_year_med',
                    'change_last_week_med',
                    'change1_RUGBITR10Y',
                    'change2_RUGBITR10Y',
                    'change3_RUGBITR10Y',
                    'change4_RUGBITR10Y',
                    'change5_RUGBITR10Y',
                    'change6_RUGBITR10Y',
                    'change7_RUGBITR10Y',
                    'change8_RUGBITR10Y',
                    'change9_RUGBITR10Y',
                    'change10_RUGBITR10Y',
                    'change_3days_RUGBITR10Y',
                    'change_last_month_RUGBITR10Y',
                    'change_last_3month_RUGBITR10Y',
                    'change_last_half_of_year_RUGBITR10Y',
                    'change_last_week_RUGBITR10Y',
                    'change1_RUGBITR1Y',
                    'change2_RUGBITR1Y',
                    'change3_RUGBITR1Y',
                    'change4_RUGBITR1Y',
                    'change5_RUGBITR1Y',
                    'change6_RUGBITR1Y',
                    'change7_RUGBITR1Y',
                    'change8_RUGBITR1Y',
                    'change9_RUGBITR1Y',
                    'change10_RUGBITR1Y',
                    'change_3days_RUGBITR1Y',
                    'change_last_month_RUGBITR1Y',
                    'change_last_3month_RUGBITR1Y',
                    'change_last_half_of_year_RUGBITR1Y',
                    'change_last_week_RUGBITR1Y',
                    'change1_RGBITR',
                    'change2_RGBITR',
                    'change3_RGBITR',
                    'change4_RGBITR',
                    'change5_RGBITR',
                    'change6_RGBITR',
                    'change7_RGBITR',
                    'change8_RGBITR',
                    'change9_RGBITR',
                    'change10_RGBITR',
                    'change_3days_RGBITR',
                    'change_last_month_RGBITR',
                    'change_last_3month_RGBITR',
                    'change_last_half_of_year_RGBITR',
                    'change_last_week_RGBITR',
                    'doha_RUGBITR10Y',
                    'doha_RUGBITR1Y',
                    'change_doha_RUGBITR10Y',
                    'change_doha_RUGBITR1Y',
                    'change2_lkoh',
                    'change3_lkoh',
                    'change4_lkoh',
                    'change5_lkoh',
                    'change6_lkoh',
                    'change7_lkoh',
                    'change8_lkoh',
                    'change9_lkoh',
                    'change10_lkoh',
                    'change1_rosn',
                    'change2_rosn',
                    'change3_rosn',
                    'change4_rosn',
                    'change5_rosn',
                    'change6_rosn',
                    'change7_rosn',
                    'change8_rosn',
                    'change9_rosn',
                    'change10_rosn',
                    'change1_nvtk',
                    'change2_nvtk',
                    'change3_nvtk',
                    'change4_nvtk',
                    'change5_nvtk',
                    'change6_nvtk',
                    'change7_nvtk',
                    'change8_nvtk',
                    'change9_nvtk',
                    'change10_nvtk',
                    'change1_sngsp',
                    'change2_sngsp',
                    'change3_sngsp',
                    'change4_sngsp',
                    'change5_sngsp',
                    'change6_sngsp',
                    'change7_sngsp',
                    'change8_sngsp',
                    'change9_sngsp',
                    'change10_sngsp',
                    'change1_sngs',
                    'change2_sngs',
                    'change3_sngs',
                    'change4_sngs',
                    'change5_sngs',
                    'change6_sngs',
                    'change7_sngs',
                    'change8_sngs',
                    'change9_sngs',
                    'change10_sngs',
                    'change1_trnfp',
                    'change2_trnfp',
                    'change3_trnfp',
                    'change4_trnfp',
                    'change5_trnfp',
                    'change6_trnfp',
                    'change7_trnfp',
                    'change8_trnfp',
                    'change9_trnfp',
                    'change10_trnfp',
                    'change1_tatn',
                    'change2_tatn',
                    'change3_tatn',
                    'change4_tatn',
                    'change5_tatn',
                    'change6_tatn',
                    'change7_tatn',
                    'change8_tatn',
                    'change9_tatn',
                    'change10_tatn',
                    'change1_tatnp',
                    'change2_tatnp',
                    'change3_tatnp',
                    'change4_tatnp',
                    'change5_tatnp',
                    'change6_tatnp',
                    'change7_tatnp',
                    'change8_tatnp',
                    'change9_tatnp',
                    'change10_tatnp',
                    'change1_selg',
                    'change2_selg',
                    'change3_selg',
                    'change4_selg',
                    'change5_selg',
                    'change6_selg',
                    'change7_selg',
                    'change8_selg',
                    'change9_selg',
                    'change10_selg',
                    'change1_rasp',
                    'change2_rasp',
                    'change3_rasp',
                    'change4_rasp',
                    'change5_rasp',
                    'change6_rasp',
                    'change7_rasp',
                    'change8_rasp',
                    'change9_rasp',
                    'change10_rasp',
                    'change1_mtlrp',
                    'change2_mtlrp',
                    'change3_mtlrp',
                    'change4_mtlrp',
                    'change5_mtlrp',
                    'change6_mtlrp',
                    'change7_mtlrp',
                    'change8_mtlrp',
                    'change9_mtlrp',
                    'change10_mtlrp',
                    'change1_gmkn',
                    'change2_gmkn',
                    'change3_gmkn',
                    'change4_gmkn',
                    'change5_gmkn',
                    'change6_gmkn',
                    'change7_gmkn',
                    'change8_gmkn',
                    'change9_gmkn',
                    'change10_gmkn',
                    'change1_nlmk',
                    'change2_nlmk',
                    'change3_nlmk',
                    'change4_nlmk',
                    'change5_nlmk',
                    'change6_nlmk',
                    'change7_nlmk',
                    'change8_nlmk',
                    'change9_nlmk',
                    'change10_nlmk',
                    'change1_magn',
                    'change2_magn',
                    'change3_magn',
                    'change4_magn',
                    'change5_magn',
                    'change6_magn',
                    'change7_magn',
                    'change8_magn',
                    'change9_magn',
                    'change10_magn',
                    'change1_plzl',
                    'change2_plzl',
                    'change3_plzl',
                    'change4_plzl',
                    'change5_plzl',
                    'change6_plzl',
                    'change7_plzl',
                    'change8_plzl',
                    'change9_plzl',
                    'change10_plzl',
                    'change1_alrs',
                    'change2_alrs',
                    'change3_alrs',
                    'change4_alrs',
                    'change5_alrs',
                    'change6_alrs',
                    'change7_alrs',
                    'change8_alrs',
                    'change9_alrs',
                    'change10_alrs',
                    'change1_mtlr',
                    'change2_mtlr',
                    'change3_mtlr',
                    'change4_mtlr',
                    'change5_mtlr',
                    'change6_mtlr',
                    'change7_mtlr',
                    'change8_mtlr',
                    'change9_mtlr',
                    'change10_mtlr',
                    'change1_chmf',
                    'change2_chmf',
                    'change3_chmf',
                    'change4_chmf',
                    'change5_chmf',
                    'change6_chmf',
                    'change7_chmf',
                    'change8_chmf',
                    'change9_chmf',
                    'change10_chmf',
                    'change1_phor',
                    'change2_phor',
                    'change3_phor',
                    'change4_phor',
                    'change5_phor',
                    'change6_phor',
                    'change7_phor',
                    'change8_phor',
                    'change9_phor',
                    'change10_phor',
                    'change1_sber',
                    'change2_sber',
                    'change3_sber',
                    'change4_sber',
                    'change5_sber',
                    'change6_sber',
                    'change7_sber',
                    'change8_sber',
                    'change9_sber',
                    'change10_sber',
                    'change1_vtbr',
                    'change2_vtbr',
                    'change3_vtbr',
                    'change4_vtbr',
                    'change5_vtbr',
                    'change6_vtbr',
                    'change7_vtbr',
                    'change8_vtbr',
                    'change9_vtbr',
                    'change10_vtbr',

                    'change1_moex',
                    'change2_moex',
                    'change3_moex',
                    'change4_moex',
                    'change5_moex',
                    'change6_moex',
                    'change7_moex',
                    'change8_moex',
                    'change9_moex',
                    'change10_moex',

                    'change1_mtss',
                    'change2_mtss',
                    'change3_mtss',
                    'change4_mtss',
                    'change5_mtss',
                    'change6_mtss',
                    'change7_mtss',
                    'change8_mtss',
                    'change9_mtss',
                    'change10_mtss',
                    'change1_rtkm',
                    'change2_rtkm',
                    'change3_rtkm',
                    'change4_rtkm',
                    'change5_rtkm',
                    'change6_rtkm',
                    'change7_rtkm',
                    'change8_rtkm',
                    'change9_rtkm',
                    'change10_rtkm',
                    'change1_irao',
                    'change2_irao',
                    'change3_irao',
                    'change4_irao',
                    'change5_irao',
                    'change6_irao',
                    'change7_irao',
                    'change8_irao',
                    'change9_irao',
                    'change10_irao',
                    'change1_hydr',
                    'change2_hydr',
                    'change3_hydr',
                    'change4_hydr',
                    'change5_hydr',
                    'change6_hydr',
                    'change7_hydr',
                    'change8_hydr',
                    'change9_hydr',
                    'change10_hydr',
                    'change1_fees',
                    'change2_fees',
                    'change3_fees',
                    'change4_fees',
                    'change5_fees',
                    'change6_fees',
                    'change7_fees',
                    'change8_fees',
                    'change9_fees',
                    'change10_fees',
                    'change1_mrkc',
                    'change2_mrkc',
                    'change3_mrkc',
                    'change4_mrkc',
                    'change5_mrkc',
                    'change6_mrkc',
                    'change7_mrkc',
                    'change8_mrkc',
                    'change9_mrkc',
                    'change10_mrkc',
                    'change1_upro',
                    'change2_upro',
                    'change3_upro',
                    'change4_upro',
                    'change5_upro',
                    'change6_upro',
                    'change7_upro',
                    'change8_upro',
                    'change9_upro',
                    'change10_upro',
                    'volatility_3',
                    'volatility_7',
                    'volatility_10',
                    'volatility_15',
                    'volatility_30',
                    'change1_lkoh_val',
                    'change2_lkoh_val',
                    'change3_lkoh_val',
                    'change4_lkoh_val',
                    'change5_lkoh_val',
                    'change6_lkoh_val',
                    'change7_lkoh_val',
                    'change8_lkoh_val',
                    'change9_lkoh_val',
                    'change10_lkoh_val',
                    'change_last_month_lkoh',
                    'change_last_3month_lkoh',
                    'change_last_half_of_year_lkoh',
                    'change_last_week_lkoh',
                    'change_3days_lkoh',
                    'change12_lkoh',
                    'change13_lkoh',
                    'change14_lkoh',
                    'change15_lkoh',
                    'change16_lkoh',
                    'change17_lkoh',
                    'change18_lkoh',
                    'change19_lkoh',
                    'change110_lkoh']
    else:
        number=1501
        priznaki = ['change1_imoex',
                    'change2_imoex',
                    'change3_imoex',
                    'change4_imoex',
                    'change5_imoex',
                    'change6_imoex',
                    'change7_imoex',
                    'change8_imoex',
                    'change9_imoex',
                    'change10_imoex',
                    'change29_imoex',
                    'change22_imoex',
                    'change23_imoex',
                    'change24_imoex',
                    'change25_imoex',
                    'change1_usdrub',
                    'change2_usdrub',
                    'change3_usdrub',
                    'change4_usdrub',
                    'change5_usdrub',
                    'change6_usdrub',
                    'change7_usdrub',
                    'change8_usdrub',
                    'change9_usdrub',
                    'change10_usdrub',
                    'change_3days_usdrub',
                    'change_last_month_usdrub',
                    'change_last_3month_usdrub',
                    'change_last_half_of_year_usdrub',
                    'change_last_week_usdrub',
                    'change1_metalurgi',
                    'change2_metalurgi',
                    'change3_metalurgi',
                    'change4_metalurgi',
                    'change5_metalurgi',
                    'change6_metalurgi',
                    'change7_metalurgi',
                    'change8_metalurgi',
                    'change9_metalurgi',
                    'change10_metalurgi',
                    'change_3days_metalurgi',
                    'change_last_month_metalurgi',
                    'change_last_3month_metalurgi',
                    'change_last_half_of_year_metalurgi',
                    'change_last_week_metalurgi',
                    'change1_urals',
                    'change2_urals',
                    'change3_urals',
                    'change4_urals',
                    'change5_urals',
                    'change6_urals',
                    'change7_urals',
                    'change8_urals',
                    'change9_urals',
                    'change10_urals',
                    'change_3days_urals',
                    'change_last_month_urals',
                    'change_last_3month_urals',
                    'change_last_half_of_year_urals',
                    'change_last_week_urals',
                    'change1_zoloto',
                    'change2_zoloto',
                    'change3_zoloto',
                    'change4_zoloto',
                    'change5_zoloto',
                    'change6_zoloto',
                    'change7_zoloto',
                    'change8_zoloto',
                    'change9_zoloto',
                    'change10_zoloto',
                    'change_3days_zoloto',
                    'change_last_month_zoloto',
                    'change_last_3month_zoloto',
                    'change_last_half_of_year_zoloto',
                    'change_last_week_zoloto',
                    'change1_palladiy',
                    'change2_palladiy',
                    'change3_palladiy',
                    'change4_palladiy',
                    'change5_palladiy',
                    'change6_palladiy',
                    'change7_palladiy',
                    'change8_palladiy',
                    'change9_palladiy',
                    'change10_palladiy',
                    'change_3days_palladiy',
                    'change_last_month_palladiy',
                    'change_last_3month_palladiy',
                    'change_last_half_of_year_palladiy',
                    'change_last_week_palladiy',
                    'change1_platina',
                    'change2_platina',
                    'change3_platina',
                    'change4_platina',
                    'change5_platina',
                    'change6_platina',
                    'change7_platina',
                    'change8_platina',
                    'change9_platina',
                    'change10_platina',
                    'change_3days_platina',
                    'change_last_month_platina',
                    'change_last_3month_platina',
                    'change_last_half_of_year_platina',
                    'change_last_week_platina',
                    'change1_ugol',
                    'change2_ugol',
                    'change3_ugol',
                    'change4_ugol',
                    'change5_ugol',
                    'change6_ugol',
                    'change7_ugol',
                    'change8_ugol',
                    'change9_ugol',
                    'change10_ugol',
                    'change_3days_ugol',
                    'change_last_month_ugol',
                    'change_last_3month_ugol',
                    'change_last_half_of_year_ugol',
                    'change_last_week_ugol',
                    'change1_nikel',
                    'change2_nikel',
                    'change3_nikel',
                    'change4_nikel',
                    'change5_nikel',
                    'change6_nikel',
                    'change7_nikel',
                    'change8_nikel',
                    'change9_nikel',
                    'change10_nikel',
                    'change_3days_nikel',
                    'change_last_month_nikel',
                    'change_last_3month_nikel',
                    'change_last_half_of_year_nikel',
                    'change_last_week_nikel',
                    'change1_ghelezo',
                    'change2_ghelezo',
                    'change3_ghelezo',
                    'change4_ghelezo',
                    'change5_ghelezo',
                    'change6_ghelezo',
                    'change7_ghelezo',
                    'change8_ghelezo',
                    'change9_ghelezo',
                    'change10_ghelezo',
                    'change_3days_ghelezo',
                    'change_last_month_ghelezo',
                    'change_last_3month_ghelezo',
                    'change_last_half_of_year_ghelezo',
                    'change_last_week_ghelezo',
                    'change1_med',
                    'change2_med',
                    'change3_med',
                    'change4_med',
                    'change5_med',
                    'change6_med',
                    'change7_med',
                    'change8_med',
                    'change9_med',
                    'change10_med',
                    'change_3days_med',
                    'change_last_month_med',
                    'change_last_3month_med',
                    'change_last_half_of_year_med',
                    'change_last_week_med',
                    'change1_RUGBITR10Y',
                    'change2_RUGBITR10Y',
                    'change3_RUGBITR10Y',
                    'change4_RUGBITR10Y',
                    'change5_RUGBITR10Y',
                    'change6_RUGBITR10Y',
                    'change7_RUGBITR10Y',
                    'change8_RUGBITR10Y',
                    'change9_RUGBITR10Y',
                    'change10_RUGBITR10Y',
                    'change_3days_RUGBITR10Y',
                    'change_last_month_RUGBITR10Y',
                    'change_last_3month_RUGBITR10Y',
                    'change_last_half_of_year_RUGBITR10Y',
                    'change_last_week_RUGBITR10Y',
                    'change1_RUGBITR1Y',
                    'change2_RUGBITR1Y',
                    'change3_RUGBITR1Y',
                    'change4_RUGBITR1Y',
                    'change5_RUGBITR1Y',
                    'change6_RUGBITR1Y',
                    'change7_RUGBITR1Y',
                    'change8_RUGBITR1Y',
                    'change9_RUGBITR1Y',
                    'change10_RUGBITR1Y',
                    'change_3days_RUGBITR1Y',
                    'change_last_month_RUGBITR1Y',
                    'change_last_3month_RUGBITR1Y',
                    'change_last_half_of_year_RUGBITR1Y',
                    'change_last_week_RUGBITR1Y',
                    'change1_RGBITR',
                    'change2_RGBITR',
                    'change3_RGBITR',
                    'change4_RGBITR',
                    'change5_RGBITR',
                    'change6_RGBITR',
                    'change7_RGBITR',
                    'change8_RGBITR',
                    'change9_RGBITR',
                    'change10_RGBITR',
                    'change_3days_RGBITR',
                    'change_last_month_RGBITR',
                    'change_last_3month_RGBITR',
                    'change_last_half_of_year_RGBITR',
                    'change_last_week_RGBITR',
                    'doha_RUGBITR10Y',
                    'doha_RUGBITR1Y',
                    'change_doha_RUGBITR10Y',
                    'change_doha_RUGBITR1Y',
                    'change2_lkoh',
                    'change3_lkoh',
                    'change4_lkoh',
                    'change5_lkoh',
                    'change6_lkoh',
                    'change7_lkoh',
                    'change8_lkoh',
                    'change9_lkoh',
                    'change10_lkoh',
                    'change1_rosn',
                    'change2_rosn',
                    'change3_rosn',
                    'change4_rosn',
                    'change5_rosn',
                    'change6_rosn',
                    'change7_rosn',
                    'change8_rosn',
                    'change9_rosn',
                    'change10_rosn',
                    'change1_nvtk',
                    'change2_nvtk',
                    'change3_nvtk',
                    'change4_nvtk',
                    'change5_nvtk',
                    'change6_nvtk',
                    'change7_nvtk',
                    'change8_nvtk',
                    'change9_nvtk',
                    'change10_nvtk',
                    'change1_sngsp',
                    'change2_sngsp',
                    'change3_sngsp',
                    'change4_sngsp',
                    'change5_sngsp',
                    'change6_sngsp',
                    'change7_sngsp',
                    'change8_sngsp',
                    'change9_sngsp',
                    'change10_sngsp',
                    'change1_sngs',
                    'change2_sngs',
                    'change3_sngs',
                    'change4_sngs',
                    'change5_sngs',
                    'change6_sngs',
                    'change7_sngs',
                    'change8_sngs',
                    'change9_sngs',
                    'change10_sngs',
                    'change1_trnfp',
                    'change2_trnfp',
                    'change3_trnfp',
                    'change4_trnfp',
                    'change5_trnfp',
                    'change6_trnfp',
                    'change7_trnfp',
                    'change8_trnfp',
                    'change9_trnfp',
                    'change10_trnfp',
                    'change1_tatn',
                    'change2_tatn',
                    'change3_tatn',
                    'change4_tatn',
                    'change5_tatn',
                    'change6_tatn',
                    'change7_tatn',
                    'change8_tatn',
                    'change9_tatn',
                    'change10_tatn',
                    'change1_tatnp',
                    'change2_tatnp',
                    'change3_tatnp',
                    'change4_tatnp',
                    'change5_tatnp',
                    'change6_tatnp',
                    'change7_tatnp',
                    'change8_tatnp',
                    'change9_tatnp',
                    'change10_tatnp',
                    'change1_selg',
                    'change2_selg',
                    'change3_selg',
                    'change4_selg',
                    'change5_selg',
                    'change6_selg',
                    'change7_selg',
                    'change8_selg',
                    'change9_selg',
                    'change10_selg',
                    'change1_rasp',
                    'change2_rasp',
                    'change3_rasp',
                    'change4_rasp',
                    'change5_rasp',
                    'change6_rasp',
                    'change7_rasp',
                    'change8_rasp',
                    'change9_rasp',
                    'change10_rasp',
                    'change1_mtlrp',
                    'change2_mtlrp',
                    'change3_mtlrp',
                    'change4_mtlrp',
                    'change5_mtlrp',
                    'change6_mtlrp',
                    'change7_mtlrp',
                    'change8_mtlrp',
                    'change9_mtlrp',
                    'change10_mtlrp',
                    'change1_gmkn',
                    'change2_gmkn',
                    'change3_gmkn',
                    'change4_gmkn',
                    'change5_gmkn',
                    'change6_gmkn',
                    'change7_gmkn',
                    'change8_gmkn',
                    'change9_gmkn',
                    'change10_gmkn',
                    'change1_nlmk',
                    'change2_nlmk',
                    'change3_nlmk',
                    'change4_nlmk',
                    'change5_nlmk',
                    'change6_nlmk',
                    'change7_nlmk',
                    'change8_nlmk',
                    'change9_nlmk',
                    'change10_nlmk',
                    'change1_magn',
                    'change2_magn',
                    'change3_magn',
                    'change4_magn',
                    'change5_magn',
                    'change6_magn',
                    'change7_magn',
                    'change8_magn',
                    'change9_magn',
                    'change10_magn',
                    'change1_plzl',
                    'change2_plzl',
                    'change3_plzl',
                    'change4_plzl',
                    'change5_plzl',
                    'change6_plzl',
                    'change7_plzl',
                    'change8_plzl',
                    'change9_plzl',
                    'change10_plzl',
                    'change1_alrs',
                    'change2_alrs',
                    'change3_alrs',
                    'change4_alrs',
                    'change5_alrs',
                    'change6_alrs',
                    'change7_alrs',
                    'change8_alrs',
                    'change9_alrs',
                    'change10_alrs',
                    'change1_mtlr',
                    'change2_mtlr',
                    'change3_mtlr',
                    'change4_mtlr',
                    'change5_mtlr',
                    'change6_mtlr',
                    'change7_mtlr',
                    'change8_mtlr',
                    'change9_mtlr',
                    'change10_mtlr',
                    'change1_chmf',
                    'change2_chmf',
                    'change3_chmf',
                    'change4_chmf',
                    'change5_chmf',
                    'change6_chmf',
                    'change7_chmf',
                    'change8_chmf',
                    'change9_chmf',
                    'change10_chmf',
                    'change1_phor',
                    'change2_phor',
                    'change3_phor',
                    'change4_phor',
                    'change5_phor',
                    'change6_phor',
                    'change7_phor',
                    'change8_phor',
                    'change9_phor',
                    'change10_phor',
                    'change1_sber',
                    'change2_sber',
                    'change3_sber',
                    'change4_sber',
                    'change5_sber',
                    'change6_sber',
                    'change7_sber',
                    'change8_sber',
                    'change9_sber',
                    'change10_sber',
                    'change1_vtbr',
                    'change2_vtbr',
                    'change3_vtbr',
                    'change4_vtbr',
                    'change5_vtbr',
                    'change6_vtbr',
                    'change7_vtbr',
                    'change8_vtbr',
                    'change9_vtbr',
                    'change10_vtbr',

                    'change1_mtss',
                    'change2_mtss',
                    'change3_mtss',
                    'change4_mtss',
                    'change5_mtss',
                    'change6_mtss',
                    'change7_mtss',
                    'change8_mtss',
                    'change9_mtss',
                    'change10_mtss',
                    'change1_rtkm',
                    'change2_rtkm',
                    'change3_rtkm',
                    'change4_rtkm',
                    'change5_rtkm',
                    'change6_rtkm',
                    'change7_rtkm',
                    'change8_rtkm',
                    'change9_rtkm',
                    'change10_rtkm',
                    'change1_irao',
                    'change2_irao',
                    'change3_irao',
                    'change4_irao',
                    'change5_irao',
                    'change6_irao',
                    'change7_irao',
                    'change8_irao',
                    'change9_irao',
                    'change10_irao',
                    'change1_hydr',
                    'change2_hydr',
                    'change3_hydr',
                    'change4_hydr',
                    'change5_hydr',
                    'change6_hydr',
                    'change7_hydr',
                    'change8_hydr',
                    'change9_hydr',
                    'change10_hydr',
                    'change1_fees',
                    'change2_fees',
                    'change3_fees',
                    'change4_fees',
                    'change5_fees',
                    'change6_fees',
                    'change7_fees',
                    'change8_fees',
                    'change9_fees',
                    'change10_fees',
                    'change1_mrkc',
                    'change2_mrkc',
                    'change3_mrkc',
                    'change4_mrkc',
                    'change5_mrkc',
                    'change6_mrkc',
                    'change7_mrkc',
                    'change8_mrkc',
                    'change9_mrkc',
                    'change10_mrkc',
                    'change1_upro',
                    'change2_upro',
                    'change3_upro',
                    'change4_upro',
                    'change5_upro',
                    'change6_upro',
                    'change7_upro',
                    'change8_upro',
                    'change9_upro',
                    'change10_upro',
                    'volatility_3',
                    'volatility_7',
                    'volatility_10',
                    'volatility_15',
                    'volatility_30',
                    'change1_lkoh_val',
                    'change2_lkoh_val',
                    'change3_lkoh_val',
                    'change4_lkoh_val',
                    'change5_lkoh_val',
                    'change6_lkoh_val',
                    'change7_lkoh_val',
                    'change8_lkoh_val',
                    'change9_lkoh_val',
                    'change10_lkoh_val',
                    'change_last_month_lkoh',
                    'change_last_3month_lkoh',
                    'change_last_half_of_year_lkoh',
                    'change_last_week_lkoh',
                    'change_3days_lkoh',
                    'change12_lkoh',
                    'change13_lkoh',
                    'change14_lkoh',
                    'change15_lkoh',
                    'change16_lkoh',
                    'change17_lkoh',
                    'change18_lkoh',
                    'change19_lkoh',
                    'change110_lkoh']

    def ML(itog, priznaki, num):


        itog_test = itog.iloc[num:]
        itog = itog.iloc[:num-1]

        # выбор целевой переменной
        data = itog

        # Определите признаки (X) и целевую переменную (Y)
        X = data[priznaki]
        y = data['change1_lkoh']

        # Создайте объект SimpleImputer для заполнения NaN средними значениями
        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте признаки X, чтобы заполнить отсутствующие значения
        X_filled = imputer.fit_transform(X)

        # Создайте объект SimpleImputer для заполнения NaN средними значениями
        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте целевую переменную y, чтобы заполнить отсутствующие значения
        y_filled = imputer.fit_transform(y.values.reshape(-1, 1))

        # Разделите данные на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X_filled, y_filled, test_size=0.1, random_state=0)

        # Создайте модель градиентного бустинга на основе деревьев
        model = lgb.LGBMRegressor()

        # Определение параметров для перебора
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8]

        }

        tscv = TimeSeriesSplit(n_splits=3)  # Задаем количество разбиений

        # Используем TimeSeriesSplit в GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')

        # Подгонка модели к данным
        grid_search.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Определите признаки (X) и целевую переменную (Y)
        best_model = grid_search.best_estimator_
        y_pred1 = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred1)

        # Вывести MSE на экран
        print("MSE:", mse)
        feature_importance = best_model.feature_importances_

        # Создание DataFrame с признаками и их важностью
        importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})

        # Сортировка по важности

        importance_df = importance_df.sort_values(by='importance', ascending=False)
        top15 = []
        for index, row in importance_df.head(15).iterrows():
            top15.append(row['feature'])

        X = data[top15]
        y = data['change1_lkoh']

        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте признаки X, чтобы заполнить отсутствующие значения
        X_filled = imputer.fit_transform(X)

        # Создайте объект SimpleImputer для заполнения NaN средними значениями
        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте целевую переменную y, чтобы заполнить отсутствующие значения
        y_filled = imputer.fit_transform(y.values.reshape(-1, 1))

        # Разделите данные на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X_filled, y_filled, test_size=0.1, random_state=0)

        # Создайте модель градиентного бустинга на основе деревьев
        model = lgb.LGBMRegressor(**grid_search.best_params_)

        model.fit(X_train, y_train)

        X = itog_test[top15]
        y = itog_test['change1_lkoh']

        # Создайте объект SimpleImputer для заполнения NaN средними значениями
        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте признаки X, чтобы заполнить отсутствующие значения
        X_filled = imputer.fit_transform(X)

        # Создайте объект SimpleImputer для заполнения NaN средними значениями
        imputer = SimpleImputer(strategy='mean')

        # Преобразуйте целевую переменную y, чтобы заполнить отсутствующие значения
        y_filled = imputer.fit_transform(y.values.reshape(-1, 1))

        # Предскажите значения на тестовом наборе
        y_pred_test = model.predict(X)
        y_actual = y_filled.flatten()

        # Получите список дат из индекса itog_test
        dates = []
        ticker = []
        for index, row in itog_test.iterrows():
            dates.append(index)
            ticker.append(row['ticker'])
        compare = pd.DataFrame()
        ans_pred = []
        ans = []
        for i in itog_test['change1_lkoh']:
            ans.append((np.exp(i) - 1) * (itog1['change1_lkoh'].max() - itog1['change1_lkoh'].min()) + itog1[
                'change1_lkoh'].min())
        for i in y_pred_test:
            ans_pred.append((np.exp(i) - 1) * (itog1['change1_lkoh'].max() - itog1['change1_lkoh'].min()) + itog1[
                'change1_lkoh'].min())

        compare['y_pred_test'] = ans_pred
        compare['y_actual'] = ans
        compare['dates'] = dates
        compare['y_diff'] = compare['y_pred_test'] - compare['y_actual']
        compare['ticker'] = ticker
        compare.set_index('dates', inplace=True)

        return compare

    table = ML(itog, priznaki, number)
    table.to_csv('stocks/'+'GAZP.csv')
    table = pd.read_csv('stocks/' +'GAZP.csv')

    table['y_diff'] = table['y_pred_test'] - table['y_actual']
    table.set_index('dates', inplace=True)

    #замечание о том, что изменение 24 февраля равно реальному
    table.loc['2020-03-10', 'y_pred_test'] = table.loc['2020-03-10', 'y_actual']
    table.loc['2022-02-24', 'y_pred_test'] = table.loc['2022-02-24', 'y_actual']

    #составление индексов
    table['index_real_lkoh_pred'] = 100
    table['index_real_lkoh'] = 100
    table['dates'] = table.index
    table.reset_index(drop=True, inplace=True)
    prev_index_lkoh_real = None
    prev_index_real_lkoh_pred = None

    for index, row in table.iterrows():
        if prev_index_real_lkoh_pred is not None and prev_index_lkoh_real is not None:
            row['index_real_lkoh'] = prev_index_lkoh_real
            row['index_real_lkoh_pred'] = prev_index_real_lkoh_pred

        table.loc[index, 'index_real_lkoh'] = row['index_real_lkoh'] * (1 + (row['y_actual'] / 100))
        table.loc[index, 'index_real_lkoh_pred'] = row['index_real_lkoh_pred'] * (1 + row['y_pred_test'] / 100)

        prev_index_real_lkoh_pred = table.loc[index, 'index_real_lkoh_pred']
        prev_index_lkoh_real = table.loc[index, 'index_real_lkoh']

    #график логарифмической разницы
    table['y_diff1'] = np.log1p(table['index_real_lkoh'] / table['index_real_lkoh_pred'])

    last=len(table)-1

    #построение z-score
    z_scores = []
    table['dates'] = pd.to_datetime(table['dates'])
    # Предположим, у вас есть массив данных data
    for i in range(91, last):
        y_diff1_values = -table.iloc[i - 90:i - 1]['y_diff1']
        mean_value = np.mean(y_diff1_values)
        std_value = np.std(y_diff1_values)
        z_score = (-table.loc[i, 'y_diff1'] - mean_value) / std_value
        z_scores.append(z_score)

    table['ticker'] = str(stock_to_do)
    table = table.iloc[91:last]
    table['result'] = z_scores

    table.to_csv('stocks/'+str(stock_to_do)+'.csv')
    print(table)

resh('LKOH')
resh('ROSN')
resh('NVTK')
resh('SNGS')
resh('SNGSP')
resh('TRNFP')
resh('TATN')
resh('TATNP')

resh('GMKN')
resh('NLMK')
resh('MAGN')
resh('MTLR')
resh('CHMF')
resh('MTLRP')

resh('MGNT')
resh('MTSS')
resh('RTKM')
resh('SBER')
resh('SBERP')
resh('MOEX')
resh('VTBR')

resh('IRAO')
resh('HYDR')
resh('FEES')
resh('MRKC')
resh('UPRO')



