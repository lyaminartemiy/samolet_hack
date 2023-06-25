!pip install tsfresh

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings
import xgboost as xgb
import pickle


# Функция для распределения объема закупок
def get_minimum_spend(data):
    values = data.iloc[:, 1].astype(float)
    markers = []
    pos = 0
    total = 0
    while pos < len(values):
        temp = np.diff(values[pos:pos+10])
        temp = np.insert(temp, 0, np.nan)
        temp[1:] = np.cumsum(temp[1:])
        counter = 0
        for counter in range(len(temp)):
            counter += 1
            if pos + counter == len(values):
                break
            if counter == len(temp):
                break
            if temp[counter] < 0:
                break
        markers.append(counter)
        markers.extend([0 for _ in range(counter-1)])
        total += counter * values[pos]
        pos += counter
    return(markers, total)


def function_for_forecast(date: str, path_to_model=['/data/xgb_reg1.pkl', 
                                                    '/data/xgb_reg1.pkl', 
                                                    '/data/xgb_reg1.pkl'], 
                          path_to_features='/data/train.xlsx', 
                          path_to_test='/data/test.xlsx'):

  train = pd.read_excel(path_to_features)
  test = pd.read_excel(path_to_test)

  model0 = pickle.load(open(path_to_model[0], "rb"))
  model1 = pickle.load(open(path_to_model[1], "rb"))
  model2 = pickle.load(open(path_to_model[2], "rb"))
                            
  # Подготовка тренировочной выборки
  df_train = pd.DataFrame()
  df_train['Date'] = pd.to_datetime(train['dt'])
  df_train['Price'] = train['Цена на арматуру']
  df_train['istest'] = 0
  
  # Подготовка тестовой выборки
  df_test = pd.DataFrame()
  df_test['Date'] = pd.to_datetime(test['dt'])
  df_test['Price'] = test['Цена на арматуру']
  df_test['istest'] = 1
  
  # Конкатинируем в один датасет с метрой `istest`
  raw = pd.concat((df_train, df_test)).reset_index(drop=True)
  
  # Выделяем временные признаки
  raw["day_sin"] = np.sin(raw["Date"].dt.day)
  raw["day_cos"] = np.cos(raw["Date"].dt.day)
  
  # Создадим матрицы со сдвигом, то есть сделаем так, чтобы значение
  # цены на следующую неделю являлось "неизвестным для модели"
  X = raw.drop(columns=['Price']).iloc[:-1, :]
  y = raw['Price'].values[1:]
  
  X['Price_target'] = y
  X['Price_source'] = raw['Price'].values[:-1]
  
  # Подсчитаем разницу между текущим и прошлым значением цены на покупку арматуры
  price_diff = X['Price_source'] - X['Price_target']
  
  # Удаляем первую строку
  X = X.drop([0])
  
  # Добавляем разницу в цене
  X['Price_Diff'] = price_diff.values[:-1]
  
  # Удаление ненужных столбцов
  y = X['Price_target']
  X = X.drop(columns=['Price_target', 'Date'])

  # Используем следующий набор параметров
  
  # Рассчитываются коэффициенты skewness и kurtosis, считается, сколько раз
  # повторялось значение минимума и максимума, различные квантили, оконные
  # статистики, автокорреляции и т. д.
  settings_efficient = settings.EfficientFCParameters()
  len(settings_efficient)
  
  # Подготовка фреймов длины 5
  data_fot_tsfresh = X['Price_source'].values
  indexes = list(range(len(data_fot_tsfresh), 4, -1))
  
  ts_for_tsfresh = []
  
  for i in range(len(indexes)):
      indexes_ts = indexes[i:i+6][::-1]
      values_ts = data_fot_tsfresh[indexes_ts[0]:indexes_ts[-1]]
      # print(values_ts)
      if len(values_ts) < 5:
          break
      ts_for_tsfresh.append(values_ts)
  
  ts_for_tsfresh = pd.DataFrame(ts_for_tsfresh[::-1])
  
  # Создание фрейма необходимого формата для выделения признаков
  data_long = pd.DataFrame({'data': ts_for_tsfresh.values.flatten(),
                'id': np.repeat(np.array(ts_for_tsfresh.index), 5)})
  
  # Генерация признаков
  X_tsfresh = extract_features(data_long, column_id='id', impute_function=impute, default_fc_parameters=settings_efficient)
  
  # Удаление пустых строк
  X = X.drop(list(range(1, 10)))
  y = y.drop(list(range(1, 10)))
  
  # Reset индексов
  X = X.reset_index(drop=True)
  y = y.reset_index(drop=True)
  
  # Добавляем выделенные признаки в тренировочную и тестовую выборки
  for col in X_tsfresh.columns:
      X[col] = X_tsfresh[col]

  data_for_test = X[X['istest'] == 1]
  data_for_test = data_for_test.drop(columns=['istest'])
  
  ypred = (model0.predict(data_for_test) +
           model1.predict(data_for_test) +
           model2.predict(data_for_test)) / 3

  result = pd.DataFrame({'real': y[X['istest'] == 1][1:],
                'pred': ypred[:-1]})
  result['error'] = result['real'] - result['pred']
  source_price = test['Цена на арматуру']
  predictions_costs = pd.DataFrame(np.array(list(y[X['istest'] == 0].tail(1).values) + list(ypred)))
  test['Цена на арматуру'] = predictions_costs

  # Потраченная сумма на арматуру 
  markers, total = get_minimum_spend(test)

  test['Объем'] = markers
  
  test['Цена на арматуру'] = source_price
  return test[test['dt'] == date]['Объем'].values
  
